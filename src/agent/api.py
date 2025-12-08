"""
FastAPI backend for the Crypto Tracer Agent.
Provides a conversational chat interface that collects information before tracing.
"""
import asyncio
import json
import logging
import re
import uuid
from typing import Optional, AsyncGenerator, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import gen_trace_id, trace
from agents.mcp import MCPServerStdio
from agent.models import TracerConfig, TraceResult
from agent.mcp_client import MCPClient
from agent.tracer import CryptoTracer
from agent.reporting import build_report
from agent.theft_detection import parse_case_description_with_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Continuation option for interactive tracing
class ContinuationOption(BaseModel):
    tx_hash: str
    to_address: str
    amount: float
    asset: str
    time: Optional[str] = None
    description: str


# Session state for multi-turn conversations
class SessionState(BaseModel):
    session_id: str
    step: str = "initial"  # initial, collecting, confirming, tracing, trace_complete, awaiting_continuation
    collected_info: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    # Trace state for continuation
    last_trace_result: Optional[Dict[str, Any]] = None
    continuation_point: Optional[Dict[str, Any]] = None  # {address, blockchain, asset, token_id}
    continuation_options: List[Dict[str, Any]] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class TraceRequest(BaseModel):
    description: Optional[str] = None
    victim_address: Optional[str] = None
    blockchain: str = "eth"
    asset: Optional[str] = None
    date: Optional[str] = None
    tx_hashes: Optional[list[str]] = None
    tx_hash: Optional[str] = None
    theft_asset: Optional[str] = None


# In-memory session storage (use Redis in production)
sessions: Dict[str, SessionState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Crypto Tracer API...")
    yield
    logger.info("Shutting down Crypto Tracer API...")


app = FastAPI(
    title="Crypto Tracer Agent API",
    description="API for the AMLBot Crypto Tracing Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fast_parse_input(message: str) -> Dict[str, Any]:
    """
    Fast regex-based parsing for simple inputs like addresses and tx hashes.
    Avoids LLM call for simple cases.
    """
    result: Dict[str, Any] = {}

    # Ethereum address pattern (0x + 40 hex chars)
    eth_address = re.search(r'\b(0x[a-fA-F0-9]{40})\b', message)
    if eth_address:
        result["victim_address"] = eth_address.group(1)
        result["blockchain_name"] = "eth"

    # Tron address pattern (T + 33 chars)
    tron_address = re.search(r'\b(T[a-zA-Z0-9]{33})\b', message)
    if tron_address:
        result["victim_address"] = tron_address.group(1)
        result["blockchain_name"] = "trx"

    # Bitcoin address patterns
    btc_address = re.search(r'\b(bc1[a-zA-Z0-9]{39,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b', message)
    if btc_address:
        result["victim_address"] = btc_address.group(1)
        result["blockchain_name"] = "btc"

    # Ethereum tx hash (0x + 64 hex chars)
    eth_tx = re.search(r'\b(0x[a-fA-F0-9]{64})\b', message)
    if eth_tx:
        result["tx_hash"] = eth_tx.group(1)
        if "blockchain_name" not in result:
            result["blockchain_name"] = "eth"

    # Tron tx hash (64 hex chars without 0x)
    tron_tx = re.search(r'\b([a-fA-F0-9]{64})\b', message)
    if tron_tx and "tx_hash" not in result:
        # Check if it's likely a Tron tx (if we already detected Tron address)
        if result.get("blockchain_name") == "trx":
            result["tx_hash"] = tron_tx.group(1)
        # Or if message mentions tron/trx
        elif re.search(r'\b(tron|trx)\b', message, re.IGNORECASE):
            result["tx_hash"] = tron_tx.group(1)
            result["blockchain_name"] = "trx"

    # Detect blockchain from keywords
    if "blockchain_name" not in result:
        if re.search(r'\b(ethereum|eth)\b', message, re.IGNORECASE):
            result["blockchain_name"] = "eth"
        elif re.search(r'\b(tron|trx)\b', message, re.IGNORECASE):
            result["blockchain_name"] = "trx"
        elif re.search(r'\b(bitcoin|btc)\b', message, re.IGNORECASE):
            result["blockchain_name"] = "btc"
        elif re.search(r'\b(polygon|matic|poly)\b', message, re.IGNORECASE):
            result["blockchain_name"] = "poly"
        elif re.search(r'\b(bsc|binance)\b', message, re.IGNORECASE):
            result["blockchain_name"] = "bsc"

    # Detect asset from keywords
    asset_match = re.search(r'\b(USDT|USDC|ETH|BTC|TRX|BNB|MATIC)\b', message, re.IGNORECASE)
    if asset_match:
        result["asset_symbol"] = asset_match.group(1).upper()
        result["theft_asset"] = result["asset_symbol"]

    # Detect date patterns
    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', message)
    if date_match:
        result["approx_date"] = date_match.group(1)

    return result


def get_or_create_session(session_id: Optional[str]) -> SessionState:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    session = SessionState(session_id=new_id)
    sessions[new_id] = session
    return session


def format_collected_info(info: Dict[str, Any]) -> str:
    """Format collected information for display."""
    lines = []

    field_labels = {
        "victim_address": "📍 Victim Address",
        "tx_hash": "🔗 Transaction Hash",
        "blockchain_name": "⛓️ Blockchain",
        "asset_symbol": "💰 Asset",
        "theft_asset": "💰 Stolen Asset",
        "approx_date": "📅 Approximate Date",
        "known_tx_hashes": "🔗 Additional Tx Hashes",
        "description": "📝 Description",
    }

    for key, label in field_labels.items():
        value = info.get(key)
        if value:
            if key == "known_tx_hashes" and isinstance(value, list):
                if value:
                    lines.append(f"- {label}: {', '.join(value)}")
            elif key == "description":
                # Truncate long descriptions
                desc = value[:100] + "..." if len(value) > 100 else value
                lines.append(f"- {label}: {desc}")
            else:
                lines.append(f"- {label}: `{value}`")

    return "\n".join(lines) if lines else "No information collected yet."


def get_missing_required_fields(info: Dict[str, Any]) -> List[str]:
    """Check what required fields are still missing."""
    missing = []

    # Must have either victim_address or tx_hash
    has_address = bool(info.get("victim_address"))
    has_tx = bool(info.get("tx_hash"))

    if not has_address and not has_tx:
        missing.append("victim_address or tx_hash")

    # Blockchain is required
    if not info.get("blockchain_name"):
        missing.append("blockchain")

    # For tx_hash mode, theft_asset is recommended
    if has_tx and not has_address and not info.get("theft_asset") and not info.get("asset_symbol"):
        missing.append("theft_asset (recommended for transaction mode)")

    return missing


def build_clarification_message(info: Dict[str, Any], missing: List[str]) -> str:
    """Build a message asking for missing information."""
    current = format_collected_info(info)

    msg = "## 📋 Information Collected\n\n"
    msg += current + "\n\n"

    if missing:
        msg += "## ❓ Still Needed\n\n"
        for field in missing:
            if "victim_address or tx_hash" in field:
                msg += "- **Victim wallet address** OR **Theft transaction hash**\n"
            elif "blockchain" in field:
                msg += "- **Blockchain** (eth, trx, btc, bsc, poly, etc.)\n"
            elif "theft_asset" in field:
                msg += "- **Stolen asset symbol** (e.g., USDT, ETH, BTC) - helpful for accurate tracing\n"

        msg += "\nPlease provide the missing information."

    return msg


def build_continuation_message(session: SessionState) -> Dict[str, Any]:
    """Build a message offering continuation options."""
    options = session.continuation_options

    if not options:
        return {
            "type": "message",
            "message": "✅ Trace complete. No further continuation points available.",
            "session_id": session.session_id
        }

    msg = "## 🔄 Trace Paused - Continuation Options\n\n"
    msg += "The trace has reached endpoints that could be explored further.\n\n"
    msg += "### Available Options:\n\n"

    for i, opt in enumerate(options, 1):
        addr_short = f"{opt['address'][:10]}...{opt['address'][-6:]}"
        risk_info = f" ⚠️ Risk: {opt['risk_score']:.2f}" if opt.get('risk_score') and opt['risk_score'] > 0.5 else ""
        msg += f"**{i}.** `{addr_short}`{risk_info}\n"
        msg += f"   - Role: {opt.get('role', 'unknown')}\n"
        msg += f"   - Last amount: {opt['last_amount']:,.2f} {opt['asset']}\n"
        msg += f"   - Reason stopped: {opt.get('stop_reason', 'N/A')}\n\n"

    msg += "---\n\n"
    msg += "**What would you like to do?**\n"
    msg += "- Type **'continue 1'** (or 2, 3...) to trace from that address\n"
    msg += "- Paste a **transaction hash** to trace from a specific tx\n"
    msg += "- Type **'done'** to finish the trace\n"

    return {
        "type": "continuation",
        "message": msg,
        "session_id": session.session_id,
        "continuation_options": options
    }


def build_confirmation_message(info: Dict[str, Any]) -> str:
    """Build a confirmation message before starting trace."""
    current = format_collected_info(info)

    mode = "Transaction Hash" if info.get("tx_hash") else "Wallet Address"

    msg = "## ✅ Ready to Trace\n\n"
    msg += f"**Mode:** {mode}\n\n"
    msg += "### Collected Information\n\n"
    msg += current + "\n\n"
    msg += "---\n\n"
    msg += "**Would you like to start the trace?**\n\n"
    msg += "- Type **'yes'**, **'start'**, or **'trace'** to begin\n"
    msg += "- Type **'edit'** to modify the information\n"
    msg += "- Or provide additional details to update"

    return msg


async def run_trace_streaming(
    config: TracerConfig,
    session: Optional[SessionState] = None
) -> AsyncGenerator[str, None]:
    """Run a trace and yield streaming updates."""

    yield json.dumps({"type": "status", "message": "Initializing MCP server..."}) + "\n"

    async with MCPServerStdio(
        name="AMLBot MCP Server",
        params={
            "command": "docker",
            "args": ["run", "-i", "--rm", "-e", "USER_ID=a2fa961b7f4977981e3796916328d930", "mcp-server-amlbot:local"]
        },
        client_session_timeout_seconds=300.0,
    ) as server:
        trace_id = gen_trace_id()
        yield json.dumps({
            "type": "trace_started",
            "trace_id": trace_id,
            "trace_url": f"https://platform.openai.com/traces/trace?trace_id={trace_id}"
        }) + "\n"

        with trace(workflow_name="Crypto Tracer Agent", trace_id=trace_id):
            client = MCPClient(server)
            tracer = CryptoTracer(client)

            yield json.dumps({"type": "status", "message": "Running trace analysis..."}) + "\n"

            try:
                result = await tracer.trace(config)

                yield json.dumps({"type": "status", "message": "Building report..."}) + "\n"
                report = build_report(result)

                # Extract continuation options - only when user decision is needed
                continuation_options = extract_continuation_options(result, client, config)

                # Store trace result in session for continuation
                if session:
                    session.last_trace_result = report
                    session.continuation_options = continuation_options
                    if continuation_options:
                        session.step = "awaiting_continuation"
                    else:
                        session.step = "trace_complete"

                # Only include continuation options if there are any
                response_data = {
                    "type": "result",
                    "report": report,
                    "trace_id": trace_id,
                }

                if continuation_options:
                    response_data["continuation_options"] = continuation_options
                    response_data["can_continue"] = True
                else:
                    response_data["can_continue"] = False

                yield json.dumps(response_data) + "\n"

            except Exception as e:
                logger.error(f"Trace error: {e}")
                yield json.dumps({
                    "type": "error",
                    "message": str(e)
                }) + "\n"


def extract_continuation_options(
    result: TraceResult,
    client: MCPClient,
    config: TracerConfig
) -> List[Dict[str, Any]]:
    """
    Extract potential continuation points from trace result.
    Only returns options when user decision is genuinely needed:
    - When trace hit a confirmed terminal (CEX, mixer, etc.) - user may want to investigate further
    - When there's ambiguity the agent couldn't resolve

    The agent should have already traced as far as possible automatically.
    """
    options = []

    # Find endpoints (last addresses in each path)
    for path in result.paths:
        if not path.steps:
            continue

        last_step = path.steps[-1]
        last_address = last_step.to_address

        # Find entity info for this address
        entity = None
        for e in result.entities:
            if e.address == last_address:
                entity = e
                break

        # Check stop reason - only offer continuation for specific cases
        stop_reason = (path.stop_reason or "").lower()

        # Cases where we offer continuation options:
        # 1. Hit a CEX/exchange - user might want to get withdrawal info
        # 2. Hit a bridge - user might want to trace on destination chain
        # 3. Explicit "needs user decision" in stop reason
        # 4. Multiple branches that agent couldn't auto-resolve

        needs_user_decision = (
            "user" in stop_reason or
            "decision" in stop_reason or
            "ambiguous" in stop_reason or
            "multiple" in stop_reason or
            "choose" in stop_reason
        )

        is_bridge = entity and entity.role == "bridge_service"
        is_cex = entity and entity.role == "cex_deposit"

        # Only offer continuation for bridges (cross-chain) or when explicitly needed
        if is_bridge or needs_user_decision:
            option = {
                "address": last_address,
                "path_id": path.path_id,
                "last_amount": last_step.amount_estimate,
                "asset": last_step.asset,
                "chain": last_step.chain,
                "last_tx_hash": last_step.tx_hash,
                "role": entity.role if entity else "unknown",
                "risk_score": entity.risk_score if entity else None,
                "stop_reason": path.stop_reason or "Path ended",
                "description": f"Continue from {last_address[:8]}...{last_address[-6:]} ({entity.role if entity else 'unknown'})"
            }
            options.append(option)

    return options


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "crypto-tracer-api"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handle chat messages with multi-turn conversation flow.
    Collects information before starting a trace.
    """
    message = request.message.strip()
    logger.info(f"Chat request received: {message[:100]}...")

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Get or create session
    session = get_or_create_session(request.session_id)
    session_id = session.session_id
    logger.info(f"Session: {session_id}, step: {session.step}")

    # Check for confirmation triggers
    message_lower = message.lower().strip()
    is_confirmation = message_lower in ["yes", "y", "start", "trace", "go", "begin", "ok", "confirm"]
    is_edit_request = message_lower in ["edit", "change", "modify", "update"]
    is_reset = message_lower in ["reset", "clear", "new", "restart"]

    # Handle reset
    if is_reset:
        session.collected_info = {}
        session.step = "initial"
        logger.info("Session reset")
        return {
            "type": "message",
            "message": "🔄 Session cleared. Let's start fresh!\n\nPlease describe your case or provide:\n- Victim wallet address\n- Transaction hash\n- Blockchain (eth, trx, btc, etc.)",
            "session_id": session_id,
            "collected_info": {}
        }

    # Handle edit request
    if is_edit_request and session.step == "confirming":
        session.step = "collecting"
        logger.info("Edit mode activated")
        return {
            "type": "message",
            "message": "📝 What would you like to change?\n\n" + format_collected_info(session.collected_info),
            "session_id": session_id,
            "collected_info": session.collected_info
        }

    # Handle continuation from previous trace
    if session.step == "awaiting_continuation":
        # Check if user wants to continue from a specific address
        continue_match = re.search(r'continue\s+(?:from\s+)?(\d+|[a-fA-F0-9x]+)', message_lower)
        if continue_match or message_lower in ["continue", "next", "more"]:
            # Find which option to continue from
            option_index = 0
            if continue_match:
                match_val = continue_match.group(1)
                if match_val.isdigit():
                    option_index = int(match_val) - 1  # 1-indexed for user
                else:
                    # Match by address prefix
                    for i, opt in enumerate(session.continuation_options):
                        if opt["address"].lower().startswith(match_val.lower()):
                            option_index = i
                            break

            if 0 <= option_index < len(session.continuation_options):
                opt = session.continuation_options[option_index]
                logger.info(f"Continuing trace from: {opt}")

                # Build config for continuation
                config = TracerConfig(
                    description=f"Continuation trace from {opt['address']}",
                    victim_address=opt["address"],
                    blockchain_name=opt["chain"],
                    asset_symbol=opt["asset"],
                    theft_asset=opt["asset"],
                )

                session.step = "tracing"

                return StreamingResponse(
                    run_trace_streaming(config, session),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Session-Id": session_id,
                    }
                )
            else:
                return {
                    "type": "message",
                    "message": f"Invalid option. Please choose 1-{len(session.continuation_options)} or type 'done' to finish.",
                    "session_id": session_id,
                    "continuation_options": session.continuation_options
                }

        # Check if user wants to stop
        if message_lower in ["done", "stop", "finish", "no", "end"]:
            session.step = "trace_complete"
            return {
                "type": "message",
                "message": "✅ Trace complete. You can start a new trace by typing 'reset' or describing a new case.",
                "session_id": session_id
            }

        # Check if user wants to enter a custom tx hash
        tx_match = re.search(r'\b(0x[a-fA-F0-9]{64})\b', message) or re.search(r'\b([a-fA-F0-9]{64})\b', message)
        if tx_match:
            tx_hash = tx_match.group(1)
            logger.info(f"User provided custom tx hash: {tx_hash}")

            # Use the blockchain from the last trace
            blockchain = session.continuation_options[0]["chain"] if session.continuation_options else "eth"

            config = TracerConfig(
                description=f"Custom continuation from tx {tx_hash}",
                tx_hash=tx_hash,
                blockchain_name=blockchain,
            )

            session.step = "tracing"

            return StreamingResponse(
                run_trace_streaming(config, session),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Session-Id": session_id,
                }
            )

        # Show continuation options again
        return build_continuation_message(session)

    # Handle confirmation to start trace
    if is_confirmation and session.step == "confirming":
        info = session.collected_info
        logger.info(f"Starting trace with info: {info}")

        # Build config
        config = TracerConfig(
            description=info.get("description"),
            victim_address=info.get("victim_address"),
            blockchain_name=info.get("blockchain_name", "eth"),
            asset_symbol=info.get("asset_symbol"),
            approx_date=info.get("approx_date"),
            known_tx_hashes=info.get("known_tx_hashes", []),
            tx_hash=info.get("tx_hash"),
            theft_asset=info.get("theft_asset") or info.get("asset_symbol"),
        )

        session.step = "tracing"

        return StreamingResponse(
            run_trace_streaming(config, session),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-Id": session_id,
            }
        )

    # Try fast regex parsing first for simple inputs
    parsed_info = fast_parse_input(message)

    # If fast parse didn't find much, use LLM for complex descriptions
    if not parsed_info.get("victim_address") and not parsed_info.get("tx_hash") and len(message) > 50:
        logger.info("Using LLM to parse complex message...")
        try:
            llm_parsed = await parse_case_description_with_llm(message)
            # Merge LLM results with fast parse (LLM takes precedence)
            for key, value in llm_parsed.items():
                if value is not None:
                    parsed_info[key] = value
            logger.info(f"LLM parsed info: {parsed_info}")
        except Exception as e:
            logger.error(f"Failed to parse message with LLM: {e}")
    else:
        logger.info(f"Fast parsed info: {parsed_info}")

    # Merge new info with existing (new values override)
    for key, value in parsed_info.items():
        if value is not None:
            # Special handling for lists
            if key == "known_tx_hashes":
                existing = session.collected_info.get(key, [])
                if isinstance(value, list):
                    session.collected_info[key] = list(set(existing + value))
                elif value:
                    session.collected_info[key] = list(set(existing + [value]))
            else:
                session.collected_info[key] = value

    # Store description if not already stored
    if not session.collected_info.get("description") and len(message) > 20:
        session.collected_info["description"] = message

    # Check what's missing
    missing = get_missing_required_fields(session.collected_info)

    if missing:
        # Still collecting information
        session.step = "collecting"
        return {
            "type": "collecting",
            "message": build_clarification_message(session.collected_info, missing),
            "session_id": session_id,
            "collected_info": session.collected_info,
            "missing_fields": missing
        }
    else:
        # All required info collected, ask for confirmation
        session.step = "confirming"
        return {
            "type": "confirming",
            "message": build_confirmation_message(session.collected_info),
            "session_id": session_id,
            "collected_info": session.collected_info
        }


@app.post("/api/trace")
async def start_trace(request: TraceRequest):
    """
    Start a trace with explicit parameters (bypasses conversation flow).
    Returns a streaming response with progress updates.
    """
    if not request.victim_address and not request.tx_hash:
        raise HTTPException(
            status_code=400,
            detail="Either victim_address or tx_hash must be provided"
        )

    config = TracerConfig(
        description=request.description,
        victim_address=request.victim_address,
        blockchain_name=request.blockchain,
        asset_symbol=request.asset,
        approx_date=request.date,
        known_tx_hashes=request.tx_hashes or [],
        tx_hash=request.tx_hash,
        theft_asset=request.theft_asset,
    )

    return StreamingResponse(
        run_trace_streaming(config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get current session state."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "session_id": session.session_id,
        "step": session.step,
        "collected_info": session.collected_info
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}


@app.get("/api/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "active_sessions": len(sessions)
    }


def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "agent.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
