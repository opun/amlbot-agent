"""
FastAPI backend for the Crypto Tracer Agent.
Provides a conversational chat interface that collects information before tracing.
"""
import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Load .env file before anything else
from dotenv import load_dotenv

load_dotenv()

import os

from agents import gen_trace_id, set_tracing_disabled, trace

if os.getenv("AGENT_DISABLE_OPENAI_TRACING", "").lower() in ("1", "true", "yes"):
    set_tracing_disabled(True)
from agents.mcp import MCPServerStdio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.http_tracer import HTTPTracer
from agent.mcp_client import MCPClient
from agent.mcp_http_client import MCPHTTPClient
from agent.mcp_tracer import MCPTracer
from agent.models import TracerConfig, TraceResult
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
    time: str | None = None
    description: str


# Session state for multi-turn conversations
class SessionState(BaseModel):
    session_id: str
    step: str = "initial"  # initial, collecting, confirming, tracing, trace_complete, awaiting_continuation
    collected_info: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    # Trace state for continuation
    last_trace_result: dict[str, Any] | None = None
    continuation_point: dict[str, Any] | None = None  # {address, blockchain, asset, token_id}
    continuation_options: list[dict[str, Any]] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None  # Can be passed from NextAuth session


class TraceRequest(BaseModel):
    description: str | None = None
    victim_address: str | None = None
    blockchain: str = "eth"
    asset: str | None = None
    date: str | None = None
    tx_hashes: list[str] | None = None
    tx_hash: str | None = None
    theft_asset: str | None = None
    user_id: str | None = None  # Can be passed from NextAuth session
    stolen_amount: float | None = None
    cex_single_cluster_threshold: float | None = None
    traced_amount_tolerance: float | None = None


# In-memory session storage (use Redis in production)
sessions: dict[str, SessionState] = {}
SESSION_MAX_COUNT = int(os.getenv("SESSION_MAX_COUNT", "1000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour


def _cleanup_expired_sessions() -> int:
    """Remove sessions older than SESSION_TTL_SECONDS. Returns count of removed sessions."""
    now = datetime.now()
    expired = [
        sid for sid, s in sessions.items()
        if (now - s.created_at).total_seconds() > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del sessions[sid]
    return len(expired)


def get_user_id_from_request(request: Request, body_user_id: str | None = None) -> str | None:
    """Extract userId from multiple sources (priority order):
    1. Body parameter (from NextAuth session)
    2. X-User-Id header
    3. userId cookie
    """
    # Priority 1: Body parameter (passed from frontend with NextAuth session)
    if body_user_id:
        return body_user_id

    # Priority 2: X-User-Id header
    header_user_id = request.headers.get("X-User-Id")
    if header_user_id:
        return header_user_id

    # Priority 3: Cookie
    return request.cookies.get("userId")


async def _session_cleanup_loop():
    """Periodically clean up expired sessions."""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        removed = _cleanup_expired_sessions()
        if removed:
            logger.info("Cleaned up %d expired sessions, %d remaining", removed, len(sessions))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Crypto Tracer API...")
    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    yield
    cleanup_task.cancel()
    logger.info("Shutting down Crypto Tracer API...")


app = FastAPI(
    title="Crypto Tracer Agent API",
    description="API for the AMLBot Crypto Tracing Agent",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# OpenAPI spec path (served to Swagger/Redoc)
OPENAPI_PATH = Path(__file__).resolve().parents[2] / "docs" / "openapi.yaml"

# Enable CORS for frontend
_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/openapi.yaml", include_in_schema=False)
async def openapi_yaml():
    """Serve the curated OpenAPI spec for developer docs."""
    if not OPENAPI_PATH.exists():
        raise HTTPException(status_code=500, detail="OpenAPI spec not found")
    return FileResponse(OPENAPI_PATH, media_type="application/yaml", filename="openapi.yaml")


@app.get("/docs", include_in_schema=False)
async def swagger_ui():
    """Swagger UI for the curated OpenAPI spec."""
    return get_swagger_ui_html(
        openapi_url="/openapi.yaml",
        title="AMLBot Crypto Tracer API Docs",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_ui():
    """Redoc UI for the curated OpenAPI spec."""
    return get_redoc_html(
        openapi_url="/openapi.yaml",
        title="AMLBot Crypto Tracer API Docs",
    )


# Pre-compiled regex patterns for fast_parse_input
_RE_ETH_ADDRESS = re.compile(r'\b(0x[a-fA-F0-9]{40})\b')
_RE_TRON_ADDRESS = re.compile(r'\b(T[a-zA-Z0-9]{33})\b')
_RE_BTC_ADDRESS = re.compile(r'\b(bc1[a-zA-Z0-9]{39,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b')
_RE_ETH_TX = re.compile(r'\b(0x[a-fA-F0-9]{64})\b')
_RE_PLAIN_TX = re.compile(r'\b([a-fA-F0-9]{64})\b')
# Solana tx hashes are base58-encoded, typically 87-88 chars (no 0, O, I, l)
_RE_SOL_TX = re.compile(r'\b([1-9A-HJ-NP-Za-km-z]{80,90})\b')
_RE_ETH_KW = re.compile(r'\b(ethereum|eth)\b', re.IGNORECASE)
_RE_TRON_KW = re.compile(r'\b(tron|trx)\b', re.IGNORECASE)
_RE_BTC_KW = re.compile(r'\b(bitcoin|btc)\b', re.IGNORECASE)
_RE_POLY_KW = re.compile(r'\b(polygon|matic|poly)\b', re.IGNORECASE)
_RE_BSC_KW = re.compile(r'\b(bsc|binance|bnb)\b', re.IGNORECASE)
_RE_SOL_KW = re.compile(r'\b(solana|sol)\b', re.IGNORECASE)
_RE_ARB_KW = re.compile(r'\b(arbitrum|arb)\b', re.IGNORECASE)
_RE_OP_KW = re.compile(r'\b(optimism|op)\b', re.IGNORECASE)
_RE_BASE_KW = re.compile(r'\b(base)\b', re.IGNORECASE)
_RE_AVAX_KW = re.compile(r'\b(avalanche|avax)\b', re.IGNORECASE)
_RE_BCH_KW = re.compile(r'\b(bitcoin\s*cash|bch)\b', re.IGNORECASE)
_RE_LTC_KW = re.compile(r'\b(litecoin|ltc)\b', re.IGNORECASE)
_RE_ETC_KW = re.compile(r'\b(ethereum\s*classic|etc)\b', re.IGNORECASE)
_RE_ADA_KW = re.compile(r'\b(cardano|ada)\b', re.IGNORECASE)
_RE_XRP_KW = re.compile(r'\b(ripple|xrp)\b', re.IGNORECASE)
_RE_ASSET = re.compile(r'\b(USDT|USDC|ETH|BTC|TRX|BNB|MATIC|SOL|ADA|XRP|BCH|LTC|ETC)\b', re.IGNORECASE)
_RE_DATE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')


def fast_parse_input(message: str) -> dict[str, Any]:
    """
    Fast regex-based parsing for simple inputs like addresses and tx hashes.
    Avoids LLM call for simple cases.
    """
    result: dict[str, Any] = {}

    # Ethereum address pattern (0x + 40 hex chars)
    eth_address = _RE_ETH_ADDRESS.search(message)
    if eth_address:
        result["victim_address"] = eth_address.group(1)
        result["blockchain_name"] = "eth"

    # Tron address pattern (T + 33 chars)
    tron_address = _RE_TRON_ADDRESS.search(message)
    if tron_address:
        result["victim_address"] = tron_address.group(1)
        result["blockchain_name"] = "trx"

    # Bitcoin address patterns
    btc_address = _RE_BTC_ADDRESS.search(message)
    if btc_address:
        result["victim_address"] = btc_address.group(1)
        result["blockchain_name"] = "btc"

    # Ethereum tx hash (0x + 64 hex chars) - only set blockchain if explicitly ETH format
    eth_tx = _RE_ETH_TX.search(message)
    if eth_tx:
        result["tx_hash"] = eth_tx.group(1)
        # Only auto-set blockchain if message mentions ethereum/eth
        # but NOT "ethereum classic" (which is ETC)
        if _RE_ETH_KW.search(message) and not _RE_ETC_KW.search(message):
            result["blockchain_name"] = "eth"

    # Plain 64-char hex tx hash (could be Tron or other) - DON'T auto-detect blockchain
    plain_tx = _RE_PLAIN_TX.search(message)
    if plain_tx and "tx_hash" not in result:
        result["tx_hash"] = plain_tx.group(1)
        # Only set blockchain if explicitly mentioned
        if _RE_TRON_KW.search(message):
            result["blockchain_name"] = "trx"
        # Don't auto-assume blockchain for plain hashes

    # Solana base58 tx hash (87-88 chars, no 0/O/I/l)
    if "tx_hash" not in result:
        sol_tx = _RE_SOL_TX.search(message)
        if sol_tx:
            result["tx_hash"] = sol_tx.group(1)
            result["blockchain_name"] = "sol"

    # Detect blockchain from keywords
    # Note: order matters — check more specific patterns before generic ones
    # (e.g., "bitcoin cash" before "bitcoin", "ethereum classic" before "ethereum")
    if "blockchain_name" not in result:
        if _RE_BCH_KW.search(message):
            result["blockchain_name"] = "bch"
        elif _RE_ETC_KW.search(message):
            result["blockchain_name"] = "etc"
        elif _RE_ETH_KW.search(message):
            result["blockchain_name"] = "eth"
        elif _RE_TRON_KW.search(message):
            result["blockchain_name"] = "trx"
        elif _RE_BTC_KW.search(message):
            result["blockchain_name"] = "btc"
        elif _RE_POLY_KW.search(message):
            result["blockchain_name"] = "poly"
        elif _RE_BSC_KW.search(message):
            result["blockchain_name"] = "bsc"
        elif _RE_SOL_KW.search(message):
            result["blockchain_name"] = "sol"
        elif _RE_ARB_KW.search(message):
            result["blockchain_name"] = "arb"
        elif _RE_OP_KW.search(message):
            result["blockchain_name"] = "op"
        elif _RE_BASE_KW.search(message):
            result["blockchain_name"] = "base"
        elif _RE_AVAX_KW.search(message):
            result["blockchain_name"] = "avax"
        elif _RE_LTC_KW.search(message):
            result["blockchain_name"] = "ltc"
        elif _RE_ADA_KW.search(message):
            result["blockchain_name"] = "ada"
        elif _RE_XRP_KW.search(message):
            result["blockchain_name"] = "xrp"

    # Detect asset from keywords
    # When the message is a short direct answer (≤10 chars), the user is likely answering
    # "what asset was stolen?" — accept the asset even if it matches the blockchain name.
    # For longer messages, suppress ambiguous matches (e.g., "trx" could mean blockchain).
    # Prefer asset mentioned near "stolen"/"asset" context over first occurrence.
    _NATIVE_TOKEN_TO_CHAIN = {"ETH": "ETH", "BNB": "BSC", "TRX": "TRX", "MATIC": "POLY", "SOL": "SOL", "ADA": "ADA", "XRP": "XRP", "BTC": "BTC", "BCH": "BCH", "LTC": "LTC", "ETC": "ETC"}
    _asset_context_re = re.compile(r'(?:stolen|asset|token)[:\s]+(' + '|'.join(_RE_ASSET.pattern.split('(')[1].split(')')[0].split('|')) + r')', re.IGNORECASE)
    asset_context_match = _asset_context_re.search(message)
    asset_match = asset_context_match or _RE_ASSET.search(message)
    if asset_match:
        detected_asset = asset_match.group(1).upper()
        blockchain = result.get("blockchain_name", "").upper()
        is_short_response = len(message.strip()) <= 10
        has_explicit_context = asset_context_match is not None
        # Check ambiguity: BNB is the native token for BSC, ETH for ETH, etc.
        # If the asset was explicitly mentioned near "stolen"/"asset", always accept it.
        is_native_token = _NATIVE_TOKEN_TO_CHAIN.get(detected_asset, "").upper() == blockchain
        if has_explicit_context or is_short_response or (detected_asset != blockchain and not is_native_token):
            result["asset_symbol"] = detected_asset
            result["theft_asset"] = detected_asset

    # Detect date patterns
    date_match = _RE_DATE.search(message)
    if date_match:
        result["approx_date"] = date_match.group(1)

    return result


def get_or_create_session(session_id: str | None) -> SessionState:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return sessions[session_id]

    # Enforce session limits
    if len(sessions) >= SESSION_MAX_COUNT:
        removed = _cleanup_expired_sessions()
        if removed == 0 and len(sessions) >= SESSION_MAX_COUNT:
            # Evict oldest session
            oldest_id = min(sessions, key=lambda sid: sessions[sid].created_at)
            del sessions[oldest_id]
            logger.warning("Evicted oldest session %s due to max session limit", oldest_id)

    new_id = session_id or str(uuid.uuid4())
    session = SessionState(session_id=new_id)
    sessions[new_id] = session
    return session


def format_collected_info(info: dict[str, Any]) -> str:
    """Format collected information for display."""
    lines = []

    field_labels = {
        "tx_hash": "🔗 Transaction Hash",
        "victim_address": "📍 Victim Address",
        "blockchain_name": "⛓️ Blockchain",
        "theft_asset": "💰 Stolen Asset",
        "asset_symbol": "💰 Asset",
        "approx_date": "📅 Approximate Date",
    }

    for key, label in field_labels.items():
        value = info.get(key)
        if value:
            # Skip asset_symbol if theft_asset is the same
            if key == "asset_symbol" and info.get("theft_asset") == value:
                continue
            lines.append(f"- {label}: `{value}`")

    return "\n".join(lines) if lines else "No information collected yet."


def get_missing_required_fields(info: dict[str, Any]) -> list[str]:
    """Check what required fields are still missing."""
    missing = []

    # Must have either victim_address or tx_hash
    has_address = bool(info.get("victim_address"))
    has_tx = bool(info.get("tx_hash"))

    if not has_address and not has_tx:
        missing.append("victim_address or tx_hash")

    # Blockchain is ALWAYS required
    if not info.get("blockchain_name"):
        missing.append("blockchain")

    # Stolen asset is REQUIRED when we have either tx_hash or victim_address
    if (has_tx or has_address) and not info.get("theft_asset") and not info.get("asset_symbol"):
        missing.append("theft_asset")

    return missing


def build_clarification_message(info: dict[str, Any], missing: list[str]) -> str:
    """Build a message asking for missing information."""
    current = format_collected_info(info)

    msg = "## 📋 Information Collected\n\n"
    msg += current + "\n\n"

    # Only ask for one thing at a time
    if missing:
        msg += "## ❓ Required Information\n\n"

        # Prioritize missing fields
        field_to_ask = missing[0]

        if "victim_address or tx_hash" in field_to_ask:
            msg += "**Please provide the Transaction Hash or Victim Wallet Address.**\n"
            msg += "\n💡 *Example: `0x1234...abcd`*"

        elif "blockchain" in field_to_ask:
            msg += "**Which blockchain network is this on?**\n"
            msg += "- Examples: `eth`, `trx`, `btc`, `bsc`, `polygon`\n"

        elif "theft_asset" in field_to_ask:
            msg += "**What asset was stolen?**\n"
            msg += "- Examples: `USDT`, `ETH`, `TRX`, `USDC`\n"

    return msg


def build_continuation_message(session: SessionState) -> dict[str, Any]:
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

        chain_info = f" on {opt['chain'].upper()}" if opt.get('chain') else ""

        msg += f"**{i}.** `{addr_short}`{chain_info}{risk_info}\n"
        msg += f"   - Description: {opt.get('description', '')}\n"
        msg += f"   - Last amount: {opt['last_amount']:,.2f} {opt['asset']}\n"

        if opt.get("bridge_error"):
             msg += "   - ⚠️ Could not auto-detect bridge destination. Please provide it manually.\n"

        msg += "\n"

    msg += "---\n\n"
    msg += "**What would you like to do?**\n"
    msg += "- Type **'continue 1'** (or 2...) to continue tracing\n"
    msg += "- Paste a **destination wallet address** if you know it (e.g. for bridge)\n"
    msg += "- Paste a **transaction hash** to trace from a specific tx\n"
    msg += "- Type **'done'** to finish\n"

    return {
        "type": "continuation",
        "message": msg,
        "session_id": session.session_id,
        "continuation_options": options
    }


def build_confirmation_message(info: dict[str, Any]) -> str:
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
    session: SessionState | None = None,
    user_id: str | None = None
) -> AsyncGenerator[str, None]:
    """Run a trace and yield streaming updates."""

    if not user_id:
        yield json.dumps({"type": "error", "message": "Authentication required"}) + "\n"
        return

    # Check if we should use HTTP mode
    use_http = os.getenv("MCP_USE_HTTP", "false").lower() == "true"
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8001")

    if use_http:
        # HTTP mode - uses HTTP client and OpenAI function calling
        yield json.dumps({"type": "status", "message": "Connecting to MCP HTTP server..."}) + "\n"
        async for chunk in _run_trace_http(config, session, user_id, mcp_server_url):
            yield chunk
    else:
        # Stdio mode - uses Docker container with MCP stdio protocol
        yield json.dumps({"type": "status", "message": "Starting MCP server..."}) + "\n"
        async for chunk in _run_trace_stdio(config, session, user_id):
            yield chunk


async def _run_trace_http(
    config: TracerConfig,
    session: SessionState | None,
    user_id: str,
    mcp_server_url: str
) -> AsyncGenerator[str, None]:
    """Run trace using HTTP MCP server."""
    trace_id = gen_trace_id()
    trace_url = f"https://platform.openai.com/traces/trace?trace_id={trace_id}"
    logger.info(f"Trace URL: {trace_url}")
    logger.info("Streaming event: trace_started")
    yield json.dumps({
        "type": "trace_started",
        "trace_id": trace_id,
        "trace_url": trace_url
    }) + "\n"

    http_client = MCPHTTPClient(mcp_server_url, user_id)
    viz_client = None

    try:
        with trace(workflow_name="Crypto Tracer Agent " + str(time.time()), trace_id=trace_id):
            tracer = HTTPTracer(http_client)
            progress_queue = asyncio.Queue()

            async def on_progress(message: str):
                await progress_queue.put(message)

            # Start trace in background
            trace_task = asyncio.create_task(tracer.trace(config, on_progress=on_progress))
            last_status_ts = time.time()

            # Stream progress until trace is done
            while not trace_task.done():
                try:
                    # Check for progress updates with a short timeout
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    logger.info("Streaming event: status")
                    yield json.dumps({"type": "status", "message": msg}) + "\n"
                    last_status_ts = time.time()
                except TimeoutError:
                    if time.time() - last_status_ts > 15:
                        logger.info("Streaming event: status (keepalive)")
                        yield json.dumps({"type": "status", "message": "Still analyzing..."} ) + "\n"
                        last_status_ts = time.time()
                    continue

            logger.info("Trace task done=%s cancelled=%s", trace_task.done(), trace_task.cancelled())
            # Get final result
            result = await trace_task
            logger.info("Trace task completed, building report next")

            # Align trace_id with OpenAI dashboard link
            if result.case_meta:
                result.case_meta.trace_id = trace_id

            logger.info("Streaming event: status (building_report)")
            yield json.dumps({"type": "status", "message": "Building report..."}) + "\n"
            logger.info("Building report start")
            report = build_report(result)
            logger.info("Building report done")

            # Extract continuation options using HTTP client
            logger.info("Extracting continuation options start")
            continuation_options = await _extract_continuation_options_http(result, http_client, config)
            logger.info("Extracting continuation options done: %d", len(continuation_options))

            # Store trace result in session for continuation
            if session:
                session.last_trace_result = report
                session.continuation_options = continuation_options
                if continuation_options:
                    session.step = "awaiting_continuation"
                else:
                    session.step = "trace_complete"

            response_data = {
                "type": "result",
                "report": report,
                "trace_id": trace_id,
                "trace_url": trace_url,
            }
            response_data["visualization_url"] = result.visualization_url
            # Attach raw arrays for visualization pipeline (txs + txList)
            if hasattr(tracer, "last_txs"):
                response_data["txs_array"] = tracer.last_txs
            if hasattr(tracer, "last_tx_list"):
                response_data["txlist_array"] = tracer.last_tx_list
            logger.info("Trace result ready: report_keys=%s", list(report.keys()) if isinstance(report, dict) else "non-dict")

            if continuation_options:
                response_data["continuation_options"] = continuation_options
                response_data["can_continue"] = True
            else:
                response_data["can_continue"] = False

            logger.info("Streaming event: result")
            yield json.dumps(response_data) + "\n"

    except asyncio.CancelledError:
        logger.warning("Trace stream cancelled by client")
        raise
    except Exception as e:
        logger.error(f"Trace error: {e}")
        logger.info("Streaming event: error")
        yield json.dumps({
            "type": "error",
            "message": str(e)
        }) + "\n"

    finally:
        close_tasks = [http_client.aclose()]
        if viz_client:
            close_tasks.append(viz_client.aclose())
        await asyncio.gather(*close_tasks, return_exceptions=True)


async def _run_trace_stdio(
    config: TracerConfig,
    session: SessionState | None,
    user_id: str
) -> AsyncGenerator[str, None]:
    """Run trace using stdio MCP server (Docker)."""
    trace_id = gen_trace_id()
    trace_url = f"https://platform.openai.com/traces/trace?trace_id={trace_id}"
    logger.info(f"Trace URL: {trace_url}")
    yield json.dumps({
        "type": "trace_started",
        "trace_id": trace_id,
        "trace_url": trace_url
    }) + "\n"

    viz_client = None

    with trace(workflow_name="Crypto Tracer Agent " + str(time.time()), trace_id=trace_id):
        async with MCPServerStdio(
            name="AMLBot MCP Server",
            params={
                "command": "docker",
                "args": ["run", "-i", "--rm", "-e", f"USER_ID={user_id}", "mcp-server-amlbot:local"]
            },
            client_session_timeout_seconds=300.0,
        ) as server:
            client = MCPClient(server)
            tracer = MCPTracer(client)
            progress_queue = asyncio.Queue()

            async def on_progress(message: str):
                await progress_queue.put(message)

            logger.info("Streaming event: status (running_trace)")
            yield json.dumps({"type": "status", "message": "Running trace analysis..."}) + "\n"
            last_status_ts = time.time()

            try:
                # Start trace in background
                trace_task = asyncio.create_task(tracer.trace(config, on_progress=on_progress))

                # Stream progress until trace is done
                while not trace_task.done():
                    try:
                        msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                        logger.info("Streaming event: status")
                        yield json.dumps({"type": "status", "message": msg}) + "\n"
                        last_status_ts = time.time()
                    except TimeoutError:
                        if time.time() - last_status_ts > 15:
                            logger.info("Streaming event: status (keepalive)")
                            yield json.dumps({"type": "status", "message": "Still analyzing..."} ) + "\n"
                            last_status_ts = time.time()
                        continue

                # Get final result
                result = await trace_task

                # Align trace_id with OpenAI dashboard link
                if result.case_meta:
                    result.case_meta.trace_id = trace_id

                logger.info("Streaming event: status (building_report)")
                yield json.dumps({"type": "status", "message": "Building report..."}) + "\n"
                report = build_report(result)

                # Extract continuation options - only when user decision is needed
                continuation_options = await extract_continuation_options(result, client, config)

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
                    "trace_url": trace_url,
                }
                response_data["visualization_url"] = result.visualization_url
                # Attach raw arrays for visualization pipeline (txs + txList)
                if hasattr(tracer, "last_txs"):
                    response_data["txs_array"] = tracer.last_txs
                if hasattr(tracer, "last_tx_list"):
                    response_data["txlist_array"] = tracer.last_tx_list
                logger.info("Trace result ready: report_keys=%s", list(report.keys()) if isinstance(report, dict) else "non-dict")

                if continuation_options:
                    response_data["continuation_options"] = continuation_options
                    response_data["can_continue"] = True
                else:
                    response_data["can_continue"] = False

                logger.info("Streaming event: result")
                yield json.dumps(response_data) + "\n"
            except asyncio.CancelledError:
                logger.warning("Trace stream cancelled by client")
                raise
            except Exception as e:
                logger.error(f"Trace error: {e}")
                yield json.dumps({
                    "type": "error",
                    "message": str(e)
                }) + "\n"
            finally:
                if viz_client:
                    await viz_client.aclose()


async def _extract_continuation_options_http(
    result: TraceResult,
    client: MCPHTTPClient,
    config: TracerConfig
) -> list[dict[str, Any]]:
    """Extract continuation options using HTTP client."""
    options = []

    for path in result.paths:
        if not path.steps:
            continue

        last_step = path.steps[-1]
        last_address = last_step.to_address

        entity = None
        for e in result.entities:
            if e.address == last_address:
                entity = e
                break

        stop_reason = (path.stop_reason or "").lower()

        needs_user_decision = (
            "user" in stop_reason or
            "decision" in stop_reason or
            "ambiguous" in stop_reason or
            "multiple" in stop_reason or
            "choose" in stop_reason
        )

        is_bridge = entity and entity.role == "bridge_service"
        is_cex = entity and entity.role == "cex_deposit"

        if is_bridge or is_cex or needs_user_decision:
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

            if is_bridge and last_step.tx_hash:
                try:
                    logger.info(f"Analyzing bridge tx: {last_step.tx_hash} on {last_step.chain}")
                    bridge_info = await asyncio.wait_for(
                        client.bridge_analyze(last_step.chain, last_step.tx_hash),
                        timeout=10.0
                    )

                    if bridge_info and bridge_info.get("is_bridge"):
                        dst_chain = bridge_info.get("dst_chain")
                        dst_addr = bridge_info.get("destination_address")

                        if dst_chain:
                            option["bridge_info"] = bridge_info
                            option["description"] = f"Continue on {dst_chain.upper()}"
                            if dst_addr:
                                option["address"] = dst_addr
                                option["chain"] = dst_chain
                                option["description"] += f" (Dest: {dst_addr[:8]}...)"
                except Exception as e:
                    logger.warning(f"Bridge analysis failed: {e}")
                    option["bridge_error"] = True

            options.append(option)

    return options


async def extract_continuation_options(
    result: TraceResult,
    client: MCPClient,
    config: TracerConfig
) -> list[dict[str, Any]]:
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
        if is_bridge or is_cex or needs_user_decision:
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

            # For bridges, try to pre-analyze the destination
            if is_bridge and last_step.tx_hash:
                try:
                    logger.info(f"Analyzing bridge tx: {last_step.tx_hash} on {last_step.chain}")
                    bridge_info = await client.bridge_analyze(last_step.chain, last_step.tx_hash)

                    if bridge_info and bridge_info.get("is_bridge"):
                        dst_chain = bridge_info.get("dst_chain")
                        dst_addr = bridge_info.get("destination_address")

                        if dst_chain:
                             option["bridge_info"] = bridge_info
                             option["description"] = f"Continue on {dst_chain.upper()}"
                             if dst_addr:
                                 option["address"] = dst_addr # Update target to destination address
                                 option["chain"] = dst_chain
                                 option["description"] += f" (Dest: {dst_addr[:8]}...)"
                except Exception as e:
                    logger.warning(f"Bridge analysis failed: {e}")
                    option["bridge_error"] = True

            options.append(option)

    return options


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "crypto-tracer-api"}


@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request):
    """
    Handle chat messages with multi-turn conversation flow.
    Collects information before starting a trace.
    """
    # Get userId from body, header, or cookies
    user_id = get_user_id_from_request(http_request, request.user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required. Please login.")

    message = request.message.strip()
    logger.info(f"Chat request received: {message[:100]}...")

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Get or create session
    session = get_or_create_session(request.session_id)
    session_id = session.session_id
    logger.info(f"Session: {session_id}, step: {session.step}, user_id: {user_id[:8]}...")

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

    # Auto-reset session when user provides new case data after a completed trace.
    # This lets users start a new trace without typing "reset" or reloading.
    if session.step in ("trace_complete", "awaiting_continuation") and not is_confirmation:
        # Check if the message looks like a new case (has an address, tx hash, or long description)
        _has_new_data = (
            _RE_ETH_TX.search(message)
            or _RE_ETH_ADDRESS.search(message)
            or _RE_TRON_ADDRESS.search(message)
            or _RE_BTC_ADDRESS.search(message)
            or _RE_PLAIN_TX.search(message)
            or _RE_SOL_TX.search(message)
            or len(message.strip()) > 30
        )
        if _has_new_data:
            logger.info(f"New case data detected after {session.step} — auto-resetting session")
            session.collected_info = {}
            session.step = "initial"
            session.last_trace_result = None
            session.continuation_options = []

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
                    run_trace_streaming(config, session, user_id=user_id),
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
                run_trace_streaming(config, session, user_id=user_id),
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
            run_trace_streaming(config, session, user_id=user_id),
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

    # Context-aware merge: use what's currently missing to resolve ambiguity.
    # Short responses like "eth" or "trx" match both blockchain and asset — only set
    # the field the user is most likely answering (the one that's currently missing).
    pre_missing = get_missing_required_fields(session.collected_info)
    chain_is_missing = any("blockchain" in m for m in pre_missing)
    any("theft_asset" in m for m in pre_missing)

    parsed_chain = parsed_info.get("blockchain_name", "")
    parsed_asset = (parsed_info.get("theft_asset") or "")
    is_ambiguous_pair = (
        len(message.strip()) <= 10
        and parsed_chain
        and parsed_asset
        and parsed_chain.upper() == parsed_asset.upper()
    )

    for key, value in parsed_info.items():
        if value is not None:
            # Special handling for lists
            if key == "known_tx_hashes":
                existing = session.collected_info.get(key, [])
                if isinstance(value, list):
                    session.collected_info[key] = list(set(existing + value))
                elif value:
                    session.collected_info[key] = list(set(existing + [value]))
            elif is_ambiguous_pair and key in ("theft_asset", "asset_symbol") and chain_is_missing:
                # User typed "trx" — blockchain is missing, so they're answering that.
                # Don't also auto-set asset; ask explicitly.
                continue
            elif is_ambiguous_pair and key == "blockchain_name" and not chain_is_missing:
                # User typed "ETH" — blockchain already set, they're answering asset question.
                # Don't overwrite the existing blockchain.
                continue
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
async def start_trace(request: TraceRequest, http_request: Request):
    """
    Start a trace with explicit parameters (bypasses conversation flow).
    Returns a streaming response with progress updates.
    """
    # Get userId from body, header, or cookies
    user_id = get_user_id_from_request(http_request, request.user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required. Please login.")

    if not request.victim_address and not request.tx_hash:
        raise HTTPException(
            status_code=400,
            detail="Either victim_address or tx_hash must be provided"
        )

    config_kwargs: dict[str, Any] = dict(
        description=request.description,
        victim_address=request.victim_address,
        blockchain_name=request.blockchain,
        asset_symbol=request.asset,
        approx_date=request.date,
        known_tx_hashes=request.tx_hashes or [],
        tx_hash=request.tx_hash,
        theft_asset=request.theft_asset,
    )
    if request.stolen_amount is not None:
        config_kwargs["stolen_amount"] = request.stolen_amount
    if request.cex_single_cluster_threshold is not None:
        config_kwargs["cex_single_cluster_threshold"] = request.cex_single_cluster_threshold
    if request.traced_amount_tolerance is not None:
        config_kwargs["traced_amount_tolerance"] = request.traced_amount_tolerance
    config = TracerConfig(**config_kwargs)

    return StreamingResponse(
        run_trace_streaming(config, user_id=user_id),
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

    reload = os.getenv("AGENT_RELOAD", "true").lower() in ("1", "true", "yes")
    uvicorn.run(
        "agent.api:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
    )


if __name__ == "__main__":
    main()
