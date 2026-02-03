"""
HTTP-based Crypto Tracer that uses HTTP client instead of MCP stdio.
Uses OpenAI function calling instead of MCP servers.
"""
import json
import uuid
import logging
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
from httpx import Timeout, Limits

# Timeout settings for OpenAI API calls (in seconds)
OPENAI_TIMEOUT = 45  # 45 seconds per API call
OPENAI_CONNECT_TIMEOUT = 10  # 10 seconds to establish connection

from agent.models import (
    TracerConfig,
    TraceResult,
    CaseMeta,
    TraceStats
)
from agent.mcp_http_client import MCPHTTPClient
from agent.theft_detection import (
    infer_asset_symbol,
    infer_approx_date_from_description,
    extract_victim_from_tx_hash
)
from agent.visualization import generate_visualization_payload

# Setup logger
logger = logging.getLogger("http_tracer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[HTTP_TRACE] %(message)s'))
    logger.addHandler(handler)


# Define tools for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "expert_search",
            "description": "Search for an address OR transaction hash in the explorer. Can accept both wallet addresses (42 chars ETH) and transaction hashes (66 chars ETH).",
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {"type": "string", "description": "The address or transaction hash to search for. Addresses: 42 chars (ETH), 34 chars (TRON). Tx hashes: 66 chars (ETH), 64 chars (TRON)."},
                    "filter": {"type": "string", "description": "Filter type: 'explorer' or 'entity'", "default": "explorer"}
                },
                "required": ["hash"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_address",
            "description": "Get information about a blockchain ADDRESS (wallet). NOT for transaction hashes. For ETH/EVM: address is 42 chars (0x + 40 hex). For TRON: 34 chars starting with T. Transaction hashes are 66 chars - use get_transaction for those.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blockchain_name": {"type": "string", "description": "Blockchain network (eth, trx, btc, etc.)"},
                    "address": {"type": "string", "description": "The wallet ADDRESS (NOT tx hash). ETH: 0x + 40 hex chars (42 total). TRON: T + 33 chars (34 total). Do NOT pass transaction hashes here."}
                },
                "required": ["blockchain_name", "address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "token_stats",
            "description": "Get token statistics for a wallet ADDRESS. NOT for transaction hashes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blockchain_name": {"type": "string", "description": "Blockchain network"},
                    "address": {"type": "string", "description": "The wallet ADDRESS (42 chars for ETH, 34 chars for TRON). NOT a tx hash."}
                },
                "required": ["blockchain_name", "address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "all_txs",
            "description": "Get all transactions for a wallet ADDRESS. The 'address' parameter must be a wallet address, NOT a transaction hash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Wallet ADDRESS to get transactions for. ETH: 42 chars (0x + 40 hex). TRON: 34 chars. NOT a tx hash (which is 66 chars)."},
                    "blockchain_name": {"type": "string", "description": "Blockchain network"},
                    "filter": {"type": "object", "description": "Filter criteria"},
                    "limit": {"type": "integer", "description": "Max transactions to return", "default": 20},
                    "offset": {"type": "integer", "description": "Offset for pagination", "default": 0},
                    "direction": {"type": "string", "description": "Sort direction: 'asc' or 'desc'", "default": "asc"},
                    "order": {"type": "string", "description": "Order by field", "default": "time"},
                    "transaction_type": {"type": "string", "description": "Type: 'all', 'withdrawal', 'deposit'", "default": "all"}
                },
                "required": ["address", "blockchain_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction",
            "description": "Get detailed transaction information. Use this when you have a TRANSACTION HASH (66 chars for ETH).",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Related wallet ADDRESS (42 chars ETH, 34 chars TRON) - NOT a tx hash"},
                    "tx_hash": {"type": "string", "description": "Transaction HASH (66 chars for ETH: 0x + 64 hex)"},
                    "blockchain_name": {"type": "string", "description": "Blockchain network"},
                    "token_id": {"type": "integer", "description": "Token ID (0 for native)", "default": 0},
                    "path": {"type": "string", "description": "Internal path", "default": "0"}
                },
                "required": ["address", "tx_hash", "blockchain_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_extra_address_info",
            "description": "Get extra info for a wallet ADDRESS including tags and risk score. NOT for tx hashes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Wallet ADDRESS (42 chars ETH, 34 chars TRON). NOT a transaction hash."},
                    "asset": {"type": "string", "description": "Asset symbol (e.g., 'ETH', 'USDT')"}
                },
                "required": ["address", "asset"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bridge_analyze",
            "description": "Analyze a transaction to detect if it's a cross-chain bridge operation. Requires a TRANSACTION HASH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "description": "Source chain (eth, trx, bsc, etc.)"},
                    "tx_hash": {"type": "string", "description": "Transaction HASH (66 chars for ETH: 0x + 64 hex). NOT an address."}
                },
                "required": ["chain", "tx_hash"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "token_transfers",
            "description": "Get token transfers for a transaction. Requires a TRANSACTION HASH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_hash": {"type": "string", "description": "Transaction HASH (66 chars for ETH: 0x + 64 hex). NOT an address."},
                    "blockchain_name": {"type": "string", "description": "Blockchain network"}
                },
                "required": ["tx_hash", "blockchain_name"]
            }
        }
    }
]


class HTTPCryptoTracer:
    """Crypto tracer that uses HTTP client and OpenAI function calling."""

    def __init__(self, client: MCPHTTPClient):
        self.client = client
        # Configure OpenAI client with proper httpx timeout
        # Use short timeouts and disable keepalive to prevent hanging
        import httpx
        http_client = httpx.AsyncClient(
            timeout=Timeout(OPENAI_TIMEOUT, connect=OPENAI_CONNECT_TIMEOUT),
            limits=Limits(max_keepalive_connections=0, max_connections=10),  # No keepalive
        )
        self.openai_client = AsyncOpenAI(
            http_client=http_client,
            max_retries=1  # Only retry once on transient errors (faster failure)
        )
        self.prompt_path = Path(__file__).parent / "prompts" / "trace_orchestrator.md"

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    def _strip_code_fences(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _format_address(self, addr: str) -> str:
        """Format address for display: show first 8 and last 6 chars."""
        if len(addr) > 16:
            return f"{addr[:8]}...{addr[-6:]}"
        return addr

    def _format_hash(self, tx_hash: str) -> str:
        """Format tx hash for display: show first 10 and last 6 chars."""
        if len(tx_hash) > 18:
            return f"{tx_hash[:10]}...{tx_hash[-6:]}"
        return tx_hash

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool via HTTP client."""
        tool_start = time.time()

        # Log human-readable description of what we're doing
        if tool_name == "expert_search":
            logger.info(f"🔎 Searching explorer for: {self._format_hash(arguments['hash'])}")
            result = await self.client.expert_search(
                arguments["hash"],
                arguments.get("filter", "explorer")
            )
        elif tool_name == "get_address":
            logger.info(f"🔍 Checking address info: {self._format_address(arguments['address'])}")
            result = await self.client.get_address(
                arguments["blockchain_name"],
                arguments["address"]
            )
        elif tool_name == "token_stats":
            logger.info(f"📊 Getting token stats for: {self._format_address(arguments['address'])}")
            result = await self.client.token_stats(
                arguments["blockchain_name"],
                arguments["address"]
            )
        elif tool_name == "all_txs":
            tx_type = arguments.get("transaction_type", "all")
            logger.info(f"📋 Getting {tx_type} transactions from: {self._format_address(arguments['address'])}")
            # Handle filter parameter - LLM sometimes passes it as a JSON string
            filter_param = arguments.get("filter")
            if isinstance(filter_param, str):
                try:
                    filter_param = json.loads(filter_param)
                    logger.debug(f"Parsed filter from string: {filter_param}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse filter string: {filter_param}, using None")
                    filter_param = None
            result = await self.client.all_txs(
                arguments["address"],
                arguments["blockchain_name"],
                filter_param,
                arguments.get("limit", 20),
                arguments.get("offset", 0),
                arguments.get("direction", "asc"),
                arguments.get("order", "time"),
                arguments.get("transaction_type", "all")
            )
        elif tool_name == "get_transaction":
            logger.info(f"📄 Getting transaction details: {self._format_hash(arguments['tx_hash'])}")
            result = await self.client.get_transaction(
                arguments["address"],
                arguments["tx_hash"],
                arguments["blockchain_name"],
                arguments.get("token_id", 0),
                arguments.get("path", "0")
            )
        elif tool_name == "get_extra_address_info":
            logger.info(f"🏷️  Checking entity/service info: {self._format_address(arguments['address'])}")
            result = await self.client.get_extra_address_info(
                arguments["address"],
                arguments["asset"]
            )
        elif tool_name == "bridge_analyze":
            logger.info(f"🌉 Analyzing bridge activity: {self._format_hash(arguments['tx_hash'])}")
            result = await self.client.bridge_analyze(
                arguments["chain"],
                arguments["tx_hash"]
            )
        elif tool_name == "token_transfers":
            logger.info(f"📤 Getting token transfers for tx: {self._format_hash(arguments['tx_hash'])}")
            result = await self.client.token_transfers(
                arguments["tx_hash"],
                arguments["blockchain_name"]
            )
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        elapsed = time.time() - tool_start
        logger.debug(f"Tool {tool_name} completed in {elapsed:.2f}s")
        return result

    async def _run_orchestrator(self, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the LLM orchestrator with function calling and return parsed JSON."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ]

        logger.info("🚀 Starting trace orchestrator...")
        trace_start = time.time()

        max_turns = 100  # Allow deep traces with many hops and cross-chain bridges
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        addresses_traced: List[str] = []  # Track addresses we've analyzed

        for turn in range(max_turns):
            turn_start = time.time()

            # Log what we're waiting for
            logger.info(f"⏳ Turn {turn + 1}: Waiting for OpenAI to decide next action...")

            try:
                # Log what we're sending to OpenAI
                msg_count = len(messages)
                total_chars = sum(len(str(m.get('content', ''))) for m in messages if isinstance(m, dict))
                tool_msg_count = sum(1 for m in messages if isinstance(m, dict) and m.get('role') == 'tool')
                logger.debug(f"📨 OpenAI request: {msg_count} messages, {tool_msg_count} tool results, ~{total_chars} chars")

                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )
                consecutive_timeouts = 0  # Reset on success

                # Log response stats
                usage = response.usage
                if usage:
                    logger.debug(f"📩 OpenAI response: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} tokens")

            except (asyncio.TimeoutError, APITimeoutError, APIConnectionError) as e:
                consecutive_timeouts += 1
                logger.warning(f"⚠️  Turn {turn + 1}: OpenAI API error: {type(e).__name__} - {str(e)[:100]}")
                logger.warning(f"   ({consecutive_timeouts}/{max_consecutive_timeouts} consecutive errors)")

                if consecutive_timeouts >= max_consecutive_timeouts:
                    logger.error(f"❌ OpenAI API timed out {max_consecutive_timeouts} times consecutively. Aborting trace.")
                    raise RuntimeError(
                        f"OpenAI API timed out {max_consecutive_timeouts} times consecutively after {turn + 1} turns. "
                        f"The trace may be too complex or the API is experiencing issues."
                    ) from e

                # Wait a bit before retrying
                await asyncio.sleep(2)
                logger.info(f"🔄 Retrying turn {turn + 1}...")
                continue

            turn_elapsed = time.time() - turn_start
            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            logger.info(f"✅ Turn {turn + 1}: OpenAI responded in {turn_elapsed:.1f}s (finish_reason={finish_reason})")

            # Check if we have tool calls
            if message.tool_calls:
                # Log what tools OpenAI wants to call
                tool_names = [tc.function.name for tc in message.tool_calls]
                logger.info(f"🔧 OpenAI requesting {len(tool_names)} tool(s): {', '.join(tool_names)}")

                messages.append(message)

                # Track addresses being analyzed for progress reporting
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    # Track unique addresses we've analyzed
                    if tool_name == "get_address" and "address" in arguments:
                        addr = arguments["address"]
                        if addr not in addresses_traced:
                            addresses_traced.append(addr)
                            hop_num = len(addresses_traced)
                            logger.info(f"📍 Hop {hop_num}: Analyzing new address {self._format_address(addr)}")

                    try:
                        result = await self._execute_tool(tool_name, arguments)
                        tool_result = json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"❌ Tool execution error: {e}")
                        tool_result = json.dumps({"error": str(e)})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                logger.debug(f"Turn {turn + 1}: Executed {len(message.tool_calls)} tool calls")

                # Periodic progress summary
                if turn > 0 and turn % 5 == 0:
                    total_elapsed = time.time() - trace_start
                    logger.info(f"📊 Progress: {turn + 1} turns, {len(addresses_traced)} addresses traced, {total_elapsed:.0f}s elapsed")
            else:
                # No tool calls, we have the final response
                total_elapsed = time.time() - trace_start
                raw_output = message.content or ""

                logger.info(f"🎉 Trace completed!")
                logger.info(f"   └─ Turns: {turn + 1}")
                logger.info(f"   └─ Addresses traced: {len(addresses_traced)}")
                logger.info(f"   └─ Total time: {total_elapsed:.1f}s")
                logger.info(f"   └─ Final response: {len(raw_output)} chars")
                if addresses_traced:
                    logger.info(f"   └─ Trace path: {' → '.join(self._format_address(a) for a in addresses_traced)}")
                logger.debug(f"Raw LLM output ({len(raw_output)} chars): {raw_output[:500]}{'...' if len(raw_output) > 500 else ''}")

                if not raw_output.strip():
                    logger.error("LLM returned empty response!")
                    raise ValueError("LLM returned empty response - no JSON to parse")

                cleaned = self._strip_code_fences(raw_output)
                logger.info(f"Cleaned output length: {len(cleaned)} chars")

                # Check if response looks like it might be JSON
                if not cleaned.strip().startswith('{'):
                    logger.error(f"Response does not start with '{{'. First 200 chars: {cleaned[:200]}")
                    raise ValueError(f"LLM response is not JSON. Response starts with: {cleaned[:100]}...")

                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error on first attempt: {e}")
                    logger.warning(f"Raw output that failed to parse: {cleaned[:500]}...")

                    # Retry: Ask the LLM to convert its response to JSON format
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. Please provide the complete trace result "
                            "as a valid JSON object following the TraceResult schema. "
                            "Output ONLY raw JSON with no markdown formatting, no explanations, no code blocks. "
                            "The response must be parseable by json.loads()."
                        )
                    })

                    logger.debug("Retrying with explicit JSON instruction...")

                    try:
                        retry_response = await self.openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            tools=TOOLS,
                            tool_choice="none"  # Force text response, no more tool calls
                        )
                    except (asyncio.TimeoutError, APITimeoutError, APIConnectionError) as timeout_e:
                        logger.error(f"OpenAI API error during JSON retry: {timeout_e}")
                        raise RuntimeError(
                            f"OpenAI API error while retrying JSON conversion. "
                            f"Original JSON error: {e}"
                        ) from timeout_e

                    retry_output = retry_response.choices[0].message.content or ""
                    retry_cleaned = self._strip_code_fences(retry_output)
                    logger.debug(f"Retry output ({len(retry_cleaned)} chars): {retry_cleaned[:500]}...")

                    try:
                        return json.loads(retry_cleaned)
                    except json.JSONDecodeError as retry_e:
                        logger.error(f"JSON parse error on retry: {retry_e}")
                        logger.error(f"Failed to parse after retry: {retry_cleaned[:1000]}")
                        raise ValueError(
                            f"LLM failed to produce valid JSON. First attempt error: {e}. "
                            f"Retry error: {retry_e}. Check logs for raw output."
                        ) from retry_e

        total_elapsed = time.time() - trace_start
        logger.error(f"❌ Orchestrator exceeded max turns ({max_turns}) after {total_elapsed:.0f}s")
        raise RuntimeError(f"Orchestrator exceeded max turns ({max_turns})")

    async def trace(self, config: TracerConfig) -> TraceResult:
        """Run a trace using HTTP client."""
        # 0. Setup
        case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_id = f"http-{uuid.uuid4().hex[:12]}"

        # Extract victim from tx_hash when possible
        token_id_hint: Optional[int] = None
        if not config.victim_address and config.tx_hash:
            logger.debug(f"Extracting victim address from tx_hash: {config.tx_hash}")
            victim_addr, extracted_token_id, extracted_asset, block_time = await extract_victim_from_tx_hash(
                config.tx_hash, config.blockchain_name, self.client
            )
            config.victim_address = victim_addr
            token_id_hint = extracted_token_id
            if extracted_asset and not config.asset_symbol:
                config.asset_symbol = extracted_asset
            if block_time and not config.approx_date:
                try:
                    dt = datetime.fromtimestamp(block_time)
                    config.approx_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        if not config.victim_address:
            raise ValueError("victim_address is required. Provide tx_hash or wallet address.")

        if not config.approx_date and config.description:
            config.approx_date = infer_approx_date_from_description(config.description)

        asset_symbol, detected_token_id = await infer_asset_symbol(config, self.client)
        config.asset_symbol = asset_symbol
        if token_id_hint is None:
            token_id_hint = detected_token_id

        case_meta = CaseMeta(
            case_id=case_id,
            trace_id=trace_id,
            description=config.description or "",
            victim_address=config.victim_address,
            blockchain_name=config.blockchain_name,
            chains=[config.blockchain_name],
            asset_symbol=asset_symbol,
            approx_date=config.approx_date,
        )

        prompt_template = self._load_prompt()
        prompt_text = prompt_template.format(
            victim_address=config.victim_address or "",
            tx_hash=config.tx_hash or "",
            blockchain_name=config.blockchain_name,
            asset_symbol=config.asset_symbol or "",
            approx_date=config.approx_date or "",
            description=config.description or "",
        )

        payload = {
            "case_meta": case_meta.model_dump(),
            "token_id_hint": token_id_hint,
            "known_tx_hashes": config.known_tx_hashes,
            "inputs": {
                "victim_address": config.victim_address,
                "tx_hash": config.tx_hash,
                "blockchain_name": config.blockchain_name,
                "asset_symbol": config.asset_symbol,
                "approx_date": config.approx_date,
                "description": config.description,
            },
            "rules_version": "orchestrator-http-1",
        }

        logger.debug(f"Prompt loaded from {self.prompt_path}")
        logger.debug(
            "LLM input context: victim_address=%s tx_hash=%s blockchain=%s asset=%s approx_date=%s",
            config.victim_address,
            config.tx_hash,
            config.blockchain_name,
            config.asset_symbol,
            config.approx_date,
        )

        llm_output = await self._run_orchestrator(prompt_text, payload)

        try:
            trace_result = TraceResult.model_validate(llm_output)
        except Exception as exc:
            raise ValueError(f"LLM output could not be parsed into TraceResult: {exc}") from exc

        # Attach case meta if missing fields
        trace_result.case_meta = trace_result.case_meta or case_meta
        if not trace_result.case_meta.trace_id:
            trace_result.case_meta.trace_id = trace_id

        # --- Visualization Generation & Saving ---
        try:
            logger.info("🎨 Generating visualization...")
            viz_payload = generate_visualization_payload(trace_result)
            
            logger.info("💾 Saving visualization...")
            viz_result = await self.client.save_and_share_visualization(viz_payload)
            
            share_url = viz_result.get("share_url")
            if share_url:
                logger.info(f"🔗 Visualization Link: {share_url}")
                trace_result.visualization_url = share_url
            else:
                logger.warning("Visualization saved but no share URL returned")
                
        except Exception as e:
            logger.error(f"❌ Failed to generate/save visualization: {e}")
            # Don't fail the whole trace just because viz failed
            pass

        return trace_result
