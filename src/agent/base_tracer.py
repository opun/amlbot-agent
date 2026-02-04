"""
Base Tracer with shared orchestration logic.
Interface implementations: MCPTracer (local stdio), HTTPTracer (remote HTTP).
"""
import json
import uuid
import logging
import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable
from types import SimpleNamespace
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
from agents import generation_span, function_span
from httpx import Timeout, Limits

from .models import (
    TracerConfig,
    TraceResult,
    CaseMeta,
)
from .theft_detection import (
    infer_asset_symbol,
    infer_approx_date_from_description,
    extract_victim_from_tx_hash,
)
from .trace_postprocess import postprocess_trace_result
from .config import ModelConfig

logger = logging.getLogger("tracer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[TRACE] %(message)s"))
    logger.addHandler(handler)


# ─── Constants ────────────────────────────────────────────────────────────────
OPENAI_TIMEOUT = 90
OPENAI_CONNECT_TIMEOUT = 10
TOOL_TIMEOUT = 30
MAX_TOOL_CALLS_PER_TURN = 6
MAX_TOKEN_TRANSFERS_PER_TURN = 2


# ─── Tool Definitions (OpenAI function calling format) ────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "expert_search",
            "description": "Search for an address OR transaction hash in the explorer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {"type": "string", "description": "Address or tx hash to search"},
                    "filter": {"type": "string", "description": "Filter: 'explorer' or 'entity'", "default": "explorer"}
                },
                "required": ["hash"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_address",
            "description": "Get info about a blockchain wallet ADDRESS. NOT for tx hashes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blockchain_name": {"type": "string", "description": "Network (eth, trx, btc)"},
                    "address": {"type": "string", "description": "Wallet ADDRESS (42 chars ETH, 34 chars TRON)"}
                },
                "required": ["blockchain_name", "address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "token_stats",
            "description": "Get token statistics for a wallet ADDRESS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blockchain_name": {"type": "string", "description": "Blockchain network"},
                    "address": {"type": "string", "description": "Wallet ADDRESS"}
                },
                "required": ["blockchain_name", "address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "all_txs",
            "description": "Get all transactions for a wallet ADDRESS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Wallet ADDRESS"},
                    "blockchain_name": {"type": "string", "description": "Blockchain network"},
                    "filter": {"type": "object", "description": "Filter criteria"},
                    "limit": {"type": "integer", "description": "Max txs", "default": 20},
                    "offset": {"type": "integer", "description": "Offset", "default": 0},
                    "direction": {"type": "string", "description": "'asc' or 'desc'", "default": "asc"},
                    "order": {"type": "string", "description": "Order field", "default": "time"},
                    "transaction_type": {"type": "string", "default": "all"}
                },
                "required": ["address", "blockchain_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction",
            "description": "Get detailed tx info. Use when you have a TRANSACTION HASH.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Related wallet ADDRESS"},
                    "tx_hash": {"type": "string", "description": "Transaction HASH"},
                    "blockchain_name": {"type": "string", "description": "Network"},
                    "token_id": {"type": "integer", "description": "Token ID (0=native)", "default": 0},
                    "path": {"type": "string", "description": "Internal path", "default": "0"}
                },
                "required": ["address", "tx_hash", "blockchain_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_position",
            "description": "Get position info for a tx. Returns prev/next links for tracing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Wallet ADDRESS"},
                    "tx_hash": {"type": "string", "description": "Transaction HASH"},
                    "blockchain_name": {"type": "string", "description": "Network"},
                    "token_id": {"type": "integer", "default": 0},
                    "path": {"type": "string", "default": "0"}
                },
                "required": ["address", "tx_hash", "blockchain_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_extra_address_info",
            "description": "Get extra info for ADDRESS including tags and risk score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "Wallet ADDRESS"},
                    "asset": {"type": "string", "description": "Asset symbol (ETH, USDT)"}
                },
                "required": ["address", "asset"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bridge_analyze",
            "description": "Analyze tx to detect cross-chain bridge operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain": {"type": "string", "description": "Source chain"},
                    "tx_hash": {"type": "string", "description": "Transaction HASH"}
                },
                "required": ["chain", "tx_hash"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "token_transfers",
            "description": "Get token transfers for a transaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_hash": {"type": "string", "description": "Transaction HASH"},
                    "blockchain_name": {"type": "string", "description": "Blockchain network"}
                },
                "required": ["tx_hash", "blockchain_name"]
            }
        }
    }
]


class BaseTracer(ABC):
    """
    Abstract base class for crypto tracers.
    Contains shared orchestration logic - subclasses implement tool execution.
    """

    def __init__(self):
        import httpx
        http_client = httpx.AsyncClient(
            timeout=Timeout(OPENAI_TIMEOUT, connect=OPENAI_CONNECT_TIMEOUT),
            limits=Limits(max_keepalive_connections=0, max_connections=10),
        )
        self.openai_client = AsyncOpenAI(http_client=http_client, max_retries=1)

        self.model_orchestrator = ModelConfig.ORCHESTRATOR_MODEL
        self.model_selector = ModelConfig.SELECTOR_MODEL
        self.model_validator = ModelConfig.VALIDATOR_MODEL
        self.model_json_retry = ModelConfig.JSON_RETRY_MODEL

        self.prompt_path = Path(__file__).parent / "prompts" / "trace_orchestrator.md"
        self.validator_prompt_path = Path(__file__).parent / "prompts" / "trace_validator.md"
        self.selector_prompt_path = Path(__file__).parent / "prompts" / "trace_selector.md"

        # Result storage for post-trace access
        self.last_txs: List[Dict[str, Any]] = []
        self.last_tx_list: List[Dict[str, Any]] = []

    # ─── Abstract methods (implemented by subclasses) ─────────────────────────

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call. Subclasses implement the transport (HTTP, MCP, etc)."""
        pass

    # ─── Prompt loading ───────────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    def _load_validator_prompt(self) -> str:
        if not self.validator_prompt_path.exists():
            raise FileNotFoundError(f"Validator prompt not found: {self.validator_prompt_path}")
        return self.validator_prompt_path.read_text(encoding="utf-8")

    def _load_selector_prompt(self) -> str:
        if not self.selector_prompt_path.exists():
            raise FileNotFoundError(f"Selector prompt not found: {self.selector_prompt_path}")
        return self.selector_prompt_path.read_text(encoding="utf-8")

    # ─── Helpers ──────────────────────────────────────────────────────────────

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
        if len(addr) > 16:
            return f"{addr[:8]}...{addr[-6:]}"
        return addr

    def _format_hash(self, tx_hash: str) -> str:
        if len(tx_hash) > 18:
            return f"{tx_hash[:10]}...{tx_hash[-6:]}"
        return tx_hash

    def _coerce_message_dict(self, msg: Any) -> Dict[str, Any]:
        if isinstance(msg, dict):
            return msg
        if hasattr(msg, "model_dump"):
            try:
                dumped = msg.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass
        role = getattr(msg, "role", None) or "assistant"
        content = getattr(msg, "content", None)
        if content is None:
            content = str(msg)
        return {"role": role, "content": content}

    def _serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        return [self._coerce_message_dict(m) for m in messages]

    def _normalize_usage(self, usage_obj: Any) -> Optional[Dict[str, Any]]:
        input_tokens = 0
        output_tokens = 0
        if usage_obj is not None:
            usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }

    def _coerce_tool_call(self, tool_call: Any) -> Any:
        if isinstance(tool_call, dict):
            fn = tool_call.get("function") or {}
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, dict):
                args = json.dumps(args)
            return SimpleNamespace(
                id=tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                function=SimpleNamespace(
                    name=fn.get("name"),
                    arguments=args,
                ),
            )
        return tool_call

    def _tool_call_to_dict(self, tool_call: Any) -> Dict[str, Any]:
        if isinstance(tool_call, dict):
            fn = tool_call.get("function") or {}
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, dict):
                args = json.dumps(args)
            return {
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fn.get("name"),
                    "arguments": args,
                },
            }
        fn = getattr(tool_call, "function", None)
        name = getattr(fn, "name", None) if fn else None
        args = getattr(fn, "arguments", None) if fn else None
        if isinstance(args, dict):
            args = json.dumps(args)
        return {
            "id": getattr(tool_call, "id", None) or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        }

    def _extract_tool_calls(self, choice: Any, message: Any, finish_reason: str) -> List[Any]:
        tool_calls: List[Any] = []
        choice_dump: Optional[Dict[str, Any]] = None

        try:
            raw_calls = getattr(message, "tool_calls", None)
            if raw_calls:
                tool_calls = list(raw_calls)
        except Exception:
            pass

        if not tool_calls:
            try:
                choice_dump = choice.model_dump()
                raw_calls = choice_dump.get("message", {}).get("tool_calls")
                if raw_calls:
                    tool_calls = raw_calls
            except Exception:
                choice_dump = None

        if not tool_calls:
            try:
                fc = getattr(message, "function_call", None)
                if fc:
                    fc_name = getattr(fc, "name", None) if not isinstance(fc, dict) else fc.get("name")
                    fc_args = getattr(fc, "arguments", None) if not isinstance(fc, dict) else fc.get("arguments")
                    tool_calls = [{
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "function": {"name": fc_name, "arguments": fc_args},
                    }]
                elif choice_dump:
                    fc_dict = choice_dump.get("message", {}).get("function_call")
                    if fc_dict:
                        tool_calls = [{
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "function": {"name": fc_dict.get("name"), "arguments": fc_dict.get("arguments")},
                        }]
            except Exception:
                pass

        normalized = [self._coerce_tool_call(tc) for tc in tool_calls] if tool_calls else []

        if finish_reason == "tool_calls" and not normalized:
            try:
                message_dump = message.model_dump() if hasattr(message, "model_dump") else {}
            except Exception:
                message_dump = {}
            keys = list(message_dump.keys()) if isinstance(message_dump, dict) else []
            logger.warning("finish_reason=tool_calls but no tool_calls found. message_keys=%s", keys)
            if choice_dump:
                try:
                    preview = json.dumps(choice_dump, ensure_ascii=False)[:1500]
                    logger.debug("choice_dump=%s", preview)
                except Exception:
                    pass

        return normalized

    def _max_tokens_arg(self, model: str, max_tokens: int) -> Dict[str, int]:
        return {}

    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"entities": 0, "paths": 0, "txs": 0}
        entities = payload.get("entities")
        paths = payload.get("paths")
        txs = payload.get("txs") or payload.get("transactions")
        return {
            "entities": len(entities) if isinstance(entities, list) else 0,
            "paths": len(paths) if isinstance(paths, list) else 0,
            "txs": len(txs) if isinstance(txs, list) else 0,
        }

    def _trim_messages(self, messages: List[Dict[str, Any]], max_messages: int = 12) -> List[Dict[str, Any]]:
        if len(messages) <= max_messages:
            return messages
        if len(messages) <= 2:
            return messages

        system_msg = messages[0]
        user_msg = messages[1]

        last_tool_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
                last_tool_idx = i
                break

        if last_tool_idx is None:
            tail = messages[-(max_messages - 2):]
            return [system_msg, user_msg] + tail

        tail = messages[last_tool_idx:]
        return [system_msg, user_msg] + tail

    def _compact_tool_result(self, tool_name: str, result: Any) -> Any:
        """Reduce tool payload size to avoid LLM overload."""
        if not isinstance(result, dict):
            return result

        data = result.get("data") if isinstance(result.get("data"), list) else None

        if tool_name == "all_txs" and data is not None:
            compact = []
            for item in data[:20]:
                compact.append({
                    "hash": item.get("hash") or item.get("tx_hash"),
                    "amount": item.get("amount") or item.get("amount_coerced"),
                    "block_time": item.get("block_time") or item.get("time"),
                    "token_id": item.get("token_id"),
                    "type": item.get("type"),
                })
            return {"data": compact}

        if tool_name == "token_transfers" and data is not None:
            compact = []
            for item in data[:10]:
                compact.append({
                    "input": item.get("input"),
                    "output": item.get("output"),
                    "amount": item.get("amount") or item.get("amount_coerced"),
                    "block_time": item.get("block_time"),
                    "token_id": item.get("token_id"),
                    "asset": item.get("asset") or item.get("symbol"),
                })
            return {"data": compact}

        if tool_name == "get_address":
            data_obj = result.get("data") if isinstance(result.get("data"), dict) else {}
            return {"data": {"owner": data_obj.get("owner"), "riskscore": data_obj.get("riskscore")}}

        if tool_name == "get_extra_address_info":
            data_obj = result.get("data") if isinstance(result.get("data"), dict) else {}
            return {"data": {"services": data_obj.get("services")}}

        return result

    # ─── Selector ─────────────────────────────────────────────────────────────

    async def _run_selector(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            messages = [
                {"role": "system", "content": self._load_selector_prompt()},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ]
            summary = self._summarize_payload(context)
            logger.info(f"[PROMPT=trace_selector] txs={summary['txs']}")
            with generation_span(
                input=self._serialize_messages(messages),
                model=self.model_selector,
                model_config={"purpose": "selector"},
            ) as gen_span:
                response = await self.openai_client.chat.completions.create(
                    model=self.model_selector,
                    messages=messages,
                    **self._max_tokens_arg(self.model_selector, 300)
                )
                try:
                    if hasattr(gen_span, "span_data"):
                        msg_dump = response.choices[0].message.model_dump()
                        gen_span.span_data.output = [msg_dump]
                        gen_span.span_data.usage = self._normalize_usage(response.usage)
                except Exception:
                    pass
            output = response.choices[0].message.content or ""
            return json.loads(self._strip_code_fences(output))
        except Exception as exc:
            logger.warning(f"Selector failed: {exc}")
            return None

    # ─── Validator ────────────────────────────────────────────────────────────

    async def _run_validator(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        summary = self._summarize_payload(payload)
        logger.info("[PROMPT=trace_validator] entities=%d paths=%d txs=%d", 
                    summary["entities"], summary["paths"], summary["txs"])
        messages = [
            {"role": "system", "content": self._load_validator_prompt()},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ]
        with generation_span(
            input=self._serialize_messages(messages),
            model=self.model_validator,
            model_config={"purpose": "validator"},
        ) as gen_span:
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=self.model_validator, messages=messages
                ),
                timeout=60.0
            )
            try:
                if hasattr(gen_span, "span_data"):
                    msg_dump = response.choices[0].message.model_dump()
                    gen_span.span_data.output = [msg_dump]
                    gen_span.span_data.usage = self._normalize_usage(response.usage)
            except Exception:
                pass
        output = response.choices[0].message.content or ""
        return json.loads(self._strip_code_fences(output))

    # ─── Orchestrator (main loop) ─────────────────────────────────────────────

    async def _run_orchestrator(
        self, prompt: str, payload: Dict[str, Any],
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Run the LLM orchestrator with function calling and return parsed JSON."""
        messages: List[Any] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ]
        payload_summary = self._summarize_payload(payload)
        logger.info("[PROMPT=trace_orchestrator] entities=%d paths=%d txs=%d",
                    payload_summary["entities"], payload_summary["paths"], payload_summary["txs"])

        logger.info("🚀 Starting trace orchestrator...")
        trace_start = time.time()

        max_turns = 100
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3
        addresses_traced: List[str] = []

        allowed_token_hashes: Optional[set] = None
        txs_collected: List[Dict[str, Any]] = []
        tx_list_collected: List[Dict[str, Any]] = []
        all_txs_map: Dict[str, Dict[str, Any]] = {}
        risk_map: Dict[str, float] = {}
        txs_seen: set = set()
        empty_tool_call_turns = 0

        try:
            for turn in range(max_turns):
                turn_start = time.time()
                logger.info(f"⏳ Turn {turn + 1}: Waiting for LLM decision...")

                try:
                    messages = self._trim_messages(messages)
                    msg_count = len(messages)
                    tool_msg_count = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
                    total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))

                    logger.info(
                        "[PROMPT=trace_orchestrator:turn_%d] messages=%d tool_results=%d chars=%d entities=%d paths=%d txs=%d",
                        turn + 1,
                        msg_count,
                        tool_msg_count,
                        total_chars,
                        payload_summary["entities"],
                        payload_summary["paths"],
                        payload_summary["txs"],
                    )

                    with generation_span(
                        input=self._serialize_messages(messages),
                        model=self.model_orchestrator,
                        model_config={"tool_choice": "auto"},
                    ) as gen_span:
                        response = await asyncio.wait_for(
                            self.openai_client.chat.completions.create(
                                model=self.model_orchestrator,
                                messages=messages,
                                tools=TOOLS,
                                tool_choice="auto",
                                **self._max_tokens_arg(self.model_orchestrator, 1200)
                            ),
                            timeout=OPENAI_TIMEOUT + 10
                        )
                        try:
                            if hasattr(gen_span, "span_data"):
                                msg_dump = response.choices[0].message.model_dump()
                                gen_span.span_data.output = [msg_dump]
                                gen_span.span_data.usage = self._normalize_usage(response.usage)
                        except Exception:
                            pass
                    consecutive_timeouts = 0

                except (asyncio.TimeoutError, APITimeoutError, APIConnectionError) as e:
                    consecutive_timeouts += 1
                    logger.warning(f"⚠️ Turn {turn + 1}: API error: {type(e).__name__}")
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        raise RuntimeError(f"API timed out {max_consecutive_timeouts} times consecutively")
                    await asyncio.sleep(2)
                    continue

                if on_progress:
                    await on_progress(f"Analyzing hop {len(addresses_traced) + 1}...")

                turn_elapsed = time.time() - turn_start
                choice = response.choices[0]
                message = choice.message
                finish_reason = choice.finish_reason

                logger.info(f"✅ Turn {turn + 1}: Response in {turn_elapsed:.1f}s (reason={finish_reason})")

                tool_calls = self._extract_tool_calls(choice, message, finish_reason)
                if finish_reason == "tool_calls" and not tool_calls:
                    empty_tool_call_turns += 1
                    logger.warning("finish_reason=tool_calls but no tool_calls. retrying=%d", empty_tool_call_turns)
                    if empty_tool_call_turns >= 2:
                        raise RuntimeError("LLM returned finish_reason=tool_calls without tool_calls payload.")
                    await asyncio.sleep(1)
                    continue

                if tool_calls:
                    empty_tool_call_turns = 0
                    tool_names = [getattr(tc.function, "name", None) for tc in tool_calls if getattr(tc, "function", None)]
                    tool_names = [name for name in tool_names if name]
                    logger.info(f"🔧 LLM requesting {len(tool_names)} tool(s): {', '.join(tool_names)}")

                    assistant_msg: Dict[str, Any]
                    if hasattr(message, "model_dump"):
                        try:
                            assistant_msg = message.model_dump()
                        except Exception:
                            assistant_msg = {"role": "assistant", "content": message.content or ""}
                    else:
                        assistant_msg = {"role": "assistant", "content": message.content or ""}

                    if not isinstance(assistant_msg, dict):
                        assistant_msg = {"role": "assistant", "content": message.content or ""}

                    if not assistant_msg.get("tool_calls"):
                        assistant_msg["tool_calls"] = [self._tool_call_to_dict(tc) for tc in tool_calls]

                    messages.append(assistant_msg)

                    # Cap tool calls per turn
                    skipped_tool_calls = []
                    if len(tool_calls) > MAX_TOOL_CALLS_PER_TURN:
                        skipped_tool_calls = tool_calls[MAX_TOOL_CALLS_PER_TURN:]
                        tool_calls = tool_calls[:MAX_TOOL_CALLS_PER_TURN]
                        logger.warning(f"⚠️ Tool cap reached. Skipping {len(skipped_tool_calls)} calls.")

                    # Cap token_transfers per turn
                    filtered_calls = []
                    token_count = 0
                    for tc in tool_calls:
                        if tc.function.name == "token_transfers":
                            if token_count < MAX_TOKEN_TRANSFERS_PER_TURN:
                                filtered_calls.append(tc)
                                token_count += 1
                            else:
                                skipped_tool_calls.append(tc)
                        else:
                            filtered_calls.append(tc)
                    tool_calls = filtered_calls

                    # Enforce selector hashes (never block the primary tx_hash)
                    if allowed_token_hashes:
                        filtered_calls = []
                        primary_tx_hash = payload.get("inputs", {}).get("tx_hash")
                        for tc in tool_calls:
                            if tc.function.name == "token_transfers":
                                try:
                                    args = json.loads(tc.function.arguments)
                                    tx_hash = args.get("tx_hash")
                                    if tx_hash == primary_tx_hash or tx_hash in allowed_token_hashes:
                                        filtered_calls.append(tc)
                                    else:
                                        skipped_tool_calls.append(tc)
                                except Exception:
                                    filtered_calls.append(tc)
                            else:
                                filtered_calls.append(tc)
                        tool_calls = filtered_calls

                    # Emit skipped tool call errors
                    for skipped in skipped_tool_calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": skipped.id,
                            "content": json.dumps({"error": "tool_call_skipped", "tool": skipped.function.name})
                        })

                    all_txs_results: List[Dict[str, Any]] = []

                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        raw_args = tool_call.function.arguments
                        try:
                            arguments = json.loads(raw_args) if raw_args else {}
                        except Exception:
                            logger.warning(f"Invalid tool arguments for {tool_name}. Using empty args.")
                            arguments = {}

                        if tool_name == "get_address" and "address" in arguments:
                            addr = arguments["address"]
                            if addr not in addresses_traced:
                                addresses_traced.append(addr)
                                logger.info(f"📍 Hop {len(addresses_traced)}: Analyzing {self._format_address(addr)}")

                        tool_input = json.dumps(arguments, ensure_ascii=False)
                        tool_input = tool_input[:2000] if len(tool_input) > 2000 else tool_input
                        with function_span(tool_name, input=tool_input) as tool_span:
                            try:
                                result = await asyncio.wait_for(
                                    self.execute_tool(tool_name, arguments),
                                    timeout=TOOL_TIMEOUT
                                )
                                compact = self._compact_tool_result(tool_name, result)
                                tool_result = json.dumps(compact, ensure_ascii=False)
                                try:
                                    if hasattr(tool_span, "span_data"):
                                        tool_span.span_data.output = compact
                                except Exception:
                                    pass
                            except asyncio.TimeoutError:
                                logger.error(f"❌ Tool timeout: {tool_name}")
                                tool_result = json.dumps({"error": "tool_timeout", "tool": tool_name})
                                try:
                                    tool_span.set_error({"message": "tool_timeout", "data": {"tool": tool_name}})
                                    tool_span.span_data.output = {"error": "tool_timeout", "tool": tool_name}
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.error(f"❌ Tool error: {e}")
                                tool_result = json.dumps({"error": str(e), "tool": tool_name})
                                try:
                                    tool_span.set_error({"message": str(e), "data": {"tool": tool_name}})
                                    tool_span.span_data.output = {"error": str(e), "tool": tool_name}
                                except Exception:
                                    pass

                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                        # Collect tx data for visualization
                        if tool_name == "all_txs":
                            try:
                                parsed = json.loads(tool_result)
                                all_txs_results.append(parsed)
                                data_list = parsed.get("data", [])
                                if isinstance(data_list, list):
                                    for item in data_list:
                                        tx_hash = item.get("hash")
                                        if tx_hash:
                                            all_txs_map[tx_hash] = item
                            except Exception:
                                pass
                        elif tool_name == "get_address":
                            try:
                                parsed = json.loads(tool_result)
                                data_obj = parsed.get("data", {}) if isinstance(parsed, dict) else {}
                                riskscore = data_obj.get("riskscore") or data_obj.get("risk_score")
                                if isinstance(riskscore, dict):
                                    risk_val = riskscore.get("value", 0.0)
                                else:
                                    risk_val = riskscore or 0.0
                                address = arguments.get("address")
                                if address:
                                    risk_map[address] = float(risk_val) if risk_val is not None else 0.0
                            except Exception:
                                pass
                        elif tool_name == "token_transfers":
                            self._collect_token_transfer_data(
                                tool_result, arguments, all_txs_map, risk_map, 
                                txs_collected, tx_list_collected, txs_seen
                            )

                    # Run selector after all_txs
                    if all_txs_results:
                        selector_context = {
                            "chain": payload.get("case_meta", {}).get("blockchain_name"),
                            "asset": payload.get("case_meta", {}).get("asset_symbol"),
                            "txs": all_txs_results[0].get("data") if isinstance(all_txs_results[0], dict) else []
                        }
                        selector_result = await self._run_selector(selector_context)
                        if selector_result and isinstance(selector_result, dict):
                            selected = selector_result.get("selected_hashes") or []
                            if selected:
                                allowed_token_hashes = set(selected)
                            messages.append({"role": "assistant", "content": f"SELECTOR_RESULT: {json.dumps(selector_result)}"})

                    if turn > 0 and turn % 5 == 0:
                        total_elapsed = time.time() - trace_start
                        logger.info(f"📊 Progress: {turn + 1} turns, {len(addresses_traced)} addresses, {total_elapsed:.0f}s")
                else:
                    # Final response
                    total_elapsed = time.time() - trace_start
                    raw_output = message.content or ""

                    logger.info("🎉 Trace completed!")
                    logger.info(f"   └─ Turns: {turn + 1}, Addresses: {len(addresses_traced)}, Time: {total_elapsed:.1f}s")
                    if addresses_traced:
                        logger.info(f"   └─ Path: {' → '.join(self._format_address(a) for a in addresses_traced)}")

                    if not raw_output.strip() or finish_reason == "length":
                        return await self._retry_json_response(messages, message)

                    cleaned = self._strip_code_fences(raw_output)
                    if not cleaned.strip().startswith("{"):
                        raise ValueError(f"Response is not JSON: {cleaned[:100]}...")

                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error: {e}")
                        return await self._retry_json_response(messages, message)

            raise RuntimeError(f"Orchestrator exceeded max turns ({max_turns})")
        finally:
            self.last_txs = txs_collected
            self.last_tx_list = tx_list_collected
            logger.info(f"TXS_ARRAY={json.dumps(self.last_txs, ensure_ascii=False)}")
            logger.info(f"TXLIST_ARRAY={json.dumps(self.last_tx_list, ensure_ascii=False)}")

    def _collect_token_transfer_data(
        self, tool_result: str, arguments: Dict[str, Any],
        all_txs_map: Dict, risk_map: Dict,
        txs_collected: List, tx_list_collected: List, txs_seen: set
    ):
        """Helper to collect token transfer data for visualization."""
        try:
            parsed = json.loads(tool_result)
            transfers = parsed.get("data", []) if isinstance(parsed, dict) else []
            if isinstance(transfers, list) and transfers:
                transfer = transfers[0]
                input_data = transfer.get("input") or {}
                output_data = transfer.get("output") or {}
                from_addr = input_data.get("address") if isinstance(input_data, dict) else None
                to_addr = output_data.get("address") if isinstance(output_data, dict) else None

                tx_hash = arguments.get("tx_hash")
                chain = arguments.get("blockchain_name")
                tx_info = all_txs_map.get(tx_hash, {})
                token_id = tx_info.get("token_id") or transfer.get("token_id") or 0
                amount = tx_info.get("amount") or transfer.get("amount") or 0
                block_time = tx_info.get("block_time") or transfer.get("block_time") or 0

                if tx_hash and tx_hash not in txs_seen:
                    idx = len(txs_collected)
                    txs_seen.add(tx_hash)
                    txs_collected.append({
                        "currency": chain,
                        "descriptor": f"{tx_hash}-{chain}-{token_id}-{idx}",
                        "hash": tx_hash,
                        "token_id": token_id,
                        "x": 100 + idx * 40,
                        "y": 100 + idx * 40,
                        "color": "#EC292C",
                        "path": "0",
                        "type": "txEth"
                    })

                if tx_hash and from_addr and to_addr:
                    riskscore_from = risk_map.get(from_addr, 0.0)
                    riskscore_to = risk_map.get(to_addr, 0.0)
                    amount_val = float(amount) if amount is not None else 0.0
                    if chain == "trx":
                        amount_val = int(amount_val * 1e6)

                    tx_list_collected.append({
                        "inputs": [{"address": from_addr, "riskscore": riskscore_from}],
                        "outputs": [{"address": to_addr, "riskscore": riskscore_to}],
                        "hash": tx_hash,
                        "fiatRate": 1.0,
                        "addressesCount": 2,
                        "amount": amount_val,
                        "currency": chain,
                        "tokenId": token_id,
                        "poolTime": block_time,
                        "date": block_time,
                        "path": "0",
                        "type": "txEth"
                    })
        except Exception:
            pass

    async def _retry_json_response(self, messages: List, message) -> Dict[str, Any]:
        """Retry to get valid JSON from LLM after failed parse."""
        logger.info("[PROMPT=json_completion_retry]")
        messages.append(self._coerce_message_dict(message))
        messages.append({
            "role": "user",
            "content": "Your response was not valid JSON. Please provide the complete TraceResult as valid JSON only. No markdown, no explanations."
        })
        try:
            with generation_span(
                input=self._serialize_messages(messages),
                model=self.model_json_retry,
                model_config={"purpose": "json_retry"},
            ) as gen_span:
                retry_response = await asyncio.wait_for(
                    self.openai_client.chat.completions.create(
                        model=self.model_json_retry,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="none",
                        **self._max_tokens_arg(self.model_json_retry, 1200)
                    ),
                    timeout=60.0
                )
                try:
                    if hasattr(gen_span, "span_data"):
                        msg_dump = retry_response.choices[0].message.model_dump()
                        gen_span.span_data.output = [msg_dump]
                        gen_span.span_data.usage = self._normalize_usage(retry_response.usage)
                except Exception:
                    pass
            retry_output = retry_response.choices[0].message.content or ""
            retry_cleaned = self._strip_code_fences(retry_output)
            return json.loads(retry_cleaned)
        except Exception as e:
            raise ValueError(f"Failed to get valid JSON after retry: {e}") from e

    # ─── Main trace entry point ───────────────────────────────────────────────

    async def trace(
        self, config: TracerConfig,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> TraceResult:
        """Run a trace."""
        case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        if on_progress:
            await on_progress("Analyzing transaction context...")

        # Extract victim from tx_hash when possible
        token_id_hint: Optional[int] = None
        if not config.victim_address and config.tx_hash:
            logger.debug(f"Extracting victim from tx_hash: {config.tx_hash}")
            victim_addr, extracted_token_id, extracted_asset, block_time = await extract_victim_from_tx_hash(
                config.tx_hash, config.blockchain_name, self._get_client()
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
            raise ValueError("victim_address is required")

        if not config.approx_date and config.description:
            config.approx_date = infer_approx_date_from_description(config.description)

        asset_symbol, detected_token_id = await infer_asset_symbol(config, self._get_client())
        config.asset_symbol = asset_symbol.upper() if asset_symbol else asset_symbol
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
        asset_for_prompt = (config.theft_asset or config.asset_symbol or "").upper()
        blockchain_for_prompt = (config.blockchain_name or "eth").lower()
        prompt_text = prompt_template.format(
            victim_address=config.victim_address or "",
            tx_hash=config.tx_hash or "",
            blockchain_name=blockchain_for_prompt,
            asset_symbol=asset_for_prompt,
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
                "asset_symbol": asset_for_prompt,
                "approx_date": config.approx_date,
                "description": config.description,
            },
            "rules_version": "orchestrator-unified-1",
        }

        if on_progress:
            await on_progress("Starting trace orchestrator...")

        llm_output = await self._run_orchestrator(prompt_text, payload, on_progress=on_progress)

        try:
            if on_progress:
                await on_progress("Validating results...")
            validated = await self._run_validator(llm_output)
        except Exception as exc:
            logger.warning(f"Validator failed, using raw output: {exc}")
            validated = llm_output

        try:
            trace_result = TraceResult.model_validate(validated)
        except Exception as exc:
            raise ValueError(f"LLM output could not be parsed: {exc}") from exc

        trace_result.case_meta = trace_result.case_meta or case_meta
        if not trace_result.case_meta.trace_id:
            trace_result.case_meta.trace_id = trace_id

        trace_result = postprocess_trace_result(trace_result)

        logger.info("🛑 Visualization generation disabled. Skipping save/share.")

        return trace_result

    @abstractmethod
    def _get_client(self):
        """Return the underlying client for theft_detection helpers."""
        pass
