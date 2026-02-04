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
from dataclasses import dataclass
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
from .visualization import generate_visualization_payload
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


@dataclass
class HopJob:
    path_id: str
    current_address: str
    incoming_tx_hash: Optional[str]
    incoming_amount: float
    incoming_time: Optional[int]
    chain: str
    asset: str
    token_id: int
    hop_index: int


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
        self.selector_prompt_path = Path(__file__).parent / "prompts" / "trace_hop_selector.md"
        self.hop_classifier_prompt_path = Path(__file__).parent / "prompts" / "trace_hop_classifier.md"

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

    def _load_hop_classifier_prompt(self) -> str:
        if not self.hop_classifier_prompt_path.exists():
            raise FileNotFoundError(f"Hop classifier prompt not found: {self.hop_classifier_prompt_path}")
        return self.hop_classifier_prompt_path.read_text(encoding="utf-8")

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

    def _flatten_strings(self, value: Any, limit: int = 200) -> List[str]:
        items: List[str] = []

        def _walk(obj: Any):
            if len(items) >= limit:
                return
            if isinstance(obj, str):
                items.append(obj)
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str):
                        items.append(k)
                    _walk(v)
                return
            if isinstance(obj, list):
                for v in obj:
                    _walk(v)

        _walk(value)
        return items

    def _extract_risk_score(self, result: Any) -> float:
        try:
            data_obj = result.get("data", {}) if isinstance(result, dict) else {}
            riskscore = data_obj.get("riskscore") or data_obj.get("risk_score")
            if isinstance(riskscore, dict):
                return float(riskscore.get("value", 0.0) or 0.0)
            return float(riskscore or 0.0)
        except Exception:
            return 0.0

    def _parse_transfer(
        self,
        result: Any,
        expected_from: Optional[str] = None,
        token_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        transfers = result.get("data", [])
        if not isinstance(transfers, list) or not transfers:
            return None

        def _amount(tr):
            amt = tr.get("amount") or tr.get("amount_coerced") or tr.get("value")
            try:
                return float(amt)
            except Exception:
                return 0.0

        candidates = []
        for tr in transfers:
            input_data = tr.get("input") or {}
            output_data = tr.get("output") or {}
            from_addr = input_data.get("address") if isinstance(input_data, dict) else tr.get("from")
            to_addr = output_data.get("address") if isinstance(output_data, dict) else tr.get("to")
            tid = tr.get("token_id") or tr.get("tokenId") or 0
            if token_id not in (None, 0) and tid not in (None, 0) and int(tid) != int(token_id):
                continue
            if expected_from and from_addr and from_addr != expected_from:
                continue
            candidates.append((tr, from_addr, to_addr, output_data))

        pool = candidates if candidates else [(tr, (tr.get("input") or {}).get("address") if isinstance(tr.get("input"), dict) else tr.get("from"),
                                              (tr.get("output") or {}).get("address") if isinstance(tr.get("output"), dict) else tr.get("to"),
                                              tr.get("output") or {}) for tr in transfers]

        if not pool:
            return None

        transfer, from_addr, to_addr, output_data = max(pool, key=lambda item: _amount(item[0]))
        amount = transfer.get("amount_coerced") or transfer.get("amount") or transfer.get("value") or 0.0
        try:
            amount = float(amount)
        except Exception:
            amount = 0.0
        block_time = transfer.get("block_time")
        token_id_val = transfer.get("token_id") or transfer.get("tokenId") or 0
        output_owner = output_data.get("owner") if isinstance(output_data, dict) else None

        return {
            "from": from_addr,
            "to": to_addr,
            "amount": amount,
            "block_time": block_time,
            "token_id": token_id_val,
            "output_owner": output_owner,
            "input_riskscore": (transfer.get("input") or {}).get("riskscore") if isinstance(transfer.get("input"), dict) else None,
            "output_riskscore": (transfer.get("output") or {}).get("riskscore") if isinstance(transfer.get("output"), dict) else None,
        }

    def _parse_date_to_ts(self, date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            return int(dt.timestamp())
        except Exception:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return int(dt.timestamp())
            except Exception:
                return None

    def _normalize_amount(self, amount: Any, chain: str, asset: Optional[str] = None) -> float:
        try:
            val = float(amount)
        except Exception:
            return 0.0
        asset_upper = asset.upper() if isinstance(asset, str) else None
        six_dec_assets = {"USDT", "USDC", "TUSD", "USDP", "USDD", "BUSD"}
        if asset_upper in six_dec_assets and val >= 1e6:
            return val / 1e6
        if chain == "trx" and val >= 1e6:
            # TRX/TRC20 amounts are typically in base units (1e6).
            # If we get a value that still looks over-scaled, downscale further.
            scaled = val / 1e6
            if scaled >= 1e9:
                return scaled / 1e6
            return scaled
        return val

    def _resolve_amount(
        self,
        tx_hash: Optional[str],
        amount: Any,
        chain: str,
        all_txs_map: Dict[str, Dict[str, Any]],
        asset: Optional[str] = None,
    ) -> float:
        if tx_hash and tx_hash in all_txs_map:
            amt = all_txs_map[tx_hash].get("amount_coerced")
            if amt is None:
                amt = all_txs_map[tx_hash].get("amount")
            if amt is not None:
                return self._normalize_amount(amt, chain, asset)
        return self._normalize_amount(amount, chain, asset)

    def _heuristic_classify(self, owner: Any, services: Any, owner_hint: Any = None) -> Dict[str, Any]:
        owner_texts = self._flatten_strings(owner)
        hint_texts = self._flatten_strings(owner_hint)
        service_texts = self._flatten_strings(services)
        combined = " ".join(owner_texts + hint_texts + service_texts).lower()

        def _has_any(text: str, keywords: List[str]) -> bool:
            return any(k in text for k in keywords)

        bridge_keywords = [
            "bridge", "layerzero", "stargate", "wormhole", "allbridge",
            "synapse", "hop", "multichain", "bridger", "bridgers", "bridgers.xyz", "router"
        ]
        cex_keywords = ["exchange", "binance", "coinbase", "kraken", "okx", "huobi", "kucoin", "bybit", "gate"]
        dex_keywords = ["dex", "swap", "uniswap", "sushiswap", "pancakeswap", "curve"]
        mixer_keywords = ["mixer", "tornado", "blender"]
        otc_keywords = ["otc"]

        if _has_any(combined, mixer_keywords):
            return {"role": "unidentified_service", "terminal": True, "service_label": "Mixer", "protocol": None}
        if _has_any(combined, cex_keywords):
            return {"role": "cex_deposit", "terminal": True, "service_label": "Exchange", "protocol": None}
        if _has_any(combined, otc_keywords):
            return {"role": "otc_service", "terminal": True, "service_label": "OTC", "protocol": None}
        if _has_any(combined, bridge_keywords):
            return {"role": "bridge_service", "terminal": True, "service_label": "Bridge", "protocol": None}
        if _has_any(combined, dex_keywords):
            return {"role": "bridge_service", "terminal": True, "service_label": "DEX", "protocol": None}

        return {"role": "intermediate", "terminal": False, "service_label": None, "protocol": None}

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

    def _accumulate_hashes(
        self,
        txs: List[Dict[str, Any]],
        incoming_amount: Optional[float],
        chain: str,
        max_select: int = 25,
    ) -> List[str]:
        """Chronological accumulation per trace_orchestrator.md."""
        if not txs:
            return []
        if not incoming_amount or incoming_amount <= 0:
            first_hash = txs[0].get("hash") or txs[0].get("tx_hash")
            return [first_hash] if first_hash else []

        accumulated = 0.0
        incoming = float(incoming_amount)
        selected: List[str] = []

        for item in txs:
            tx_hash = item.get("hash") or item.get("tx_hash")
            if not tx_hash:
                continue
            amount_val = item.get("amount_coerced")
            if amount_val is None:
                amount_val = item.get("amount")
            amount_norm = self._normalize_amount(amount_val or 0.0, chain)
            accumulated += amount_norm
            selected.append(tx_hash)

            if accumulated >= incoming:
                break
            gap = (incoming - accumulated) / incoming if incoming else 1.0
            if gap <= 0.015:
                break
            if len(selected) >= max_select:
                break

        return selected

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

    async def _run_hop_classifier(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            messages = [
                {"role": "system", "content": self._load_hop_classifier_prompt()},
                {"role": "user", "content": json.dumps(context, indent=2)}
            ]
            logger.info("[PROMPT=hop_classifier] address=%s", context.get("address"))
            with generation_span(
                input=self._serialize_messages(messages),
                model=self.model_selector,
                model_config={"purpose": "hop_classifier"},
            ) as gen_span:
                response = await self.openai_client.chat.completions.create(
                    model=self.model_selector,
                    messages=messages,
                    **self._max_tokens_arg(self.model_selector, 250)
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
            logger.warning(f"Hop classifier failed: {exc}")
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

    async def _run_agentic_trace(
        self, payload: Dict[str, Any],
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """Agentic split prompts: selector + hop classifier, tool execution in code."""
        case_meta = payload.get("case_meta", {})
        inputs = payload.get("inputs", {})
        chain = (inputs.get("blockchain_name") or case_meta.get("blockchain_name") or "eth").lower()
        asset = (inputs.get("asset_symbol") or case_meta.get("asset_symbol") or "").upper()
        tx_hash = inputs.get("tx_hash")
        victim_address = inputs.get("victim_address")
        approx_date = inputs.get("approx_date")
        token_id_hint = payload.get("token_id_hint") or 0

        max_hops = 12
        max_paths = 3

        entities: Dict[str, Dict[str, Any]] = {}
        annotations: List[Dict[str, Any]] = []
        paths: Dict[str, Dict[str, Any]] = {}
        path_seen_addresses: Dict[str, set] = {}
        path_seen_hashes: Dict[str, set] = {}
        path_counter = 1

        all_txs_map: Dict[str, Dict[str, Any]] = {}
        risk_map: Dict[str, float] = {}
        owner_hints: Dict[str, Any] = {}
        txs_collected: List[Dict[str, Any]] = []
        tx_list_collected: List[Dict[str, Any]] = []
        txs_seen: set = set()

        async def _call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            tool_input = json.dumps(arguments, ensure_ascii=False)
            tool_input = tool_input[:2000] if len(tool_input) > 2000 else tool_input
            with function_span(tool_name, input=tool_input) as tool_span:
                try:
                    result = await asyncio.wait_for(
                        self.execute_tool(tool_name, arguments),
                        timeout=TOOL_TIMEOUT
                    )
                    try:
                        if hasattr(tool_span, "span_data"):
                            tool_span.span_data.output = self._compact_tool_result(tool_name, result)
                    except Exception:
                        pass
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"❌ Tool timeout: {tool_name}")
                    try:
                        tool_span.set_error({"message": "tool_timeout", "data": {"tool": tool_name}})
                        tool_span.span_data.output = {"error": "tool_timeout", "tool": tool_name}
                    except Exception:
                        pass
                    return {"error": "tool_timeout", "tool": tool_name}
                except Exception as e:
                    logger.error(f"❌ Tool error: {e}")
                    try:
                        tool_span.set_error({"message": str(e), "data": {"tool": tool_name}})
                        tool_span.span_data.output = {"error": str(e), "tool": tool_name}
                    except Exception:
                        pass
                    return {"error": str(e), "tool": tool_name}

        def _ensure_entity(address: str, role: str, risk_score: float = 0.0, labels: Optional[List[str]] = None, notes: Optional[str] = None):
            if not address:
                return
            current = entities.get(address)
            if current:
                # Keep higher-priority roles
                priority = {
                    "victim": 5,
                    "perpetrator": 4,
                    "bridge_service": 4,
                    "cex_deposit": 4,
                    "otc_service": 4,
                    "unidentified_service": 4,
                    "cluster": 3,
                    "intermediate": 1,
                }
                if priority.get(role, 1) > priority.get(current.get("role"), 1):
                    current["role"] = role
                if risk_score is not None:
                    current["risk_score"] = max(current.get("risk_score", 0.0), risk_score)
                if labels:
                    current["labels"] = list(set((current.get("labels") or []) + labels))
                if notes and not current.get("notes"):
                    current["notes"] = notes
                return
            entities[address] = {
                "address": address,
                "chain": chain,
                "role": role,
                "risk_score": float(risk_score or 0.0),
                "riskscore_signals": {},
                "labels": labels or [],
                "notes": notes,
            }

        def _add_step(path_id: str, step: Dict[str, Any]):
            paths[path_id]["steps"].append(step)
            tx_hash = step.get("tx_hash")
            if tx_hash:
                path_seen_hashes.setdefault(path_id, set()).add(tx_hash)

        def _copy_path(new_id: str, from_id: str):
            paths[new_id] = {
                "path_id": new_id,
                "description": paths[from_id]["description"],
                "steps": [dict(s) for s in paths[from_id]["steps"]],
                "stop_reason": None,
            }
            path_seen_addresses[new_id] = set(path_seen_addresses.get(from_id, set()))
            path_seen_hashes[new_id] = set(path_seen_hashes.get(from_id, set()))

        hop_queue: List[HopJob] = []

        async def _fetch_outgoing_txs(
            address: str,
            chain_name: str,
            incoming_time: Optional[int],
            token_id: Optional[int],
            max_pages: int = 5,
            page_limit: int = 50,
        ) -> List[Dict[str, Any]]:
            """Fetch outgoing txs with pagination, ordered by time asc."""
            all_items: List[Dict[str, Any]] = []
            offset = 0
            pages = 0
            while pages < max_pages:
                filter_obj: Dict[str, Any] = {}
                if incoming_time:
                    filter_obj["time"] = {">=": incoming_time}
                if token_id not in (None, 0):
                    filter_obj["token_id"] = [token_id]
                filter_arg = filter_obj or None

                result = await _call_tool("all_txs", {
                    "address": address,
                    "blockchain_name": chain_name,
                    "filter": filter_arg,
                    "limit": page_limit,
                    "offset": offset,
                    "direction": "asc",
                    "order": "time",
                    "transaction_type": "withdrawal",
                })
                data_list = result.get("data", []) if isinstance(result, dict) else []
                if not data_list:
                    break
                for item in data_list:
                    tx_h = item.get("hash")
                    if tx_h:
                        all_txs_map[tx_h] = item
                all_items.extend(data_list)
                if len(data_list) < page_limit:
                    break
                offset += page_limit
                pages += 1
            return all_items

        def _parse_bridge_info(result: Any) -> Dict[str, Any]:
            if not isinstance(result, dict):
                return {}

            data = result.get("data") if isinstance(result.get("data"), dict) else result
            if not isinstance(data, dict):
                return {}

            def _find_key(obj: Any, keys: set) -> Any:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in keys and v is not None:
                            return v
                        if isinstance(v, (dict, list)):
                            found = _find_key(v, keys)
                            if found is not None:
                                return found
                elif isinstance(obj, list):
                    for v in obj:
                        found = _find_key(v, keys)
                        if found is not None:
                            return found
                return None

            def _find_dest_obj(obj: Any) -> Optional[Dict[str, Any]]:
                if isinstance(obj, dict):
                    for key in ("destination", "dest", "dst", "destination_info", "dst_info"):
                        v = obj.get(key)
                        if isinstance(v, dict):
                            return v
                    for v in obj.values():
                        if isinstance(v, (dict, list)):
                            found = _find_dest_obj(v)
                            if found:
                                return found
                elif isinstance(obj, list):
                    for v in obj:
                        found = _find_dest_obj(v)
                        if found:
                            return found
                return None

            is_bridge_val = _find_key(data, {"is_bridge", "isBridge", "bridge_tx", "bridgeTx", "is_bridge_tx"})
            if isinstance(is_bridge_val, bool):
                is_bridge = is_bridge_val
            elif isinstance(is_bridge_val, str):
                is_bridge = is_bridge_val.strip().lower() in {"true", "yes", "1"}
            elif isinstance(is_bridge_val, (int, float)):
                is_bridge = bool(is_bridge_val)
            else:
                is_bridge = False

            dest_obj = _find_dest_obj(data)
            if dest_obj:
                dst_chain = dest_obj.get("chain") or dest_obj.get("dst_chain") or dest_obj.get("destination_chain")
                dst_addr = dest_obj.get("address") or dest_obj.get("destination_address") or dest_obj.get("dst_address")
            else:
                dst_chain = _find_key(data, {"dst_chain", "dest_chain", "destination_chain", "dstChain", "destinationChain"})
                dst_addr = _find_key(data, {"destination_address", "dst_address", "dstAddress", "destinationAddress", "address"})

            amount_out = _find_key(data, {"amount_out", "output_amount", "amount", "outputAmount"})
            dst_tx_hash = _find_key(data, {"dst_tx_hash", "destination_tx_hash", "dstTxHash"})
            dst_block_time = _find_key(data, {"dst_block_time", "destination_block_time", "dstBlockTime"})
            protocol = _find_key(data, {"protocol", "bridge", "service", "bridge_name"})

            if not is_bridge and (dst_chain or dst_addr):
                is_bridge = True

            return {
                "is_bridge": bool(is_bridge),
                "dst_chain": dst_chain,
                "dst_address": dst_addr,
                "amount_out": amount_out,
                "dst_tx_hash": dst_tx_hash,
                "dst_block_time": dst_block_time,
                "protocol": protocol,
            }

        async def _resolve_token_id_for_chain(
            chain_name: str,
            address: str,
            asset_symbol: str,
            fallback: Optional[int] = None,
        ) -> Optional[int]:
            if not asset_symbol or not address:
                return fallback
            try:
                stats = await _call_tool("token_stats", {
                    "blockchain_name": chain_name,
                    "address": address,
                })
                data_list = stats.get("data", []) if isinstance(stats, dict) else []
                for item in data_list:
                    symbol = item.get("symbol") or item.get("asset") or ""
                    if symbol and symbol.upper() == asset_symbol.upper():
                        token_id = item.get("token_id") or item.get("tokenId")
                        if token_id is not None:
                            return int(token_id)
            except Exception:
                return fallback
            return fallback

        # Seed path(s)
        if tx_hash:
            transfer_result = await _call_tool("token_transfers", {
                "tx_hash": tx_hash,
                "blockchain_name": chain
            })
            transfer = self._parse_transfer(transfer_result, token_id=token_id_hint)
            if not transfer or not transfer.get("from") or not transfer.get("to"):
                raise RuntimeError("Unable to extract theft transfer details from tx_hash")

            from_addr = transfer["from"]
            to_addr = transfer["to"]
            amount = self._resolve_amount(tx_hash, transfer.get("amount", 0.0), chain, all_txs_map, asset)
            block_time = transfer.get("block_time")
            token_id = transfer.get("token_id") or token_id_hint
            if transfer.get("output_owner"):
                owner_hints[to_addr] = transfer.get("output_owner")
            if transfer.get("input_riskscore") is not None:
                risk_map[from_addr] = float(transfer.get("input_riskscore") or 0.0)
            if transfer.get("output_riskscore") is not None:
                risk_map[to_addr] = float(transfer.get("output_riskscore") or 0.0)

            paths["1"] = {
                "path_id": "1",
                "description": "Primary theft flow",
                "steps": [],
                "stop_reason": None,
            }
            _ensure_entity(from_addr, "victim", 0.0, notes="Victim")
            _add_step("1", {
                "step_index": 0,
                "from": from_addr,
                "to": to_addr,
                "tx_hash": tx_hash,
                "chain": chain,
                "asset": asset,
                "amount_estimate": float(amount or 0.0),
                "time": block_time,
                "direction": "out",
                "step_type": "direct_transfer",
                "service_label": None,
                "protocol": None,
                "reasoning": "Primary theft transaction provided by user.",
            })

            self._collect_token_transfer_data(
                json.dumps(self._compact_tool_result("token_transfers", transfer_result), ensure_ascii=False),
                {"tx_hash": tx_hash, "blockchain_name": chain},
                all_txs_map,
                risk_map,
                txs_collected,
                tx_list_collected,
                txs_seen,
            )

            hop_queue.append(HopJob(
                path_id="1",
                current_address=to_addr,
                incoming_tx_hash=tx_hash,
                incoming_amount=float(amount or 0.0),
                incoming_time=block_time,
                chain=chain,
                asset=asset,
                token_id=int(token_id or 0),
                hop_index=1,
            ))
        elif victim_address:
            paths["1"] = {
                "path_id": "1",
                "description": "Primary theft flow",
                "steps": [],
                "stop_reason": None,
            }
            _ensure_entity(victim_address, "victim", 0.0, notes="Victim")
            date_ts = self._parse_date_to_ts(approx_date)
            time_filter = None
            if date_ts:
                seven_days = 7 * 24 * 3600
                time_filter = {">=": date_ts - seven_days, "<=": date_ts + seven_days}
            filter_obj: Dict[str, Any] = {}
            if time_filter:
                filter_obj["time"] = time_filter
            if token_id_hint:
                filter_obj["token_id"] = [token_id_hint]
            filter_obj = filter_obj or None

            data_list = await _fetch_outgoing_txs(
                victim_address,
                chain,
                date_ts,
                token_id_hint,
            )
            selected_hashes = self._accumulate_hashes(data_list, None, chain)
            used_accumulation = bool(selected_hashes)
            if not selected_hashes and data_list:
                # Fallback to selector only if accumulation can't decide
                selector_context = {
                    "chain": chain,
                    "asset": asset,
                    "incoming_amount": None,
                    "incoming_time": date_ts,
                    "txs": data_list,
                }
                selector_result = await self._run_selector(selector_context)
                selected_hashes = (selector_result or {}).get("selected_hashes") or []
                used_accumulation = False
            if not selected_hashes and data_list:
                first_hash = data_list[0].get("hash")
                selected_hashes = [first_hash] if first_hash else []
                used_accumulation = False

            if not used_accumulation:
                selected_hashes = selected_hashes[:max_paths]

            seen_recipients: set = set()
            for idx, sel_hash in enumerate(selected_hashes):
                transfer_result = await _call_tool("token_transfers", {
                    "tx_hash": sel_hash,
                    "blockchain_name": chain
                })
                transfer = self._parse_transfer(transfer_result, expected_from=victim_address, token_id=token_id_hint)
                if not transfer or not transfer.get("to"):
                    continue
                to_addr = transfer["to"]
                if to_addr in seen_recipients:
                    continue
                seen_recipients.add(to_addr)
                amount = self._resolve_amount(sel_hash, transfer.get("amount", 0.0), chain, all_txs_map, asset)
                block_time = transfer.get("block_time")
                token_id = transfer.get("token_id") or token_id_hint
                if transfer.get("output_owner"):
                    owner_hints[to_addr] = transfer.get("output_owner")
                if transfer.get("input_riskscore") is not None:
                    risk_map[victim_address] = float(transfer.get("input_riskscore") or 0.0)
                if transfer.get("output_riskscore") is not None:
                    risk_map[to_addr] = float(transfer.get("output_riskscore") or 0.0)

                path_id = "1" if idx == 0 else str(path_counter + idx)
                if path_id not in paths:
                    _copy_path(path_id, "1")
                _add_step(path_id, {
                    "step_index": 0,
                    "from": victim_address,
                    "to": to_addr,
                    "tx_hash": sel_hash,
                    "chain": chain,
                    "asset": asset,
                    "amount_estimate": float(amount or 0.0),
                    "time": block_time,
                    "direction": "out",
                    "step_type": "direct_transfer",
                    "service_label": None,
                    "protocol": None,
                    "reasoning": "Selected as primary theft candidate from victim outflows.",
                })

                self._collect_token_transfer_data(
                    json.dumps(self._compact_tool_result("token_transfers", transfer_result), ensure_ascii=False),
                    {"tx_hash": sel_hash, "blockchain_name": chain},
                    all_txs_map,
                    risk_map,
                    txs_collected,
                    tx_list_collected,
                    txs_seen,
                )

                hop_queue.append(HopJob(
                    path_id=path_id,
                    current_address=to_addr,
                    incoming_tx_hash=sel_hash,
                    incoming_amount=float(amount or 0.0),
                    incoming_time=block_time,
                    chain=chain,
                    asset=asset,
                    token_id=int(token_id or 0),
                    hop_index=1,
                ))

        else:
            raise RuntimeError("victim_address or tx_hash is required")

        processed_paths = 0
        while hop_queue and processed_paths < max_paths:
            job = hop_queue.pop(0)
            if job.hop_index > max_hops:
                if paths[job.path_id]["stop_reason"] is None:
                    paths[job.path_id]["stop_reason"] = "Max hop limit reached"
                processed_paths += 1
                continue

            if job.current_address in path_seen_addresses.get(job.path_id, set()):
                paths[job.path_id]["stop_reason"] = "Loop detected - address revisited"
                processed_paths += 1
                continue
            path_seen_addresses.setdefault(job.path_id, set()).add(job.current_address)

            if on_progress:
                await on_progress(f"Analyzing hop {job.hop_index + 1}...")

            # Classify current address
            get_addr_result = await _call_tool("get_address", {
                "blockchain_name": job.chain,
                "address": job.current_address,
            })
            get_extra_result = await _call_tool("get_extra_address_info", {
                "address": job.current_address,
                "asset": job.asset,
            })
            risk_score = self._extract_risk_score(get_addr_result)
            risk_map[job.current_address] = risk_score

            owner = None
            services = {}
            try:
                data_obj = get_addr_result.get("data", {}) if isinstance(get_addr_result, dict) else {}
                owner = data_obj.get("owner")
            except Exception:
                owner = None
            try:
                data_obj = get_extra_result.get("data", {}) if isinstance(get_extra_result, dict) else {}
                services = data_obj.get("services") or {}
            except Exception:
                services = {}

            owner_hint = owner_hints.get(job.current_address)
            heuristic = self._heuristic_classify(owner, services, owner_hint)

            classifier_context = {
                "address": job.current_address,
                "chain": job.chain,
                "asset": job.asset,
                "incoming_tx_hash": job.incoming_tx_hash,
                "incoming_amount": job.incoming_amount,
                "get_address": get_addr_result,
                "get_extra_address_info": get_extra_result,
                "owner_hint": owner_hint,
            }
            classification = await self._run_hop_classifier(classifier_context) or {}
            role = classification.get("role") or "intermediate"
            terminal = bool(classification.get("terminal"))
            stop_reason = classification.get("stop_reason")
            labels = classification.get("labels") or []
            notes = classification.get("notes")

            if heuristic.get("terminal"):
                terminal = True
                role = heuristic.get("role") or role
                if not stop_reason:
                    stop_reason = f"Reached {heuristic.get('service_label') or 'terminal'} service"
            if not classification.get("service_label") and heuristic.get("service_label"):
                classification["service_label"] = heuristic.get("service_label")
            if not classification.get("protocol") and heuristic.get("protocol"):
                classification["protocol"] = heuristic.get("protocol")

            # Enrich labels with known owner name if present.
            owner_name = None
            if isinstance(owner, dict):
                owner_name = owner.get("name") or owner.get("slug")
            if not owner_name and isinstance(owner_hint, dict):
                owner_name = owner_hint.get("name") or owner_hint.get("slug")
            if owner_name and owner_name not in labels:
                labels.append(str(owner_name))

            if risk_score and risk_score > 0.75 and "High Risk" not in labels:
                labels.append("High Risk")

            if job.hop_index == 1 and role == "intermediate" and not heuristic.get("terminal"):
                # First hop with no identity → suspect perpetrator
                role = "perpetrator"
                if "Suspected Perpetrator" not in labels:
                    labels.append("Suspected Perpetrator")

            _ensure_entity(job.current_address, role, risk_score, labels=labels, notes=notes)

            if risk_score and risk_score > 0.75:
                annotations.append({
                    "id": f"ann-{len(annotations)+1}",
                    "label": "High Risk",
                    "related_addresses": [job.current_address],
                    "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                    "text": f"Passed through high-risk address (score: {risk_score:.2f})",
                })

            # Bridge detection & continuation (probe early if needed)
            bridge_info = None
            if job.incoming_tx_hash:
                service_label = classification.get("service_label") or heuristic.get("service_label") or ""
                bridge_candidate = (
                    role == "bridge_service"
                    or heuristic.get("role") == "bridge_service"
                    or ("bridge" in str(service_label).lower())
                )
                # Probe a few early hops even if metadata is missing
                if bridge_candidate or job.hop_index <= 8:
                    bridge_result = await _call_tool("bridge_analyze", {
                        "chain": job.chain,
                        "tx_hash": job.incoming_tx_hash,
                    })
                    bridge_info = _parse_bridge_info(bridge_result)

            if bridge_info and bridge_info.get("is_bridge"):
                dst_chain = bridge_info.get("dst_chain")
                dst_address = bridge_info.get("dst_address")
                dst_tx_hash = bridge_info.get("dst_tx_hash")
                dst_block_time = bridge_info.get("dst_block_time")
                amount_out = bridge_info.get("amount_out")

                if dst_chain and dst_address:
                    # Promote to bridge service if tool confirms with destination
                    role = "bridge_service"
                    terminal = True
                    if "Bridge" not in labels:
                        labels.append("Bridge")
                    _ensure_entity(job.current_address, role, risk_score, labels=labels, notes=notes)

                    dst_chain_norm = str(dst_chain).lower()
                    bridge_amount = self._normalize_amount(
                        amount_out if amount_out is not None else (job.incoming_amount or 0.0),
                        dst_chain_norm,
                        job.asset,
                    )
                    if amount_out is not None and job.incoming_amount:
                        gap = abs(self._normalize_amount(amount_out, dst_chain_norm, job.asset) - float(job.incoming_amount or 0.0)) / max(float(job.incoming_amount or 1.0), 1.0)
                        if gap > 0.2:
                            annotations.append({
                                "id": f"ann-{len(annotations)+1}",
                                "label": "Bridge Aggregation",
                                "related_addresses": [job.current_address, dst_address],
                                "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                                "text": f"Bridge aggregation detected - output amount ({bridge_amount}) differs from input ({job.incoming_amount}).",
                            })

                    step_index = len(paths[job.path_id]["steps"])
                    _add_step(job.path_id, {
                        "step_index": step_index,
                        "from": job.current_address,
                        "to": dst_address,
                        "tx_hash": dst_tx_hash,
                        "chain": dst_chain_norm,
                        "asset": job.asset,
                        "amount_estimate": float(bridge_amount or 0.0),
                        "time": int(dst_block_time) if dst_block_time else job.incoming_time,
                        "direction": "out",
                        "step_type": "bridge_transfer",
                        "service_label": classification.get("service_label") or heuristic.get("service_label") or "Bridge",
                        "protocol": bridge_info.get("protocol") or classification.get("protocol"),
                        "reasoning": "Bridge detected; continuing on destination chain.",
                    })

                    new_token_id = await _resolve_token_id_for_chain(
                        dst_chain_norm,
                        dst_address,
                        job.asset,
                        fallback=job.token_id,
                    )

                    hop_queue.append(HopJob(
                        path_id=job.path_id,
                        current_address=dst_address,
                        incoming_tx_hash=dst_tx_hash or job.incoming_tx_hash,
                        incoming_amount=float(bridge_amount or 0.0),
                        incoming_time=int(dst_block_time) if dst_block_time else job.incoming_time,
                        chain=dst_chain_norm,
                        asset=job.asset,
                        token_id=int(new_token_id or job.token_id or 0),
                        hop_index=job.hop_index + 1,
                    ))
                    # Continue on destination chain
                    continue

                # If tool hints at bridge but no destination, only stop if metadata already says bridge.
                if role == "bridge_service" or heuristic.get("role") == "bridge_service":
                    paths[job.path_id]["stop_reason"] = "Reached bridge service - destination unknown"
                    processed_paths += 1
                    continue

            if terminal:
                paths[job.path_id]["stop_reason"] = stop_reason or "Reached terminal entity"
                processed_paths += 1
                continue

            # Find next outgoing transactions (chronological accumulation)
            data_list = await _fetch_outgoing_txs(
                job.current_address,
                job.chain,
                job.incoming_time,
                job.token_id,
            )
            if not data_list:
                paths[job.path_id]["stop_reason"] = "Dead end - no outgoing transactions"
                processed_paths += 1
                continue

            selector_result = None
            selected_hashes = self._accumulate_hashes(data_list, job.incoming_amount, job.chain)
            used_accumulation = bool(selected_hashes)
            if not selected_hashes and data_list:
                selector_context = {
                    "chain": job.chain,
                    "asset": job.asset,
                    "incoming_amount": job.incoming_amount,
                    "incoming_time": job.incoming_time,
                    "txs": data_list,
                }
                selector_result = await self._run_selector(selector_context)
                selected_hashes = (selector_result or {}).get("selected_hashes") or []
                used_accumulation = False
            if not selected_hashes and data_list:
                first_hash = data_list[0].get("hash")
                selected_hashes = [first_hash] if first_hash else []
                used_accumulation = False

            if not used_accumulation:
                selected_hashes = selected_hashes[:max_paths]
            base_path_id = job.path_id

            took_step = False
            seen_recipients: set = set()
            for idx, sel_hash in enumerate(selected_hashes):
                if sel_hash in path_seen_hashes.get(job.path_id, set()):
                    continue
                transfer_result = await _call_tool("token_transfers", {
                    "tx_hash": sel_hash,
                    "blockchain_name": job.chain
                })
                transfer = self._parse_transfer(transfer_result, expected_from=job.current_address, token_id=job.token_id)
                if not transfer or not transfer.get("to"):
                    continue
                to_addr = transfer["to"]
                if to_addr in seen_recipients:
                    continue
                seen_recipients.add(to_addr)
                if to_addr in path_seen_addresses.get(job.path_id, set()):
                    continue
                amount = self._resolve_amount(sel_hash, transfer.get("amount", 0.0), job.chain, all_txs_map, job.asset)
                block_time = transfer.get("block_time")
                token_id = transfer.get("token_id") or job.token_id
                if transfer.get("output_owner"):
                    owner_hints[to_addr] = transfer.get("output_owner")
                if transfer.get("input_riskscore") is not None:
                    risk_map[job.current_address] = float(transfer.get("input_riskscore") or 0.0)
                if transfer.get("output_riskscore") is not None:
                    risk_map[to_addr] = float(transfer.get("output_riskscore") or 0.0)

                if idx == 0:
                    path_id = base_path_id
                else:
                    path_counter += 1
                    path_id = str(path_counter)
                    _copy_path(path_id, base_path_id)

                step_index = len(paths[path_id]["steps"])
                _add_step(path_id, {
                    "step_index": step_index,
                    "from": job.current_address,
                    "to": to_addr,
                    "tx_hash": sel_hash,
                    "chain": job.chain,
                    "asset": job.asset,
                    "amount_estimate": float(amount or 0.0),
                    "time": block_time,
                    "direction": "out",
                    "step_type": "direct_transfer",
                    "service_label": classification.get("service_label"),
                    "protocol": classification.get("protocol"),
                    "reasoning": (selector_result or {}).get("reasoning") or "Selected by hop selector.",
                })

                self._collect_token_transfer_data(
                    json.dumps(self._compact_tool_result("token_transfers", transfer_result), ensure_ascii=False),
                    {"tx_hash": sel_hash, "blockchain_name": job.chain},
                    all_txs_map,
                    risk_map,
                    txs_collected,
                    tx_list_collected,
                    txs_seen,
                )

                hop_queue.append(HopJob(
                    path_id=path_id,
                    current_address=to_addr,
                    incoming_tx_hash=sel_hash,
                    incoming_amount=float(amount or 0.0),
                    incoming_time=block_time,
                    chain=job.chain,
                    asset=job.asset,
                    token_id=int(token_id or 0),
                    hop_index=job.hop_index + 1,
                ))
                took_step = True

            if not took_step:
                paths[job.path_id]["stop_reason"] = "Loop detected - no new transactions"
                processed_paths += 1

        # Set termination reasons for any remaining paths
        for path in paths.values():
            if not path["stop_reason"]:
                path["stop_reason"] = "Trace completed"

        # De-duplicate identical or prefix paths
        def _sig(p):
            return tuple(step.get("tx_hash") or step.get("to") for step in p.get("steps", []))

        path_items = list(paths.items())
        signatures = {pid: _sig(pdata) for pid, pdata in path_items}
        remove_ids = set()

        for pid, sig in signatures.items():
            for oid, osig in signatures.items():
                if pid == oid:
                    continue
                if sig == osig:
                    # Keep the first one
                    if pid > oid:
                        remove_ids.add(pid)
                elif len(sig) < len(osig) and osig[:len(sig)] == sig:
                    if paths[pid]["stop_reason"] in ["Max hop limit reached", "Trace completed"]:
                        remove_ids.add(pid)

        for rid in remove_ids:
            paths.pop(rid, None)

        initial_amount = 0.0
        if paths:
            first_path = next(iter(paths.values()))
            if first_path["steps"]:
                initial_amount = float(first_path["steps"][0].get("amount_estimate") or 0.0)

        self.last_txs = txs_collected
        self.last_tx_list = tx_list_collected
        logger.info(f"TXS_ARRAY={json.dumps(self.last_txs, ensure_ascii=False)}")
        logger.info(f"TXLIST_ARRAY={json.dumps(self.last_tx_list, ensure_ascii=False)}")

        return {
            "case_meta": case_meta,
            "paths": list(paths.values()),
            "entities": list(entities.values()),
            "annotations": annotations,
            "trace_stats": {
                "initial_amount_estimate": initial_amount,
                "explored_paths": len(paths),
                "terminated_reason": "All paths reached terminal entities or dead ends",
            },
        }

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
                def _amount(tr):
                    amt = tr.get("amount") or tr.get("amount_coerced") or tr.get("value")
                    try:
                        return float(amt)
                    except Exception:
                        return 0.0

                transfer = max(transfers, key=_amount)
                input_data = transfer.get("input") or {}
                output_data = transfer.get("output") or {}
                from_addr = input_data.get("address") if isinstance(input_data, dict) else None
                to_addr = output_data.get("address") if isinstance(output_data, dict) else None

                tx_hash = arguments.get("tx_hash")
                chain = arguments.get("blockchain_name")
                tx_info = all_txs_map.get(tx_hash, {})
                token_id = tx_info.get("token_id") or transfer.get("token_id") or 0
                amount_raw = tx_info.get("amount")
                if amount_raw is None:
                    amount_raw = transfer.get("amount")
                if amount_raw is None:
                    amount_raw = transfer.get("amount_coerced")
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
                    if riskscore_from == 0.0 and isinstance(input_data, dict):
                        riskscore_from = float(input_data.get("riskscore") or 0.0)
                    if riskscore_to == 0.0 and isinstance(output_data, dict):
                        riskscore_to = float(output_data.get("riskscore") or 0.0)

                    amount_val = float(amount_raw) if amount_raw is not None else 0.0
                    if amount_val == 0.0 and transfer.get("amount_coerced") is not None and chain == "trx":
                        # Convert coerced amount back to base units for TRX UI helpers.
                        amount_val = float(transfer.get("amount_coerced") or 0.0) * 1e6

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

        payload = {
            "case_meta": case_meta.model_dump(),
            "token_id_hint": token_id_hint,
            "known_tx_hashes": config.known_tx_hashes,
            "inputs": {
                "victim_address": config.victim_address,
                "tx_hash": config.tx_hash,
                "blockchain_name": config.blockchain_name,
                "asset_symbol": (config.theft_asset or config.asset_symbol or "").upper(),
                "approx_date": config.approx_date,
                "description": config.description,
            },
            "rules_version": "orchestrator-unified-1",
        }

        if on_progress:
            await on_progress("Starting trace orchestrator...")

        llm_output = await self._run_agentic_trace(payload, on_progress=on_progress)

        try:
            trace_result = TraceResult.model_validate(llm_output)
        except Exception as exc:
            raise ValueError(f"TraceResult could not be parsed: {exc}") from exc

        trace_result.case_meta = trace_result.case_meta or case_meta
        if not trace_result.case_meta.trace_id:
            trace_result.case_meta.trace_id = trace_id

        trace_result = postprocess_trace_result(trace_result)
        await self._maybe_save_visualization(trace_result)

        return trace_result

    async def _maybe_save_visualization(self, trace_result: TraceResult) -> None:
        """Generate and save/share visualization, if possible."""
        try:
            tx_list = getattr(self, "last_tx_list", None)
            txs = getattr(self, "last_txs", None)
            viz_payload = generate_visualization_payload(trace_result, tx_list=tx_list, txs=txs)
        except Exception as exc:
            logger.warning(f"⚠️ Visualization payload generation failed: {exc}")
            return

        client = self._get_client()
        save_fn = getattr(client, "save_and_share_visualization", None)
        if not callable(save_fn):
            logger.warning("⚠️ Visualization save/share not supported by client.")
            return

        try:
            save_input = {
                "title": viz_payload.get("title"),
                "type": viz_payload.get("type", "address"),
                "payload": viz_payload.get("payload", {}),
                "helpers": viz_payload.get("helpers", {}),
                "extras": viz_payload.get("extras", {}),
            }
            result = await asyncio.wait_for(save_fn(save_input), timeout=30.0)
        except Exception as exc:
            logger.warning(f"⚠️ Visualization save/share failed: {exc}")
            return

        share_url = None
        if isinstance(result, dict):
            share_url = (
                result.get("share_url") or
                result.get("url") or
                result.get("data", {}).get("url") or
                result.get("data", {}).get("share_url")
            )
            share_obj = result.get("share_result") if isinstance(result.get("share_result"), dict) else {}
            if not share_url and share_obj:
                share_url = share_obj.get("url") or share_obj.get("share_url")

        if share_url:
            trace_result.visualization_url = share_url
            logger.info(f"✅ Visualization saved/shared: {share_url}")
        else:
            logger.info("✅ Visualization saved/shared (no URL returned).")

    @abstractmethod
    def _get_client(self):
        """Return the underlying client for theft_detection helpers."""
        pass
