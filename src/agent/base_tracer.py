"""
Base Tracer with shared orchestration logic.
Interface implementations: MCPTracer (local stdio), HTTPTracer (remote HTTP).
"""
import asyncio
import heapq
import itertools
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents import function_span
from httpx import Limits, Timeout
from openai import AsyncOpenAI

from .config import ModelConfig
from .currency_registry import get_registry as _get_currency_registry
from .llm_client import LLMResult, call_llm
from .model_registry import resolve_model
from .models import (
    CaseMeta,
    DecisionRef,
    TracerConfig,
    TraceResult,
)
from .prompt_loader import PromptSpec, load_prompt
from .recorder import MissingReplayEvent, TraceRecorder
from .theft_detection import (
    extract_victim_from_tx_hash,
    infer_approx_date_from_description,
    infer_asset_symbol,
)
from .tool_dispatch import BRIDGE_ANALYZER_MODEL
from .trace_postprocess import postprocess_trace_result
from .visualization import generate_visualization_payload

logger = logging.getLogger("tracer")
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[TRACE] %(message)s"))
    logger.addHandler(handler)
logger.propagate = False


# ─── Constants ────────────────────────────────────────────────────────────────
OPENAI_TIMEOUT = 90
OPENAI_CONNECT_TIMEOUT = 10
TOOL_TIMEOUT = 30
TOOL_TIMEOUT_SLOW = 60
# `get_extra_address_info` is upstream-bound and regularly blows past 20s for
# high-activity hub contracts (observed ReadTimeout at 30s in prod logs), so
# give it the slow budget. `all_txs` + `bridge_analyze` were already slow.
_SLOW_TOOLS = frozenset({"all_txs", "bridge_analyze", "get_extra_address_info"})
MAX_TOOL_CALLS_PER_TURN = 6
MAX_TOKEN_TRANSFERS_PER_TURN = 2
MAX_TX_LIST = 500


@dataclass
class HopJob:
    path_id: str
    current_address: str
    incoming_tx_hash: str | None
    incoming_amount: float
    incoming_time: int | None
    chain: str
    asset: str
    token_id: int
    hop_index: int
    attributed_amount: float = 0.0  # FIFO-attributed theft-origin share


class HopScheduler:
    """Priority queue + completion-budget scheduler for hop traversal.

    The tracer used to process ``HopJob`` items FIFO and cap the outer loop at
    ``processed_paths < max_paths``, where ``processed_paths`` was only
    incremented on terminal/dead-end branches. When a perpetrator split funds
    across ``>= max_paths`` siblings, dead-end siblings would exhaust the
    budget before legitimate continuations could reach their hop-3+ terminals
    (e.g. the real CEX deposit one hop further). That made the tracer stop on
    intermediate mules and the visualization layer then mislabelled those
    mules as "Exchange deposit address".

    The scheduler fixes both issues:

    * **Priority order**: larger ``attributed_amount`` is processed first,
      with ``hop_index`` asc and FIFO tiebreak. High-value branches always
      reach their terminal before the global cap closes on smaller siblings.
    * **Completion-based budget**: the outer loop terminates once
      ``completed_paths >= max_completed`` — continuations do not count
      against the budget. Combined with a hard ``max_iterations`` safety net
      to guard against pathological input.
    """

    def __init__(self, max_completed: int, max_iterations: int | None = None):
        if max_completed <= 0:
            raise ValueError("max_completed must be positive")
        self.max_completed = max_completed
        self.max_iterations = max_iterations if max_iterations and max_iterations > 0 else max_completed * 64
        self._heap: list[tuple[float, int, int, HopJob]] = []
        self._seq = itertools.count()
        self._iterations = 0

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def exhausted(self) -> bool:
        """Hard safety net: stop even when ``completed_paths`` hasn't reached the budget."""
        return self._iterations >= self.max_iterations

    def push(self, job: HopJob) -> None:
        # Priority key: larger attributed_amount first (negated), then shallower
        # hop first, then FIFO insertion order. We include seq so that HopJob
        # instances never need to be comparable with each other.
        priority = (-float(job.attributed_amount or 0.0), int(job.hop_index), next(self._seq))
        heapq.heappush(self._heap, (*priority, job))

    def pop(self) -> HopJob:
        """Pop the highest-priority job. Raises ``IndexError`` if empty."""
        *_, job = heapq.heappop(self._heap)
        self._iterations += 1
        return job

    def should_continue(self, completed_paths: int) -> bool:
        if not self._heap:
            return False
        if self.exhausted:
            return False
        return completed_paths < self.max_completed


class FIFOLedger:
    """
    Tracks per-address inflow queues and attributes outflows using FIFO.

    Design: attributed theft-share flows through the entire path without
    consuming the global cap. The cap is only consumed when funds reach a
    *terminal* endpoint (CEX, dead-end, mixer, etc.). This prevents the
    cap from being exhausted on intermediate hops.
    """

    def __init__(self, stolen_amount: float, tolerance: float = 0.03):
        self.stolen_amount = stolen_amount
        self.tolerance = tolerance
        self.cap = stolen_amount * (1.0 + tolerance) if stolen_amount > 0 else float("inf")
        self.total_traced: float = 0.0
        self._queues: dict[str, list[dict[str, float]]] = {}
        self._audit_log: list[dict[str, Any]] = []

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return self._audit_log

    def record_inflow(self, address: str, amount: float, theft_share: float):
        """Record an inflow to an address. theft_share is the attributed theft portion."""
        clamped_share = min(theft_share, amount)
        self._queues.setdefault(address, []).append({
            "amount": amount,
            "theft_share": clamped_share,
        })
        self._audit_log.append({
            "op": "inflow",
            "address": address,
            "amount": amount,
            "theft_share": clamped_share,
            "queue_len": len(self._queues[address]),
        })

    def attribute_outflow(self, address: str, outflow_amount: float) -> float:
        """
        Attribute an outflow from an address using FIFO.
        Returns the theft-attributed portion of this outflow.
        Does NOT consume the global cap — use claim_terminal for that.
        """
        queue = self._queues.get(address, [])
        if not queue:
            return 0.0

        remaining = outflow_amount
        attributed = 0.0
        entries_drained = 0

        while remaining > 0 and queue:
            entry = queue[0]
            take = min(remaining, entry["amount"])
            if entry["amount"] > 0:
                ratio = entry["theft_share"] / entry["amount"]
            else:
                ratio = 0.0
            theft_take = take * ratio

            attributed += theft_take
            entry["amount"] -= take
            entry["theft_share"] -= theft_take
            remaining -= take

            if entry["amount"] <= 0.001:
                queue.pop(0)
                entries_drained += 1

        self._audit_log.append({
            "op": "outflow",
            "address": address,
            "outflow_amount": outflow_amount,
            "attributed": attributed,
            "entries_drained": entries_drained,
            "queue_len": len(queue),
        })
        return attributed

    def claim_terminal(self, attributed: float) -> float:
        """
        Claim attributed amount at a terminal endpoint against the global cap.
        Call this ONLY when funds reach a final destination (CEX, dead-end, etc.).
        Returns clamped amount (may be less if cap would be exceeded).
        """
        if self.stolen_amount <= 0:
            self._audit_log.append({
                "op": "terminal",
                "attributed_input": attributed,
                "clamped_output": attributed,
                "total_traced_after": self.total_traced,
                "headroom_before": float("inf"),
            })
            return attributed
        headroom = max(0.0, self.cap - self.total_traced)
        clamped = min(attributed, headroom)
        self.total_traced += clamped
        self._audit_log.append({
            "op": "terminal",
            "attributed_input": attributed,
            "clamped_output": clamped,
            "total_traced_after": self.total_traced,
            "headroom_before": headroom,
        })
        return clamped

    @property
    def cap_exceeded(self) -> bool:
        return self.stolen_amount > 0 and self.total_traced >= self.cap


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
                    "model": {
                        "type": "string",
                        "description": "Bridge analyzer model id (e.g. bridge-analyzer-1)",
                    },
                    "chain": {"type": "string", "description": "Source chain"},
                    "tx_hash": {"type": "string", "description": "Transaction HASH"}
                },
                "required": ["model", "chain", "tx_hash"]
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

    def __init__(self, *, recorder: "TraceRecorder | None" = None):
        import httpx
        # Stash the httpx client for lazy reuse. AsyncOpenAI() calls
        # os.environ.get("OPENAI_API_KEY") at construction and raises
        # if unset — instantiating it in __init__ means every BaseTracer
        # subclass (including no-LLM unit tests) would need the key. The
        # lazy ``openai_client`` property below defers that check until
        # the first real LLM call.
        self._openai_http_client = httpx.AsyncClient(
            timeout=Timeout(OPENAI_TIMEOUT, connect=OPENAI_CONNECT_TIMEOUT),
            limits=Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0),
            http2=True,
        )
        self._openai_client: AsyncOpenAI | None = None

        self.model_orchestrator = ModelConfig.ORCHESTRATOR_MODEL
        self.model_selector = ModelConfig.SELECTOR_MODEL
        self.model_validator = ModelConfig.VALIDATOR_MODEL
        self.model_json_retry = ModelConfig.JSON_RETRY_MODEL

        self.validator_prompt_path = Path(__file__).parent / "prompts" / "trace_validator.md"
        self.selector_prompt_path = Path(__file__).parent / "prompts" / "trace_hop_selector.md"
        self.hop_classifier_prompt_path = Path(__file__).parent / "prompts" / "trace_hop_classifier.md"

        # Result storage for post-trace access
        self.last_txs: list[dict[str, Any]] = []
        self.last_tx_list: list[dict[str, Any]] = []
        self.last_address_info: dict[str, dict[str, Any]] = {}

        # Recorder: optional record/replay of every LLM + tool call.
        self.recorder: TraceRecorder | None = recorder

    @property
    def openai_client(self) -> AsyncOpenAI:
        """Lazily construct the AsyncOpenAI client.

        Deferring this until first access keeps ``BaseTracer()`` usable
        in unit tests that don't touch LLM paths and in CI environments
        without ``OPENAI_API_KEY``.
        """
        client = self._openai_client
        if client is None:
            client = AsyncOpenAI(http_client=self._openai_http_client, max_retries=1)
            self._openai_client = client
        return client

    # ─── Abstract methods (implemented by subclasses) ─────────────────────────

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call. Subclasses implement the transport (HTTP, MCP, etc)."""
        pass

    # ─── Prompt loading ───────────────────────────────────────────────────────

    def _load_prompt_spec(self, path: Path, default_name: str) -> "PromptSpec":
        """Load a prompt and cache the parsed ``PromptSpec`` on the instance.

        Caching avoids re-reading + re-parsing the file on every hop
        (each trace calls the classifier/selector dozens of times).
        """
        cache = self.__dict__.setdefault("_prompt_cache", {})
        cached = cache.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        spec = load_prompt(path, name_default=default_name)
        cache[path] = spec
        return spec

    def _validator_spec(self) -> "PromptSpec":
        return self._load_prompt_spec(self.validator_prompt_path, "validator")

    def _selector_spec(self) -> "PromptSpec":
        return self._load_prompt_spec(self.selector_prompt_path, "hop_selector")

    def _hop_classifier_spec(self) -> "PromptSpec":
        return self._load_prompt_spec(self.hop_classifier_prompt_path, "hop_classifier")

    # Back-compat body-only helpers (used by eval harness, log lines,
    # and any downstream code that still expects a string). Don't add
    # new callers — use ``_<x>_spec()`` instead.
    def _load_validator_prompt(self) -> str:
        return self._validator_spec().body

    def _load_selector_prompt(self) -> str:
        return self._selector_spec().body

    def _load_hop_classifier_prompt(self) -> str:
        return self._hop_classifier_spec().body

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

    def _flatten_strings(self, value: Any, limit: int = 200) -> list[str]:
        items: list[str] = []

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
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _parse_transfer(
        self,
        result: Any,
        expected_from: str | None = None,
        token_id: int | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        transfers = result.get("data", [])
        if not isinstance(transfers, list) or not transfers:
            return None

        def _amount(tr):
            amt = tr.get("amount") or tr.get("amount_coerced") or tr.get("value")
            try:
                return float(amt)
            except (TypeError, ValueError):
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
        except (TypeError, ValueError):
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

    def _parse_date_to_ts(self, date_str: str | None) -> int | None:
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            return int(dt.timestamp())
        except ValueError:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return int(dt.timestamp())
            except ValueError:
                logger.warning("Could not parse date string: %s", date_str)
                return None

    _SATOSHI_CHAINS = {"btc", "bch", "ltc"}

    # Account-model chains whose native asset never shows up in token_transfers.
    # For these (chain, asset) pairs we jump straight to get_transaction and skip
    # the guaranteed-empty token_transfers round-trip.
    _NATIVE_ASSET_BY_CHAIN = {
        "eth": {"ETH"},
        "bnb": {"BNB"},
        "bsc": {"BNB"},
        "polygon": {"MATIC", "POL"},
        "matic": {"MATIC", "POL"},
        "arbitrum": {"ETH"},
        "arb": {"ETH"},
        "optimism": {"ETH"},
        "op": {"ETH"},
        "base": {"ETH"},
        "trx": {"TRX"},
        "tron": {"TRX"},
        "avax": {"AVAX"},
        "avalanche": {"AVAX"},
        "sol": {"SOL"},
        "solana": {"SOL"},
    }

    def _is_native_asset(self, chain: str | None, asset: str | None) -> bool:
        if not chain or not asset:
            return False
        natives = self._NATIVE_ASSET_BY_CHAIN.get(chain.lower())
        if not natives:
            return False
        return asset.upper() in natives

    def _should_skip_token_transfers(
        self,
        chain: str | None,
        asset_hint: str | None,
        address_hint: str | None,
        tx_token_id: int | None,
    ) -> bool:
        """Decide whether ``_resolve_transfer`` may jump straight to
        ``get_transaction`` and skip ``token_transfers``.

        Skipping is only safe when BOTH hold:
          1. The trace is walking a chain's *native* asset (ETH on eth,
             TRX on trx, …) — token_transfers is guaranteed empty and
             would just add a round-trip.
          2. The *specific tx* we're resolving is itself a native transfer
             (token_id == 0 / None). If the classifier picked a token tx
             (e.g. USDT on eth with token_id=94252), ``get_transaction``
             will return the token-contract address as ``to`` and the
             trace will dead-end at the contract instead of the real
             recipient — we must take the token_transfers route in that
             case.
        """
        if address_hint is None:
            return False
        if not self._is_native_asset(chain, asset_hint):
            return False
        if tx_token_id is not None and int(tx_token_id) != 0:
            return False
        return True

    _EVM_NATIVE_CHAINS = frozenset({"eth", "bnb", "bsc", "matic", "arb", "op", "avax", "base"})

    def _normalize_amount(self, amount: Any, chain: str, asset: str | None = None) -> float:
        """Convert a base-unit amount to display units, using currencies.json
        when available and falling back to hand-written rules for anything
        the registry doesn't know about.

        Safeguard: we only scale when ``val`` is large enough to plausibly
        be a base-unit value (``>= 10 ** (unit // 2)``). This prevents a
        second scaling pass on amounts that are already in display form
        (e.g. ``amount_coerced`` from ``all_txs`` comes through as 0.14 ETH
        and must NOT be divided again).
        """
        try:
            val = float(amount)
        except (TypeError, ValueError):
            return 0.0
        if val == 0.0:
            return 0.0

        chain_norm = (chain or "").strip().lower() or None
        asset_upper = asset.upper().strip() if isinstance(asset, str) and asset.strip() else None

        # 1. Registry-driven path: authoritative decimals from
        #    currencies.json. When asset is provided we resolve by
        #    (chain, symbol); otherwise we fall back to the chain's
        #    native token.
        if chain_norm:
            registry = _get_currency_registry()
            record = None
            if asset_upper:
                record = registry.lookup_by_symbol(chain_norm, asset_upper)
            if record is None:
                record = registry.lookup(chain_norm, 0)
            if record is not None and record.unit > 0:
                scale = 10 ** record.unit
                # Display-vs-base heuristic. The old hardcoded path used a
                # flat ``1e6`` cutoff for all assets (USDT/TRX/BTC native
                # all "divide when >= 1e6"). Registry migration shortened
                # the safeguard to ``10^(unit/2)`` which broke USDT
                # (unit=6): a display 60000-USDT value tripped the 10^3
                # threshold and got divided a second time to 0.06. We
                # restore the 1e6 floor for small-unit tokens while still
                # using ``10^(unit/2)`` for high-unit ones (unit=18 →
                # 10^9, matching the old EVM rule for gwei→ETH).
                safeguard = 10 ** max(record.unit // 2, 6)
                if val >= safeguard:
                    return val / scale
                return val

        # 2. Legacy hardcoded rules — kept as fallback for chains/tokens
        #    not yet in the registry (or in case it can't be loaded).
        six_dec_assets = {"USDT", "USDC", "TUSD", "USDP", "USDD", "BUSD"}
        if asset_upper in six_dec_assets and val >= 1e6:
            return val / 1e6
        if chain_norm == "trx" and val >= 1e6:
            scaled = val / 1e6
            if scaled >= 1e9:
                return scaled / 1e6
            return scaled
        if chain_norm in self._SATOSHI_CHAINS and val >= 1e6:
            return val / 1e8
        if chain_norm in self._EVM_NATIVE_CHAINS and val >= 1e6:
            return val / 1e9
        return val

    def _resolve_amount(
        self,
        tx_hash: str | None,
        amount: Any,
        chain: str,
        all_txs_map: dict[str, dict[str, Any]],
        asset: str | None = None,
    ) -> float:
        """Resolve a step's display amount, preferring ``amount_coerced``
        from ``all_txs`` (already in display units) over the raw on-chain
        ``amount``.

        Why ``amount_coerced`` is treated as display and NOT pushed
        through ``_normalize_amount``: the API contract is that
        ``amount_coerced`` is the human-readable per-token amount (e.g.
        ``0.14`` ETH, ``1080200`` USDT). ``_normalize_amount`` has a
        safeguard ``10 ** max(unit // 2, 6)`` to detect "is this raw or
        display?", but the threshold is exactly ``10**6`` for low-unit
        tokens (USDT/USDC unit=6). A real-world 1.08M USDT step has
        display value ``1.08e6`` which equals the safeguard, gets
        wrongly classified as "raw", and is divided again to ``1.08``.
        Downstream dust filters then trim the hop as 0.0001% of the
        stolen amount and the trace stops one hop early.

        For ``amount`` (raw on-chain), we still call
        ``_normalize_amount`` because that path comes from
        ``token_transfers`` and is genuinely in base units.
        """
        if tx_hash and tx_hash in all_txs_map:
            row = all_txs_map[tx_hash]
            amt_coerced = row.get("amount_coerced")
            if amt_coerced is not None:
                try:
                    return float(amt_coerced)
                except (TypeError, ValueError):
                    pass
            amt_raw = row.get("amount")
            if amt_raw is not None:
                return self._normalize_amount(amt_raw, chain, asset)
        return self._normalize_amount(amount, chain, asset)

    @staticmethod
    def _extract_identity_texts(owner: Any, services: Any, owner_hint: Any = None) -> str:
        """Extract only identity-relevant fields for keyword matching.

        Targets: owner.name, owner.slug, owner.subtype from get_address;
        services.use_platform[] from get_extra_address_info;
        owner_hint.name, owner_hint.slug from token_transfers.
        """
        parts: list[str] = []

        for obj in (owner, owner_hint):
            if isinstance(obj, dict):
                for key in ("name", "slug", "subtype", "title"):
                    val = obj.get(key)
                    if isinstance(val, str) and val:
                        parts.append(val)

        if isinstance(services, dict):
            use_platform = services.get("use_platform")
            if isinstance(use_platform, list):
                parts.extend(str(x) for x in use_platform if x)

        return " ".join(parts).lower()

    # Bridge brands seen under ``owner.type="other"``. The API labels many
    # cross-chain services this way without a proper ``bridge`` type, so
    # we match by name to still classify them as bridges — otherwise the
    # tracer stops at the bridge contract instead of continuing on the
    # destination chain via ``bridge_analyze``.
    _BRIDGE_BRAND_NAMES: frozenset[str] = frozenset({
        "allbridge", "bridgers", "bridgers.xyz", "layerzero", "stargate",
        "wormhole", "synapse", "hop", "multichain", "across", "router",
        "symbiosis", "mayan", "cbridge", "celer", "debridge", "squid",
        "connext", "orbiter", "thorchain", "rango", "rubic",
        # NEAR Intents / NEAR Omni Bridge / NEAR One — same protocol under
        # multiple display names. The internal aggregator address comes
        # tagged as ``"NEAR Intents Treasury"`` (type=other, no subtype),
        # so the brand-name fallback is the only path that catches it.
        "near intent", "near intents", "near-intent", "near-intents",
        "near omni", "near-omni", "near one", "near-one",
    })

    @classmethod
    def _classify_by_owner_type(cls, owner: Any) -> dict[str, Any] | None:
        """Structural classification from ``get_address.data.owner.type``.

        The AML API returns a typed ``owner`` block for every identified
        service: ``{name, slug, type, subtype}``. The ``type`` field is
        the primary structural signal (``exchange_unlicensed``,
        ``p2p_exchange_unlicensed``, ``bridge``, ``mixer``, ``other``, …),
        but the API is inconsistent for bridges and DEXes: brands like
        Bridgers arrive as ``type="other", subtype="Bridge"``, and
        SunSwap as ``type="p2p_exchange_unlicensed", subtype="DEX"``. So
        we inspect ``subtype`` (and, for ``type="other"``, a curated
        bridge-brand allowlist) before falling through to the generic
        "identified service" bucket.

        Returns ``None`` when the owner block is empty/nameless or the
        type is unrecognized — the caller then falls back to keyword
        heuristics.
        """
        if not isinstance(owner, dict):
            return None
        name = owner.get("name")
        slug = owner.get("slug")
        if (not name or not str(name).strip()) and (not slug or not str(slug).strip()):
            # No identity → nothing to anchor on, even if ``type`` is set.
            return None
        identity = str(name or slug)
        if identity.lower() in {"unknown", "null", ""}:
            return None

        owner_type = (owner.get("type") or "").strip().lower()
        subtype = (owner.get("subtype") or "").strip().lower()

        if owner_type.startswith("exchange"):
            # exchange, exchange_licensed, exchange_unlicensed
            return {
                "role": "cex_deposit", "terminal": True,
                "service_label": identity, "protocol": None,
            }
        if owner_type.startswith("p2p_exchange"):
            # DEX subtype is explicit in the API (e.g. SunSwap → subtype=DEX).
            if subtype == "dex":
                return {
                    "role": "dex_service", "terminal": True,
                    "service_label": identity, "protocol": None,
                }
            # Other p2p variants = OTC-ish. Product intent is to stop here
            # — Vasco asked for "don't step through identified services".
            return {
                "role": "otc_service", "terminal": True,
                "service_label": identity, "protocol": None,
            }
        if owner_type.startswith("bridge"):
            return {
                "role": "bridge_service", "terminal": True,
                "service_label": identity, "protocol": None,
            }
        if owner_type == "mixer":
            return {
                "role": "unidentified_service", "terminal": True,
                "service_label": "Mixer", "protocol": None,
            }
        if owner_type == "stolen_coins":
            # Community-reported victim address (e.g. "Victim report #16547").
            # The tag means the funds are known-stolen — NOT that this is a
            # destination we stop at. Our own trace's victim is a separate
            # concept (anchored at case_meta.victim_address). Flag it so the
            # UI shows "Stolen funds" but keep tracing outflows.
            return {
                "role": "intermediate", "terminal": False,
                "service_label": "Stolen funds", "protocol": None,
            }
        if owner_type == "other":
            # Subtype from the API is the most specific signal we have
            # when ``type`` is the catch-all "other". A Bridgers-style
            # bridge comes in as {type: other, subtype: Bridge}.
            if subtype == "bridge":
                return {
                    "role": "bridge_service", "terminal": True,
                    "service_label": identity, "protocol": identity.lower(),
                }
            if subtype == "dex":
                return {
                    "role": "dex_service", "terminal": True,
                    "service_label": identity, "protocol": identity.lower(),
                }
            # Brand-name fallback: map well-known cross-chain bridges by
            # name even when the API returns a bare ``type="other"``
            # without a subtype. Keeping this list narrow — it only
            # triggers for owner-confirmed identities.
            ident_lower = identity.lower()
            slug_lower = str(slug or "").lower()
            for brand in cls._BRIDGE_BRAND_NAMES:
                if brand in ident_lower or brand in slug_lower:
                    return {
                        "role": "bridge_service", "terminal": True,
                        "service_label": identity, "protocol": brand,
                    }
            # Miner / mining pool — terminal, we don't chase block
            # rewards down the miner payout tree.
            if subtype in {"miner", "mining_pool", "pool"}:
                return {
                    "role": "unidentified_service", "terminal": True,
                    "service_label": identity, "protocol": None,
                }
            # Known identity without a tighter category (e.g. Tronify).
            # We stop rather than chase — but use a generic label so the
            # UI still shows the brand name.
            return {
                "role": "unidentified_service", "terminal": True,
                "service_label": identity, "protocol": None,
            }
        return None

    @classmethod
    def _owner_matches_bridge_brand(cls, owner: Any) -> str | None:
        """Return the matched brand keyword if ``owner`` (an API owner
        block — ``{"name": …, "slug": …}``) lines up with one of the
        ``_BRIDGE_BRAND_NAMES`` entries, otherwise ``None``.

        Used both by the structural classifier (tagging an address that
        IS the bridge contract) and by the deposit-detection heuristic
        (tagging an address that SENDS to a bridge).
        """
        if not isinstance(owner, dict):
            return None
        name = str(owner.get("name") or "").lower()
        slug = str(owner.get("slug") or "").lower()
        if not name and not slug:
            return None
        for brand in cls._BRIDGE_BRAND_NAMES:
            if brand in name or brand in slug:
                return brand
        return None

    def _detect_bridge_deposit_pattern(
        self,
        data_list: list[dict[str, Any]],
        threshold: float = 0.7,
    ) -> dict[str, Any] | None:
        """Detect the "per-swap deposit address forwards to a bridge
        treasury" pattern.

        Some bridges (NEAR Intents, NEAR Omni, …) issue a fresh deposit
        address per cross-chain order; the deposit then forwards funds
        to a stable internal aggregator that DOES carry an owner tag
        (e.g. ``"NEAR Intents Treasury"``). The deposit itself has no
        tag at all, so the structural classifier sees a plain
        intermediate address and the trace would naively follow the
        deposit→treasury hop and dead-end inside the bridge's internal
        plumbing instead of crossing chains.

        Heuristic: when the dominant share of an address's outflow
        volume terminates at a known bridge brand (``threshold``
        defaults to 70%), the address itself is acting as that bridge's
        deposit-side. The cross-chain bridge tx of interest is then the
        FUNDING tx of this deposit (i.e. the tx that put us on this
        address), not the deposit→treasury tx that we just observed.

        Returns the bridge-brand owner dict if the pattern matches,
        otherwise ``None``. The caller is expected to (a) reclassify
        the current address as ``bridge_service``, (b) re-run
        ``bridge_analyze`` on its ``incoming_tx_hash`` to surface a
        cross-chain destination, and (c) suppress the ``HopJob`` that
        would have been pushed for the internal aggregator.
        """
        if not data_list:
            return None
        total_volume = 0.0
        by_brand: dict[str, dict[str, Any]] = {}
        for tx in data_list[:50]:
            amt = tx.get("amount_coerced")
            if amt is None:
                amt = tx.get("amount")
            try:
                volume = float(amt or 0.0)
            except (TypeError, ValueError):
                continue
            if volume <= 0:
                continue
            total_volume += volume
            owner: Any = None
            cps = tx.get("counterparty")
            if isinstance(cps, list) and cps and isinstance(cps[0], dict):
                owner = cps[0]
            if owner is None:
                owner = tx.get("output_owner") or tx.get("owner")
            brand = self._owner_matches_bridge_brand(owner)
            if not brand:
                continue
            entry = by_brand.setdefault(brand, {"vol": 0.0, "owner": owner})
            entry["vol"] += volume
            # Prefer the owner block carrying the most metadata.
            if isinstance(owner, dict) and len(owner) > len(entry["owner"] or {}):
                entry["owner"] = owner
        if total_volume <= 0:
            return None
        for entry in by_brand.values():
            if entry["vol"] / total_volume >= threshold:
                return entry["owner"]
        return None

    def _heuristic_classify(self, owner: Any, services: Any, owner_hint: Any = None) -> dict[str, Any]:
        # Structural signal first: owner.type matches a known service
        # family with a non-null name/slug.
        typed = self._classify_by_owner_type(owner)
        if typed:
            return typed

        # Strong signal: owner field from get_address or owner_hint from
        # token_transfers -- these identify who OWNS the address.
        owner_text = self._extract_identity_texts(owner, None, owner_hint)
        # Combined with the weaker use_platform signal for context.
        combined = self._extract_identity_texts(owner, services, owner_hint)

        def _has_any(text: str, keywords: list[str]) -> bool:
            return any(k in text for k in keywords)

        mixer_keywords = ["mixer", "tornado", "blender", "sinbad"]
        otc_keywords = ["otc"]
        cex_keywords = [
            "exchange", "binance", "coinbase", "kraken", "okx", "huobi", "kucoin",
            "bybit", "gate", "bitfinex", "mxc", "gate.io", "poloniex",
        ]
        bridge_keywords = [
            "bridge", "layerzero", "stargate", "wormhole", "allbridge",
            "synapse", "hop", "multichain", "bridger", "bridgers", "bridgers.xyz",
            "router", "across",
        ]
        dex_keywords = ["dex", "swap", "uniswap", "sushiswap", "pancakeswap", "curve"]

        if _has_any(combined, mixer_keywords):
            return {"role": "unidentified_service", "terminal": True, "service_label": "Mixer", "protocol": None}
        if _has_any(combined, otc_keywords):
            return {"role": "otc_service", "terminal": False, "service_label": "OTC", "protocol": None}
        if _has_any(combined, cex_keywords):
            # Only treat as terminal if the owner field itself matches (strong
            # signal).  services.use_platform alone just means the address
            # *interacted* with the exchange -- not that it IS the exchange.
            is_strong = _has_any(owner_text, cex_keywords)
            return {"role": "cex_deposit", "terminal": is_strong, "service_label": "Exchange", "protocol": None}
        if _has_any(combined, bridge_keywords):
            is_strong = _has_any(owner_text, bridge_keywords)
            return {"role": "bridge_service", "terminal": is_strong, "service_label": "Bridge", "protocol": None}
        if _has_any(combined, dex_keywords):
            is_strong = _has_any(owner_text, dex_keywords)
            return {"role": "dex_service", "terminal": is_strong, "service_label": "DEX", "protocol": None}

        return {"role": "intermediate", "terminal": False, "service_label": None, "protocol": None}

    @staticmethod
    def _classify_otc_like(
        total_in_volume: float,
        tx_count: int,
        counterparty_count: int,
        address_age_days: int,
        outbound_distribution: dict[str, float],
    ) -> dict[str, Any]:
        """
        Behavioral classification: detect OTC-like / Potential Service entities
        based on activity metrics. Returns classification dict (never terminal).

        Criteria (all must hold):
          - total_in_volume >= $50k
          - tx_count >= 100
          - address_age_days >= 180 (6 months)
        """
        is_otc_like = (
            total_in_volume >= 50_000
            and tx_count >= 100
            and address_age_days >= 180
        )
        if not is_otc_like:
            return {"otc_like": False}

        dominant_cex = None
        dominant_share = 0.0
        total_out = sum(outbound_distribution.values()) or 1.0
        for dest, amount in outbound_distribution.items():
            share = amount / total_out
            if share > dominant_share:
                dominant_share = share
                dominant_cex = dest

        return {
            "otc_like": True,
            "total_in_volume": total_in_volume,
            "tx_count": tx_count,
            "counterparty_count": counterparty_count,
            "address_age_days": address_age_days,
            "dominant_cex": dominant_cex,
            "dominant_cex_share": dominant_share,
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

    def _tool_call_to_dict(self, tool_call: Any) -> dict[str, Any]:
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

    def _extract_tool_calls(self, choice: Any, message: Any, finish_reason: str) -> list[Any]:
        tool_calls: list[Any] = []
        choice_dump: dict[str, Any] | None = None

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

    def _summarize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def _trim_messages(self, messages: list[dict[str, Any]], max_messages: int = 12) -> list[dict[str, Any]]:
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

    # Any single outflow ≥ MIXED_FUNDS_RATIO × incoming is a strong signal
    # the address holds pre-existing, non-theft balance that's exiting in
    # the same chronological window. Greedy accumulation would stop after
    # that one tx and miss the staged theft outflows that follow.
    _MIXED_FUNDS_RATIO: float = 1.2

    def _accumulate_hashes(
        self,
        txs: list[dict[str, Any]],
        incoming_amount: float | None,
        chain: str,
        asset: str | None = None,
        max_select: int = 25,
        *,
        hop_index: int | None = None,
    ) -> list[str]:
        """Chronological accumulation per trace_orchestrator.md.

        Selects outgoing transactions in chronological order until their
        combined amount covers ``incoming_amount`` (or the max-select cap is
        reached). We deliberately keep accumulating until we cross the
        incoming total — any earlier "close enough" shortcut (e.g. the old
        1.5% gap heuristic) silently drops the next staged payout when the
        perpetrator splits funds into one big cluster plus a smaller tail
        (e.g. CEX deposits + a side transfer to a secondary mule address),
        which then cascades into missing downstream hops.

        Mixed-funds safeguard — **hop-1 only**: when any single outflow
        alone is ``≥ _MIXED_FUNDS_RATIO × incoming`` AND we're still at
        the first hop, the "first to cross incoming" break is suppressed
        for the rest of the loop. The address almost certainly held
        pre-existing balance mixed with the theft inflow (e.g. a
        long-lived mule that already had ~100k USDT when 105k of stolen
        funds arrived, then wired 220k out in one shot before staging
        the actual theft share across many smaller sends). We widen
        selection to ``max_select`` and let ``max_paths`` downstream cap
        the fan-out.

        Hop-2+ is gated because by then ``incoming_amount`` is already
        the FIFO-narrowed theft share, and a single covering outflow is
        almost certainly THE continuation — not a mixed-balance payout.
        Without the gate we saw a 60.60k USDT inflow followed by 300k /
        121k / 58k siblings at hop 2, cluttering the graph with outflows
        that clearly carry the recipient's own balance (see
        ``recordings/2026-04-24/…_trx_USDT_e37253294b__trace_c2.jsonl``).
        ``hop_index=None`` keeps the legacy "activate unconditionally"
        behavior for callers that don't know their hop depth (eg. seed
        expansion from a victim address before hop 1 formally starts).
        """
        if not txs:
            return []
        if not incoming_amount or incoming_amount <= 0:
            first_hash = txs[0].get("hash") or txs[0].get("tx_hash")
            return [first_hash] if first_hash else []

        accumulated = 0.0
        incoming = float(incoming_amount)
        mixed_funds_threshold = incoming * self._MIXED_FUNDS_RATIO
        mixed_funds_enabled = hop_index is None or hop_index <= 1
        mixed_funds_detected = False
        # On hop_index >= 2, skip individual outflows that alone exceed
        # ``MIXED_FUNDS_RATIO × incoming`` — they almost certainly carry
        # the recipient's own balance rather than propagated theft. We
        # still need to pick SOMETHING though, so the first filtered tx
        # acts as a fallback when every candidate is oversized.
        oversized_skip_enabled = not mixed_funds_enabled
        selected: list[str] = []
        skipped_oversized: list[str] = []

        for item in txs:
            tx_hash = item.get("hash") or item.get("tx_hash")
            if not tx_hash:
                continue
            amount_val = item.get("amount_coerced")
            if amount_val is None:
                amount_val = item.get("amount")
            amount_norm = self._normalize_amount(amount_val or 0.0, chain, asset)

            if (
                oversized_skip_enabled
                and amount_norm >= mixed_funds_threshold
            ):
                # Remember the first oversized tx as a safety net for the
                # "every candidate is oversized" corner case, but don't
                # let it contaminate the normal accumulation.
                if not skipped_oversized:
                    skipped_oversized.append(tx_hash)
                continue

            accumulated += amount_norm
            selected.append(tx_hash)

            if mixed_funds_enabled and amount_norm >= mixed_funds_threshold:
                mixed_funds_detected = True

            if not mixed_funds_detected and accumulated >= incoming:
                break
            if len(selected) >= max_select:
                break

        # Safety net: if we filtered everything, surface the earliest
        # oversized tx so the trace doesn't silently dead-end. Better a
        # noisy branch than a missing one.
        if not selected and skipped_oversized:
            selected.append(skipped_oversized[0])

        return selected

    # ─── Selector ─────────────────────────────────────────────────────────────

    @staticmethod
    def _llm_result_to_decision_ref(result: LLMResult, output_summary: dict) -> "DecisionRef":
        return DecisionRef(
            prompt_name=result.prompt_name,
            prompt_version=result.prompt_version,
            model=result.model,
            family=result.family,
            reasoning_effort=result.reasoning_effort,
            input_hash=result.input_hash,
            output_summary=output_summary,
            usage=result.usage,
            latency_ms=result.latency_ms,
            decision_id=result.decision_id,
            from_replay=result.from_replay,
        )

    async def _run_selector(
        self, context: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, "DecisionRef | None"]:
        summary = self._summarize_payload(context)
        logger.info("[PROMPT=trace_hop_selector] txs=%d", summary["txs"])
        spec = self._selector_spec()
        try:
            result = await call_llm(
                openai_client=self.openai_client,
                model_spec=resolve_model(
                    spec.model_default or self.model_selector,
                    reasoning_effort=spec.reasoning_effort if resolve_model(spec.model_default or self.model_selector).is_reasoning and spec.reasoning_effort else None,
                ),
                prompt_name=spec.name,
                prompt_version=spec.version,
                system=spec.body,
                user=context,
                recorder=self.recorder,
                response_format="json",
                max_output_tokens=spec.max_output_tokens,
            )
            parsed = result.parsed or {}
            selected = parsed.get("selected_hashes") if isinstance(parsed, dict) else None
            output_summary = {
                "selected_count": len(selected) if isinstance(selected, list) else 0,
                # Truncate hashes in the summary — full list stays in the
                # recording; this is what lands in the TraceResult.
                "selected_hashes_preview": [h[:12] + "..." for h in (selected or [])[:5]]
                if isinstance(selected, list) else [],
            }
            return parsed, self._llm_result_to_decision_ref(result, output_summary)
        except Exception as exc:
            logger.warning("Selector failed: %s", exc)
            return None, None

    async def _run_hop_classifier(
        self, context: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, "DecisionRef | None"]:
        logger.info("[PROMPT=hop_classifier] address=%s", context.get("address"))
        spec = self._hop_classifier_spec()
        try:
            result = await call_llm(
                openai_client=self.openai_client,
                model_spec=resolve_model(
                    spec.model_default or self.model_selector,
                    reasoning_effort=spec.reasoning_effort if resolve_model(spec.model_default or self.model_selector).is_reasoning and spec.reasoning_effort else None,
                ),
                prompt_name=spec.name,
                prompt_version=spec.version,
                system=spec.body,
                user=context,
                recorder=self.recorder,
                response_format="json",
                max_output_tokens=spec.max_output_tokens,
            )
            parsed = result.parsed or {}
            output_summary = {
                "role": parsed.get("role") if isinstance(parsed, dict) else None,
                "terminal": parsed.get("terminal") if isinstance(parsed, dict) else None,
                "stop_reason": parsed.get("stop_reason") if isinstance(parsed, dict) else None,
            }
            return parsed, self._llm_result_to_decision_ref(result, output_summary)
        except Exception as exc:
            logger.warning("Hop classifier failed: %s", exc)
            return None, None

    # ─── Validator ────────────────────────────────────────────────────────────

    async def _run_validator(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], "DecisionRef"]:
        summary = self._summarize_payload(payload)
        logger.info("[PROMPT=trace_validator] entities=%d paths=%d txs=%d",
                    summary["entities"], summary["paths"], summary["txs"])
        spec = self._validator_spec()
        result = await call_llm(
            openai_client=self.openai_client,
            model_spec=resolve_model(
                spec.model_default or self.model_validator,
                reasoning_effort=spec.reasoning_effort if resolve_model(spec.model_default or self.model_validator).is_reasoning and spec.reasoning_effort else None,
            ),
            prompt_name=spec.name,
            prompt_version=spec.version,
            system=spec.body,
            user=payload,
            recorder=self.recorder,
            response_format="json",
            max_output_tokens=spec.max_output_tokens,
            timeout=60.0,
        )
        if result.parsed is None:
            # Validator is a hard dependency of postprocessing; raise on
            # empty/unparseable output rather than returning {} and letting
            # downstream code produce a garbage TraceResult.
            raise ValueError("validator returned non-JSON content")
        output_summary = {
            "paths": len(result.parsed.get("paths", [])) if isinstance(result.parsed, dict) else 0,
            "entities": len(result.parsed.get("entities", [])) if isinstance(result.parsed, dict) else 0,
        }
        return result.parsed, self._llm_result_to_decision_ref(result, output_summary)

    async def _run_agentic_trace(
        self, payload: dict[str, Any],
        on_progress: Callable[[str], Awaitable[None]] | None = None
    ) -> dict[str, Any]:
        """Agentic split prompts: selector + hop classifier, tool execution in code."""
        case_meta = payload.get("case_meta", {})
        inputs = payload.get("inputs", {})
        chain = self._normalize_chain(inputs.get("blockchain_name") or case_meta.get("blockchain_name") or "eth")
        asset = (inputs.get("asset_symbol") or case_meta.get("asset_symbol") or "").upper()
        tx_hash = inputs.get("tx_hash")
        victim_address = inputs.get("victim_address")
        approx_date = inputs.get("approx_date")
        token_id_hint = payload.get("token_id_hint") or 0

        max_hops = 12
        max_paths = 10

        entities: dict[str, dict[str, Any]] = {}
        annotations: list[dict[str, Any]] = []
        paths: dict[str, dict[str, Any]] = {}
        path_seen_addresses: dict[str, set] = {}
        path_seen_hashes: dict[str, set] = {}
        path_counter = 1

        all_txs_map: dict[str, dict[str, Any]] = {}
        risk_map: dict[str, float] = {}
        owner_hints: dict[str, Any] = {}
        txs_collected: list[dict[str, Any]] = []
        tx_list_collected: list[dict[str, Any]] = []
        txs_seen: set = set()

        stolen_amount = float(inputs.get("stolen_amount") or payload.get("stolen_amount") or 0.0)
        traced_tolerance = float(inputs.get("traced_amount_tolerance") or payload.get("traced_amount_tolerance") or 0.03)
        fifo_ledger = FIFOLedger(stolen_amount, traced_tolerance)
        # Dust threshold — skip pushing a HopJob when the FIFO-attributed
        # theft share of a new step falls below this fraction of the
        # original stolen_amount. Set to 0 to disable (legacy behavior).
        min_attribution_ratio = float(
            inputs.get("min_path_attribution_ratio")
            or payload.get("min_path_attribution_ratio")
            or 0.01
        )
        dust_trimmed_paths: set[str] = set()

        async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
            if "blockchain_name" in arguments:
                arguments["blockchain_name"] = self._normalize_chain(arguments.get("blockchain_name"))
            if "chain" in arguments:
                arguments["chain"] = self._normalize_chain(arguments.get("chain"))
            if "asset" in arguments:
                arguments["asset"] = self._normalize_chain(arguments.get("asset"))
            tool_input = json.dumps(arguments, ensure_ascii=False)
            tool_input = tool_input[:2000] if len(tool_input) > 2000 else tool_input
            timeout = TOOL_TIMEOUT_SLOW if tool_name in _SLOW_TOOLS else TOOL_TIMEOUT
            with function_span(tool_name, input=tool_input) as tool_span:
                # Replay short-circuit: when the recorder is in replay mode,
                # return the recorded result instead of hitting the backend.
                # Missing events fall through to a live call (useful when
                # running a partial replay against a new code path).
                if self.recorder is not None and self.recorder.is_replay:
                    try:
                        replayed = self.recorder.replay_tool_call(tool_name, arguments)
                        try:
                            if hasattr(tool_span, "span_data"):
                                tool_span.span_data.output = self._compact_tool_result(tool_name, replayed)
                        except Exception:
                            pass
                        return replayed
                    except MissingReplayEvent as miss:
                        logger.warning(
                            "replay miss for tool %s: %s — falling back to live call",
                            tool_name, miss,
                        )

                started_at = time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        self.execute_tool(tool_name, arguments),
                        timeout=timeout
                    )
                    try:
                        if hasattr(tool_span, "span_data"):
                            tool_span.span_data.output = self._compact_tool_result(tool_name, result)
                    except Exception:
                        pass
                    if self.recorder is not None and self.recorder.is_recording:
                        self.recorder.record_tool_call(
                            tool_name, arguments, result,
                            duration_ms=int((time.perf_counter() - started_at) * 1000),
                        )
                    return result
                except TimeoutError:
                    logger.error(
                        "❌ Tool timeout: %s (limit=%ss) | args=%s",
                        tool_name, timeout, tool_input,
                    )
                    try:
                        tool_span.set_error({"message": "tool_timeout", "data": {"tool": tool_name}})
                        tool_span.span_data.output = {"error": "tool_timeout", "tool": tool_name}
                    except Exception:
                        pass
                    if self.recorder is not None and self.recorder.is_recording:
                        self.recorder.record_tool_call(
                            tool_name, arguments, None,
                            duration_ms=int((time.perf_counter() - started_at) * 1000),
                            error="tool_timeout",
                        )
                    return {"error": "tool_timeout", "tool": tool_name}
                except Exception as e:
                    # Surface the FULL context for diagnosing 5xx from
                    # upstream MCP tools (bridge-analyze, etc.) — tool
                    # name, the exact arguments we sent, the exception
                    # type, and any response body attached by
                    # ``mcp_http_client`` via ``exc.response_body``.
                    resp_body = getattr(e, "response_body", None)
                    resp_status = getattr(e, "response_status", None)
                    logger.error(
                        "❌ Tool error: tool=%s err_type=%s err=%s | args=%s"
                        "%s%s",
                        tool_name,
                        type(e).__name__,
                        e,
                        tool_input,
                        f" | http_status={resp_status}" if resp_status is not None else "",
                        f" | response_body={resp_body!r}" if resp_body is not None else "",
                    )
                    try:
                        tool_span.set_error({
                            "message": str(e),
                            "data": {
                                "tool": tool_name,
                                "arguments": arguments,
                                "err_type": type(e).__name__,
                                "http_status": resp_status,
                                "response_body": resp_body,
                            },
                        })
                        tool_span.span_data.output = {"error": str(e), "tool": tool_name}
                    except Exception:
                        pass
                    if self.recorder is not None and self.recorder.is_recording:
                        self.recorder.record_tool_call(
                            tool_name, arguments, None,
                            duration_ms=int((time.perf_counter() - started_at) * 1000),
                            error=str(e),
                        )
                    return {"error": str(e), "tool": tool_name}

        def _ensure_entity(address: str, role: str, risk_score: float = 0.0, labels: list[str] | None = None, notes: str | None = None):
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
                    "dex_service": 4,
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

        def _add_step(path_id: str, step: dict[str, Any]):
            paths[path_id]["steps"].append(step)
            tx_hash = step.get("tx_hash")
            if tx_hash:
                edge_key = (tx_hash, step.get("from", ""), step.get("to", ""))
                path_seen_hashes.setdefault(path_id, set()).add(edge_key)

        def _copy_path(new_id: str, from_id: str):
            paths[new_id] = {
                "path_id": new_id,
                "description": paths[from_id]["description"],
                "steps": [dict(s) for s in paths[from_id]["steps"]],
                "stop_reason": None,
            }
            path_seen_addresses[new_id] = set(path_seen_addresses.get(from_id, set()))
            path_seen_hashes[new_id] = set(path_seen_hashes.get(from_id, set()))

        # Scheduler: priority queue keyed by (-attributed_amount, hop_index,
        # insertion_order). The completion budget is max_paths distinct
        # *finished* paths — continuations no longer count against the budget,
        # so high-attribution branches always get a chance to reach their
        # real CEX/mixer/bridge terminal instead of being starved by
        # shallower dead-end siblings. A hard iteration safety net
        # (max_paths * max_hops * 4) guards against pathological fan-out.
        hop_scheduler = HopScheduler(
            max_completed=max_paths,
            max_iterations=max_paths * max_hops * 4,
        )
        cap_annotation_emitted = False

        def _parse_get_tx_transfer(
            tx_data: dict[str, Any],
            exclude_address: str | None = None,
        ) -> dict[str, Any] | None:
            """Parse a get_transaction response into a transfer dict.

            For UTXO chains, picks the largest output whose address
            differs from *exclude_address* (the sender / change address).
            """
            if not isinstance(tx_data, dict) or not tx_data:
                return None

            inp = tx_data.get("input") or tx_data.get("inputs")
            out = tx_data.get("output") or tx_data.get("outputs")

            from_addr = None
            inp_riskscore = None
            if isinstance(inp, dict):
                from_addr = inp.get("address")
                inp_riskscore = inp.get("riskscore")
            elif isinstance(inp, list) and inp:
                first_inp = inp[0] if isinstance(inp[0], dict) else {}
                from_addr = first_inp.get("address")
                inp_riskscore = first_inp.get("riskscore")

            to_addr = None
            out_riskscore = None
            out_owner = None
            out_amount = None  # Track the selected output's amount for UTXO
            if isinstance(out, dict):
                to_addr = out.get("address")
                out_riskscore = out.get("riskscore")
                out_owner = out.get("owner")
                out_amount = out.get("amount")
            elif isinstance(out, list) and out:
                # UTXO: pick the largest output that isn't the sender (change)
                candidates = [o for o in out if isinstance(o, dict) and o.get("address") != exclude_address]
                if not candidates:
                    candidates = [o for o in out if isinstance(o, dict)]
                if candidates:
                    best = max(candidates, key=lambda o: float(o.get("amount", 0)))
                    to_addr = best.get("address")
                    out_riskscore = best.get("riskscore")
                    out_owner = best.get("owner")
                    out_amount = best.get("amount")

            if not from_addr:
                from_addr = tx_data.get("from") or tx_data.get("sender")
            if not to_addr:
                to_addr = tx_data.get("to") or tx_data.get("recipient")

            if not from_addr and not to_addr:
                return None

            # Resolve amount: prefer selected output amount (UTXO), then
            # top-level amount (account-model), then total_out/total_in.
            amount = (
                out_amount
                or tx_data.get("amount")
                or tx_data.get("total_out")
                or tx_data.get("total_in")
                or 0.0
            )

            return {
                "from": from_addr,
                "to": to_addr,
                "amount": amount,
                "block_time": tx_data.get("block_time"),
                "token_id": tx_data.get("token_id", 0),
                "output_owner": out_owner,
                "input_riskscore": inp_riskscore,
                "output_riskscore": out_riskscore,
            }

        def _parse_get_tx_transfers_utxo(
            tx_data: dict[str, Any],
            exclude_address: str | None = None,
            min_fraction: float = 0.01,
        ) -> list[dict[str, Any]]:
            """Parse a UTXO get_transaction response into a list of transfers.

            Returns one transfer dict per significant output (filtering dust
            and change outputs).  Falls back to a single-element list for
            account-model responses.
            """
            if not isinstance(tx_data, dict) or not tx_data:
                return []

            inp = tx_data.get("input") or tx_data.get("inputs")
            out = tx_data.get("output") or tx_data.get("outputs")

            # --- resolve sender ---
            from_addr = None
            inp_riskscore = None
            if isinstance(inp, dict):
                from_addr = inp.get("address")
                inp_riskscore = inp.get("riskscore")
            elif isinstance(inp, list) and inp:
                first_inp = inp[0] if isinstance(inp[0], dict) else {}
                from_addr = first_inp.get("address")
                inp_riskscore = first_inp.get("riskscore")

            if not from_addr:
                from_addr = tx_data.get("from") or tx_data.get("sender")
            if not from_addr:
                return []

            block_time = tx_data.get("block_time")
            token_id = tx_data.get("token_id", 0)
            fiat_rate = tx_data.get("fiat_rate") or tx_data.get("fiatRate")

            # --- resolve outputs ---
            if isinstance(out, dict):
                # Account-model: single output
                return [{
                    "from": from_addr,
                    "to": out.get("address"),
                    "amount": out.get("amount") or tx_data.get("amount") or 0.0,
                    "block_time": block_time,
                    "token_id": token_id,
                    "output_owner": out.get("owner"),
                    "input_riskscore": inp_riskscore,
                    "output_riskscore": out.get("riskscore"),
                    "fiat_rate": fiat_rate,
                }]

            if not isinstance(out, list) or not out:
                return []

            # UTXO: compute total to determine dust threshold
            candidates = [o for o in out if isinstance(o, dict) and o.get("address")]
            if not candidates:
                return []

            total_out = sum(float(o.get("amount", 0)) for o in candidates)
            threshold = total_out * min_fraction if total_out > 0 else 0

            results: list[dict[str, Any]] = []
            for o in candidates:
                addr = o.get("address")
                amt = float(o.get("amount", 0))
                # Skip change outputs (same address as sender)
                if addr == exclude_address:
                    continue
                # Skip dust
                if amt < threshold:
                    continue
                results.append({
                    "from": from_addr,
                    "to": addr,
                    "amount": amt,
                    "block_time": block_time,
                    "token_id": token_id,
                    "output_owner": o.get("owner"),
                    "input_riskscore": inp_riskscore,
                    "output_riskscore": o.get("riskscore"),
                    "fiat_rate": fiat_rate,
                })

            # Sort by amount descending so largest outputs come first
            results.sort(key=lambda t: float(t.get("amount", 0)), reverse=True)

            # Keep the full input/output arrays for visualization
            if results:
                results[0]["_raw_inputs"] = inp if isinstance(inp, list) else ([inp] if inp else [])
                results[0]["_raw_outputs"] = out
                results[0]["_fiat_rate"] = fiat_rate
                results[0]["_total_in"] = tx_data.get("total_in") or total_out
                results[0]["_total_out"] = tx_data.get("total_out") or total_out
                results[0]["_fee"] = tx_data.get("fee") or tx_data.get("weight")

            return results

        async def _resolve_transfer(
            tx_hash_val: str,
            chain_name: str,
            address_hint: str | None = None,
            token_id_val: int | None = None,
            expected_from: str | None = None,
            asset_hint: str | None = None,
        ) -> dict[str, Any] | None:
            """Try token_transfers first, fall back to get_transaction for native/UTXO txs.

            When we already know the transfer is of a chain's native asset
            (e.g. ETH on eth, TRX on trx) token_transfers always comes back
            empty, so we skip that round-trip and go straight to
            get_transaction.
            """
            # Inspect the already-fetched all_txs record (if any) to see
            # whether *this specific* tx is actually a token transfer. A
            # native-asset trace (asset_hint="ETH") can still legitimately
            # select a USDT/USDC/etc. tx on the hot-path — in that case we
            # MUST hit token_transfers, otherwise get_transaction returns the
            # token-contract address as ``to`` and the trace terminates at
            # ``0xdac17f…1ec7`` (USDT contract) instead of the real
            # recipient.
            tx_info_for_hash = all_txs_map.get(tx_hash_val) or {}
            tx_token_id = (
                tx_info_for_hash.get("token_id")
                if tx_info_for_hash else None
            )
            if tx_token_id is None and token_id_val is not None:
                tx_token_id = token_id_val
            tx_is_token_transfer = bool(tx_token_id) and int(tx_token_id) != 0

            skip_token_transfers = self._should_skip_token_transfers(
                chain=chain_name,
                asset_hint=asset_hint,
                address_hint=address_hint,
                tx_token_id=tx_token_id,
            )

            if not skip_token_transfers:
                transfer_result = await _call_tool("token_transfers", {
                    "tx_hash": tx_hash_val,
                    "blockchain_name": chain_name,
                })
                # For a known token tx, pass the tx's token_id explicitly
                # so _parse_transfer can filter out other tokens moved in
                # the same tx (multi-token swap routers etc.).
                parse_token_id = token_id_val
                if parse_token_id in (None, 0) and tx_is_token_transfer:
                    parse_token_id = int(tx_token_id)
                transfer = self._parse_transfer(
                    transfer_result,
                    expected_from=expected_from,
                    token_id=parse_token_id,
                )
                if transfer and transfer.get("to"):
                    return transfer

            # Fallback: get_transaction (needed for native ETH, BTC, BCH, LTC, etc.)
            if not address_hint:
                return None
            if not skip_token_transfers:
                logger.info(f"token_transfers empty for {tx_hash_val[:16]}...; falling back to get_transaction")
            tx_result = await _call_tool("get_transaction", {
                "address": address_hint,
                "tx_hash": tx_hash_val,
                "blockchain_name": chain_name,
                "token_id": 0,
                "path": "0",
            })
            tx_data = tx_result.get("data", {}) if isinstance(tx_result, dict) else {}
            if isinstance(tx_data, list) and tx_data:
                tx_data = tx_data[0]
            return _parse_get_tx_transfer(tx_data, exclude_address=address_hint)

        async def _resolve_utxo_outputs(
            tx_hash_val: str,
            chain_name: str,
            address_hint: str | None = None,
            token_id_val: int | None = None,
        ) -> list[dict[str, Any]]:
            """Resolve all significant outputs of a UTXO transaction.

            Calls get_transaction and returns a list of transfers, one per
            significant output.  Falls back to _resolve_transfer() wrapped
            in a single-element list if the UTXO parser returns nothing.
            """
            if not address_hint:
                single = await _resolve_transfer(
                    tx_hash_val, chain_name,
                    address_hint=address_hint,
                    token_id_val=token_id_val,
                )
                return [single] if single else []

            logger.info(f"Resolving UTXO outputs for {tx_hash_val[:16]}...")
            tx_result = await _call_tool("get_transaction", {
                "address": address_hint,
                "tx_hash": tx_hash_val,
                "blockchain_name": chain_name,
                "token_id": 0,
                "path": "0",
            })
            tx_data = tx_result.get("data", {}) if isinstance(tx_result, dict) else {}
            if isinstance(tx_data, list) and tx_data:
                tx_data = tx_data[0]

            outputs = _parse_get_tx_transfers_utxo(tx_data, exclude_address=address_hint)
            if outputs:
                return outputs

            # Fallback to single-transfer resolution
            single = await _resolve_transfer(
                tx_hash_val, chain_name,
                address_hint=address_hint,
                token_id_val=token_id_val,
            )
            return [single] if single else []

        async def _fetch_outgoing_txs(
            address: str,
            chain_name: str,
            incoming_time: int | None,
            token_id: int | None,
            max_pages: int = 5,
            page_limit: int = 50,
            incoming_amount: float | None = None,
            asset: str | None = None,
        ) -> list[dict[str, Any]]:
            """Fetch outgoing txs with pagination, ordered by time asc.

            Performs a primary fetch with transaction_type="withdrawal" (which
            applies a server-side `delta_coerced <= -0.0001` filter). When the
            returned items' total amount covers less than 70% of
            `incoming_amount`, a secondary fetch is issued with the
            `delta_coerced` filter disabled and the results are filtered
            client-side for outflows.

            Background: DEX swap transactions (e.g. 1inch USDT→DAI) can cause
            the per-address aggregate `delta_coerced` to be near zero or
            positive, so the server-side withdrawal filter silently drops
            them even though the specific token clearly flowed out. The
            fallback recovers those missing outflows.
            """

            def _ingest(items: list[dict[str, Any]], bucket: dict[str, dict[str, Any]]) -> None:
                for item in items:
                    tx_h = item.get("hash")
                    if not tx_h:
                        continue
                    bucket[tx_h] = item
                    all_txs_map[tx_h] = item

            async def _paginated_fetch(
                filter_obj: dict[str, Any],
                transaction_type: str,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                """Paginate `all_txs` and return (items, first_page_meta).

                We surface the first response's `meta` so the caller can check
                `filter_total` — if the server reports zero matches, further
                retries (fallback pass, wider-window pass) on the same address
                are guaranteed misses and can be skipped.
                """
                collected: list[dict[str, Any]] = []
                first_meta: dict[str, Any] = {}
                offset = 0
                pages = 0
                while pages < max_pages:
                    filter_arg = dict(filter_obj) if filter_obj else None
                    result = await _call_tool("all_txs", {
                        "address": address,
                        "blockchain_name": chain_name,
                        "filter": filter_arg,
                        "limit": page_limit,
                        "offset": offset,
                        "direction": "asc",
                        "order": "time",
                        "transaction_type": transaction_type,
                    })
                    if pages == 0 and isinstance(result, dict):
                        meta = result.get("meta")
                        if isinstance(meta, dict):
                            first_meta = meta
                    data_list = result.get("data", []) if isinstance(result, dict) else []
                    if not data_list:
                        break
                    collected.extend(data_list)
                    if len(data_list) < page_limit:
                        break
                    offset += page_limit
                    pages += 1
                return collected, first_meta

            base_filter: dict[str, Any] = {}
            if incoming_time:
                base_filter["time"] = {">=": incoming_time}
            if token_id not in (None, 0):
                base_filter["token_id"] = [token_id]

            primary_bucket: dict[str, dict[str, Any]] = {}
            primary_items, primary_meta = await _paginated_fetch(base_filter, "withdrawal")
            _ingest(primary_items, primary_bucket)

            # Dead-end short-circuit: when the server reports zero rows match
            # the full filter AND primary returned nothing, the delta-coerced
            # fallback will hit the same empty result set. Skip it instead of
            # firing another full pagination round.
            primary_filter_total = primary_meta.get("filter_total") if primary_meta else None
            if not primary_items and primary_filter_total == 0:
                return []

            # Coverage check: only attempt the fallback when we know what we
            # expected to see and the primary pass fell notably short.
            coverage_threshold = 0.7
            need_fallback = False
            if incoming_amount and incoming_amount > 0:
                total_primary = 0.0
                for item in primary_bucket.values():
                    amt = item.get("amount_coerced")
                    if amt is None:
                        amt = item.get("amount")
                    total_primary += self._normalize_amount(amt or 0.0, chain_name, asset)
                if total_primary < coverage_threshold * incoming_amount:
                    need_fallback = True
                    logger.info(
                        "Outgoing-tx coverage %.1f%% < %.0f%% for %s (incoming=%.2f, primary=%.2f); "
                        "retrying without delta_coerced filter",
                        (total_primary / incoming_amount) * 100,
                        coverage_threshold * 100,
                        self._format_address(address),
                        incoming_amount,
                        total_primary,
                    )

            merged: dict[str, dict[str, Any]] = dict(primary_bucket)
            if need_fallback:
                # Explicit empty delta_coerced → client.py strips the filter,
                # so the SAILS backend no longer drops swaps whose aggregate
                # delta sits near zero.
                fallback_filter = dict(base_filter)
                fallback_filter["delta_coerced"] = None
                fallback_items, _fallback_meta = await _paginated_fetch(fallback_filter, "all")

                added = 0
                for item in fallback_items:
                    tx_h = item.get("hash")
                    if not tx_h or tx_h in merged:
                        continue
                    # Client-side outflow check: keep items where the
                    # per-token delta is negative. Fall back to including the
                    # item when delta_coerced is absent so we don't regress on
                    # native/UTXO responses that lack the field.
                    delta = item.get("delta_coerced")
                    try:
                        delta_f = float(delta) if delta is not None else None
                    except (TypeError, ValueError):
                        delta_f = None
                    if delta_f is not None and delta_f >= 0:
                        continue
                    merged[tx_h] = item
                    all_txs_map[tx_h] = item
                    added += 1
                if added:
                    logger.info(
                        "Fallback fetch added %d missing outflow(s) for %s",
                        added,
                        self._format_address(address),
                    )

            def _sort_key(item: dict[str, Any]) -> int:
                bt = item.get("block_time") or item.get("time") or item.get("pool_time") or 0
                try:
                    return int(bt)
                except (TypeError, ValueError):
                    return 0

            return sorted(merged.values(), key=_sort_key)

        def _parse_bridge_info(result: Any) -> dict[str, Any]:
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

            def _find_dest_obj(obj: Any) -> dict[str, Any] | None:
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

            # Destination-chain fallback chain: some protocols (thorchain,
            # Chainflip) leave ``dst_chain`` null at the top level and put
            # the real chain into a nested field or encode it in the token
            # identifier. Consult those before giving up.
            if not dst_chain:
                from .currency_registry import normalize_external_chain, parse_thorchain_token_prefix
                recipient_ft = _find_key(data, {
                    "recipient_external_ft", "recipient_chain",
                    "to_chain", "destination_chain_name",
                })
                if recipient_ft:
                    mapped = normalize_external_chain(recipient_ft)
                    if mapped:
                        dst_chain = mapped
                if not dst_chain:
                    to_token = _find_key(data, {"to_token", "dst_token"})
                    mapped = parse_thorchain_token_prefix(to_token)
                    if mapped:
                        dst_chain = mapped

            # Separate destination-side amount from source-side amount.
            # Source-only fields (``from_amount``, generic ``amount``) are
            # denominated in the *source* asset and must NOT be used when
            # the bridge swaps assets (Bridgers: ``from_amount`` in ETH
            # wei, no ``to_amount`` at all → would be normalized as USDT).
            to_amount = _find_key(data, {
                "to_amount", "dst_amount", "received_amount",
                "amount_out", "output_amount", "outputAmount",
            })
            from_amount = _find_key(data, {
                "from_amount", "src_amount", "source_amount",
                "input_amount", "amount_in",
            })
            if from_amount is None:
                # Generic ``amount`` is source-side in thorchain/Bridgers.
                from_amount = _find_key(data, {"amount"})
            # Legacy field consumed by downstream code: prefer destination
            # amount when available; fall back to source only when same-asset.
            amount_out = to_amount if to_amount is not None else from_amount
            dst_tx_hash = _find_key(data, {"dst_tx_hash", "destination_tx_hash", "dstTxHash"})
            dst_block_time = _find_key(data, {"dst_block_time", "destination_block_time", "dstBlockTime"})
            protocol = _find_key(data, {"protocol", "bridge", "service", "bridge_name"})
            # Source-side timestamp, used to time-match the destination leg
            # when the API doesn't provide ``to_amount`` or ``dst_tx_hash``
            # (Bridgers shape). Accept both Unix seconds and ISO-8601.
            src_ts_raw = _find_key(data, {
                "timestamp_iso", "timestamp", "block_time", "time",
            })
            src_ts: int | None = None
            if isinstance(src_ts_raw, (int, float)):
                src_ts = int(src_ts_raw)
            elif isinstance(src_ts_raw, str):
                txt = src_ts_raw.strip()
                if txt.isdigit():
                    src_ts = int(txt)
                else:
                    try:
                        from datetime import datetime
                        # ``fromisoformat`` on 3.11+ handles the trailing ``Z``.
                        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=UTC)
                        src_ts = int(dt.timestamp())
                    except (ValueError, TypeError):
                        src_ts = None

            # Destination asset (when the bridge also swaps). Registry
            # owns the decimals, so we deliberately ignore any
            # ``to_token_decimals`` the API offers — it's been observed
            # to be wrong (thorchain returns 18 for BTC).
            dst_asset_raw = _find_key(data, {
                "to_token_symbol", "dst_asset", "dst_symbol",
                "destination_asset", "destination_symbol",
            })
            dst_asset: str | None = None
            if isinstance(dst_asset_raw, str) and dst_asset_raw.strip():
                dst_asset = dst_asset_raw.strip().upper()

            if not is_bridge and (dst_chain or dst_addr):
                is_bridge = True

            return {
                "is_bridge": bool(is_bridge),
                "dst_chain": dst_chain,
                "dst_address": dst_addr,
                "dst_asset": dst_asset,
                "amount_out": amount_out,
                "to_amount": to_amount,
                "from_amount": from_amount,
                "dst_tx_hash": dst_tx_hash,
                "dst_block_time": dst_block_time,
                "protocol": protocol,
                "src_ts": src_ts,
            }

        async def _resolve_token_id_for_chain(
            chain_name: str,
            address: str,
            asset_symbol: str,
            fallback: int | None = None,
        ) -> int | None:
            if not asset_symbol or not address:
                return fallback
            # Registry-first: the currency DB already knows the canonical
            # token_id for popular (chain, symbol) pairs (e.g. USDT/TRX =
            # 9) and avoids a network round-trip plus side effects on the
            # replay-recorder's FIFO queue. Fall back to ``token_stats``
            # when the registry doesn't know about the pair.
            try:
                from .currency_registry import get_registry
                rec = get_registry().lookup_by_symbol(chain_name, asset_symbol)
                if rec is not None and rec.token_id is not None:
                    return int(rec.token_id)
            except Exception:
                pass
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
                # ``token_stats`` rows don't always carry a ``symbol``;
                # cross-check via the currency registry using each
                # reported token_id.
                try:
                    from .currency_registry import get_registry
                    reg = get_registry()
                    for item in data_list:
                        tid = item.get("token_id") or item.get("tokenId")
                        if tid is None:
                            continue
                        rec = reg.lookup(chain_name, int(tid))
                        if rec is None:
                            continue
                        if (rec.symbol or "").upper() == asset_symbol.upper():
                            return int(tid)
                except Exception:
                    pass
            except Exception:
                return fallback
            return fallback

        # Seed path(s)
        if tx_hash:
            is_utxo_chain = chain in self._SATOSHI_CHAINS

            if is_utxo_chain:
                # ── UTXO multi-output: resolve ALL significant outputs ──
                utxo_outputs = await _resolve_utxo_outputs(
                    tx_hash, chain,
                    address_hint=victim_address,
                    token_id_val=token_id_hint,
                )
                if not utxo_outputs or not utxo_outputs[0].get("from"):
                    raise RuntimeError("Unable to extract theft transfer details from tx_hash")

                from_addr = utxo_outputs[0]["from"]
                block_time = utxo_outputs[0].get("block_time")
                token_id = utxo_outputs[0].get("token_id") or token_id_hint

                # Total theft amount across all outputs
                total_raw = sum(float(o.get("amount", 0)) for o in utxo_outputs)
                theft_amount = self._resolve_amount(tx_hash, total_raw, chain, all_txs_map, asset)

                _ensure_entity(from_addr, "victim", 0.0, notes="Victim")
                if utxo_outputs[0].get("input_riskscore") is not None:
                    risk_map[from_addr] = float(utxo_outputs[0].get("input_riskscore") or 0.0)

                if stolen_amount <= 0:
                    stolen_amount = theft_amount
                    fifo_ledger.stolen_amount = stolen_amount
                    fifo_ledger.cap = stolen_amount * (1.0 + traced_tolerance) if stolen_amount > 0 else float("inf")

                # Collect full UTXO tx data for visualization once
                self._collect_utxo_tx_data(
                    tx_hash, chain, utxo_outputs,
                    risk_map, txs_collected, tx_list_collected, txs_seen,
                )

                seen_utxo_recipients: set[str] = set()
                for idx, uout in enumerate(utxo_outputs):
                    to_addr = uout.get("to")
                    if not to_addr or to_addr in seen_utxo_recipients:
                        continue
                    seen_utxo_recipients.add(to_addr)

                    out_amount = self._resolve_amount(tx_hash, uout.get("amount", 0.0), chain, all_txs_map, asset)
                    if uout.get("output_owner"):
                        owner_hints[to_addr] = uout.get("output_owner")
                    if uout.get("output_riskscore") is not None:
                        risk_map[to_addr] = float(uout.get("output_riskscore") or 0.0)

                    path_id = "1" if idx == 0 else str(path_counter + idx)
                    if path_id not in paths:
                        paths[path_id] = {
                            "path_id": path_id,
                            "description": f"Theft output branch {idx + 1}",
                            "steps": [],
                            "stop_reason": None,
                        }

                    fifo_ledger.record_inflow(to_addr, out_amount, out_amount)
                    _add_step(path_id, {
                        "step_index": 0,
                        "from": from_addr,
                        "to": to_addr,
                        "tx_hash": tx_hash,
                        "chain": chain,
                        "asset": asset,
                        "amount_estimate": out_amount,
                        "attributed_amount": out_amount,
                        "time": block_time,
                        "direction": "out",
                        "step_type": "direct_transfer",
                        "service_label": None,
                        "protocol": None,
                        "reasoning": f"UTXO output {idx + 1}/{len(utxo_outputs)} from theft transaction.",
                    })

                    hop_scheduler.push(HopJob(
                        path_id=path_id,
                        current_address=to_addr,
                        incoming_tx_hash=tx_hash,
                        incoming_amount=out_amount,
                        incoming_time=block_time,
                        chain=chain,
                        asset=asset,
                        token_id=int(token_id or 0),
                        hop_index=1,
                        attributed_amount=out_amount,
                    ))

                path_counter = max(path_counter, 1 + len(utxo_outputs))

            else:
                # ── Account-model chain: single transfer (existing logic) ──
                transfer = await _resolve_transfer(
                    tx_hash, chain,
                    address_hint=victim_address,
                    token_id_val=token_id_hint,
                    asset_hint=asset,
                )
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
                theft_amount = float(amount or 0.0)
                if stolen_amount <= 0:
                    stolen_amount = theft_amount
                    fifo_ledger.stolen_amount = stolen_amount
                    fifo_ledger.cap = stolen_amount * (1.0 + traced_tolerance) if stolen_amount > 0 else float("inf")
                fifo_ledger.record_inflow(to_addr, theft_amount, theft_amount)
                _add_step("1", {
                    "step_index": 0,
                    "from": from_addr,
                    "to": to_addr,
                    "tx_hash": tx_hash,
                    "chain": chain,
                    "asset": asset,
                    "amount_estimate": theft_amount,
                    "attributed_amount": theft_amount,
                    "time": block_time,
                    "direction": "out",
                    "step_type": "direct_transfer",
                    "service_label": None,
                    "protocol": None,
                    "reasoning": "Primary theft transaction provided by user.",
                })

                self._collect_token_transfer_data(
                    json.dumps({"data": [{"input": {"address": transfer.get("from")}, "output": {"address": transfer.get("to"), "owner": transfer.get("output_owner")}, "amount": transfer.get("amount", 0), "block_time": transfer.get("block_time"), "token_id": transfer.get("token_id", 0)}]}, ensure_ascii=False),
                    {"tx_hash": tx_hash, "blockchain_name": chain},
                    all_txs_map,
                    risk_map,
                    txs_collected,
                    tx_list_collected,
                    txs_seen,
                )

                hop_scheduler.push(HopJob(
                    path_id="1",
                    current_address=to_addr,
                    incoming_tx_hash=tx_hash,
                    incoming_amount=theft_amount,
                    incoming_time=block_time,
                    chain=chain,
                    asset=asset,
                    token_id=int(token_id or 0),
                    hop_index=1,
                    attributed_amount=theft_amount,
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
            filter_obj: dict[str, Any] = {}
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
            selected_hashes = self._accumulate_hashes(data_list, None, chain, asset=asset)
            used_accumulation = bool(selected_hashes)
            seed_selector_decision: DecisionRef | None = None
            if not selected_hashes and data_list:
                # Fallback to selector only if accumulation can't decide
                selector_context = {
                    "chain": chain,
                    "asset": asset,
                    "incoming_amount": None,
                    "incoming_time": date_ts,
                    "txs": data_list,
                }
                selector_result, seed_selector_decision = await self._run_selector(selector_context)
                selected_hashes = (selector_result or {}).get("selected_hashes") or []
                used_accumulation = False
            if not selected_hashes and data_list:
                first_hash = data_list[0].get("hash")
                selected_hashes = [first_hash] if first_hash else []
                used_accumulation = False

            if not used_accumulation:
                selected_hashes = selected_hashes[:max_paths]

            # Pre-resolve all seed transfers in parallel; the mutation loop
            # below stays sequential so path_counter / fifo_ledger ordering is
            # preserved.
            _prefetched_transfers = await asyncio.gather(*(
                _resolve_transfer(
                    sel_hash, chain,
                    address_hint=victim_address,
                    token_id_val=token_id_hint,
                    expected_from=victim_address,
                    asset_hint=asset,
                )
                for sel_hash in selected_hashes
            )) if selected_hashes else []

            seen_recipients: set = set()
            for idx, (sel_hash, transfer) in enumerate(zip(selected_hashes, _prefetched_transfers, strict=True)):
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
                step_amount = float(amount or 0.0)
                if stolen_amount <= 0 and idx == 0:
                    stolen_amount = step_amount
                    fifo_ledger.stolen_amount = stolen_amount
                    fifo_ledger.cap = stolen_amount * (1.0 + traced_tolerance) if stolen_amount > 0 else float("inf")
                fifo_ledger.record_inflow(to_addr, step_amount, step_amount)
                _add_step(path_id, {
                    "step_index": 0,
                    "from": victim_address,
                    "to": to_addr,
                    "tx_hash": sel_hash,
                    "chain": chain,
                    "asset": asset,
                    "amount_estimate": step_amount,
                    "attributed_amount": step_amount,
                    "time": block_time,
                    "direction": "out",
                    "step_type": "direct_transfer",
                    "service_label": None,
                    "protocol": None,
                    "reasoning": "Selected as primary theft candidate from victim outflows.",
                    "llm_decisions": (
                        [seed_selector_decision.model_dump()] if seed_selector_decision else []
                    ),
                })

                self._collect_token_transfer_data(
                    json.dumps({"data": [{"input": {"address": transfer.get("from")}, "output": {"address": transfer.get("to"), "owner": transfer.get("output_owner")}, "amount": transfer.get("amount", 0), "block_time": transfer.get("block_time"), "token_id": transfer.get("token_id", 0)}]}, ensure_ascii=False),
                    {"tx_hash": sel_hash, "blockchain_name": chain},
                    all_txs_map,
                    risk_map,
                    txs_collected,
                    tx_list_collected,
                    txs_seen,
                )

                hop_scheduler.push(HopJob(
                    path_id=path_id,
                    current_address=to_addr,
                    incoming_tx_hash=sel_hash,
                    incoming_amount=step_amount,
                    incoming_time=block_time,
                    chain=chain,
                    asset=asset,
                    token_id=int(token_id or 0),
                    hop_index=1,
                    attributed_amount=step_amount,
                ))

        else:
            raise RuntimeError("victim_address or tx_hash is required")

        def _completed_paths_count() -> int:
            # Dust-trimmed branches are noise stops (sibling below the
            # attribution threshold), not real terminals. Counting them
            # toward the ``max_completed`` budget in HopScheduler starves
            # the live HopJobs: a single fan-out of 25 outflows can
            # produce 10+ dust siblings on hop 1 alone, tripping the
            # scheduler's stop condition before any hop-2 work executes.
            # Only count paths that reached a genuine endpoint.
            return sum(
                1 for p in paths.values()
                if p.get("stop_reason") and p.get("path_id") not in dust_trimmed_paths
            )

        async def _agentic_phase1_for_job(job_local: HopJob):
            _coros_pf: list = [
                _call_tool("get_address", {
                    "blockchain_name": job_local.chain,
                    "address": job_local.current_address,
                }),
                _call_tool("get_extra_address_info", {
                    "address": job_local.current_address,
                    "asset": job_local.asset,
                }),
            ]
            _eb_pf = bool(job_local.incoming_tx_hash and job_local.hop_index <= 3)
            if _eb_pf:
                _coros_pf.append(_call_tool("bridge_analyze", {
                    "model": BRIDGE_ANALYZER_MODEL,
                    "chain": job_local.chain,
                    "tx_hash": job_local.incoming_tx_hash,
                }))
            _p1_pf = await asyncio.gather(*_coros_pf)
            if _eb_pf:
                return _p1_pf[0], _p1_pf[1], _p1_pf[2]
            return _p1_pf[0], _p1_pf[1], None

        async def _agentic_hop_after_phase1(
            job: HopJob,
            get_addr_result: Any,
            get_extra_result: Any,
            _early_bridge_result: Any | None,
        ) -> None:
            nonlocal path_counter
            risk_score = self._extract_risk_score(get_addr_result)
            risk_map[job.current_address] = risk_score

            # Store raw get_address data for visualization addressInfo
            try:
                _addr_data = get_addr_result.get("data", {}) if isinstance(get_addr_result, dict) else {}
                if _addr_data and job.chain:
                    self.last_address_info.setdefault(job.current_address, {})[self._normalize_chain(job.chain)] = _addr_data
            except Exception:
                pass

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
            classification_raw, classifier_decision = await self._run_hop_classifier(classifier_context)
            classification = classification_raw or {}
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

            if job.hop_index == 1 and not owner and not heuristic.get("terminal"):
                # First hop with no confirmed owner → suspect perpetrator.
                # Weak signals (e.g. use_platform) don't count as identity.
                role = "perpetrator"
                if "Suspected Perpetrator" not in labels:
                    labels.append("Suspected Perpetrator")

            # Early-hop safeguard: don't terminate at addresses that have no
            # confirmed owner in the first few hops.  The heuristic may flag
            # an address as terminal from a weak use_platform signal alone.
            if terminal and job.hop_index <= 3 and not owner:
                logger.info(
                    "Overriding terminal=True for unowned address %s at hop %d (role=%s)",
                    self._format_address(job.current_address), job.hop_index, role,
                )
                terminal = False
                stop_reason = None

            _ensure_entity(job.current_address, role, risk_score, labels=labels, notes=notes)

            if risk_score and risk_score > 0.75:
                annotations.append({
                    "id": f"ann-{len(annotations)+1}",
                    "label": "High Risk",
                    "related_addresses": [job.current_address],
                    "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                    "text": f"Passed through high-risk address (score: {risk_score:.2f})",
                })

            # OTC-like analysis flag — deferred until after outgoing-tx fetch to reuse data
            _run_otc = (
                not terminal
                and role in ("otc_service", "unidentified_service")
            )

            # Bridge detection & continuation
            bridge_info = None
            if job.incoming_tx_hash:
                service_label = classification.get("service_label") or heuristic.get("service_label") or ""
                bridge_candidate = (
                    role == "bridge_service"
                    or heuristic.get("role") == "bridge_service"
                    or ("bridge" in str(service_label).lower())
                )
                if _early_bridge_result is not None:
                    bridge_info = _parse_bridge_info(_early_bridge_result)
                elif bridge_candidate:
                    bridge_result = await _call_tool("bridge_analyze", {
                        "model": BRIDGE_ANALYZER_MODEL,
                        "chain": job.chain,
                        "tx_hash": job.incoming_tx_hash,
                    })
                    bridge_info = _parse_bridge_info(bridge_result)

            if bridge_info and bridge_info.get("is_bridge"):
                dst_chain = bridge_info.get("dst_chain")
                dst_address = bridge_info.get("dst_address")
                dst_tx_hash = bridge_info.get("dst_tx_hash")
                dst_block_time = bridge_info.get("dst_block_time")
                to_amount_api = bridge_info.get("to_amount")
                from_amount_api = bridge_info.get("from_amount")

                if dst_chain and dst_address:
                    # Promote to bridge service if tool confirms with destination
                    role = "bridge_service"
                    terminal = True
                    if "Bridge" not in labels:
                        labels.append("Bridge")
                    _ensure_entity(job.current_address, role, risk_score, labels=labels, notes=notes)

                    dst_chain_norm = self._normalize_chain(dst_chain)
                    # Destination asset (for cross-asset bridges like
                    # thorchain swap-bridges). When the API gives us
                    # ``dst_asset``, we switch on it; otherwise the
                    # trace continues on the same asset as before.
                    dst_asset_api = bridge_info.get("dst_asset")
                    new_asset = (dst_asset_api or job.asset or "").upper() or job.asset
                    asset_changed = bool(dst_asset_api) and new_asset != (job.asset or "").upper()

                    # Pick the raw amount to carry across the bridge.
                    # Cross-asset bridges: only ``to_amount`` is meaningful;
                    # ``from_amount`` is in source-asset units and must not
                    # be normalized against the destination decimals.
                    matched_dst_tx_hash: str | None = None
                    matched_dst_block_time: int | None = None
                    dst_token_id_resolved: int | None = None
                    if to_amount_api is not None:
                        bridge_amount = self._normalize_amount(
                            to_amount_api, dst_chain_norm, new_asset,
                        )
                    elif not asset_changed:
                        raw = from_amount_api if from_amount_api is not None else (job.incoming_amount or 0.0)
                        bridge_amount = self._normalize_amount(raw, dst_chain_norm, new_asset)
                    else:
                        # Asset swapped and no destination amount in the
                        # bridge response (Bridgers shape). Time-match:
                        # look up the first incoming ``new_asset`` tx to
                        # ``dst_address`` on ``dst_chain`` within a short
                        # window around the source-tx timestamp. That
                        # recovers the real received amount (can't be
                        # inferred from ``from_amount``, which is in
                        # source units) and a real dst_tx_hash, avoiding
                        # the "MOCK DATA" tooltip produced by a synthetic
                        # ``tx-…`` hash.
                        #
                        # Window: [src_ts - 60, src_ts + 1800]. Bridge
                        # legs typically settle in under 10 minutes;
                        # 30 min gives headroom for congested chains
                        # without letting unrelated later deposits win
                        # on busy addresses. The 60 s underbound absorbs
                        # clock skew between ETH and TRON block times.
                        bridge_amount = 0.0
                        BRIDGE_MATCH_WINDOW_SECS = 1800
                        src_ts = bridge_info.get("src_ts") or dst_block_time or job.incoming_time
                        if src_ts and dst_address:
                            try:
                                dst_token_id_hint = await _resolve_token_id_for_chain(
                                    dst_chain_norm, dst_address, new_asset, fallback=None,
                                )
                                dst_token_id_resolved = dst_token_id_hint
                                filter_obj: dict[str, Any] = {
                                    "time": {
                                        ">=": int(src_ts) - 60,
                                        "<=": int(src_ts) + BRIDGE_MATCH_WINDOW_SECS,
                                    },
                                }
                                if dst_token_id_hint:
                                    filter_obj["token_id"] = [int(dst_token_id_hint)]
                                incoming = await _call_tool("all_txs", {
                                    "address": dst_address,
                                    "blockchain_name": dst_chain_norm,
                                    "filter": filter_obj,
                                    "limit": 25, "offset": 0,
                                    "direction": "asc", "order": "time",
                                    "transaction_type": "deposit",
                                })
                                rows = incoming.get("data", []) if isinstance(incoming, dict) else []
                                for row in rows:
                                    if not isinstance(row, dict):
                                        continue
                                    # Asset / token match:
                                    row_tid = row.get("token_id")
                                    if dst_token_id_hint and row_tid is not None and int(row_tid) != int(dst_token_id_hint):
                                        continue
                                    row_asset = row.get("asset") or row.get("token_symbol")
                                    row_symbol = (str(row_asset).upper() if row_asset else None)
                                    if row_symbol and row_symbol != new_asset:
                                        continue
                                    # Skip zero-value rows (TRX fee rows
                                    # and rejected txs); we want the
                                    # real asset transfer.
                                    raw_amt = (
                                        row.get("amount_coerced")
                                        or row.get("amount")
                                        or 0
                                    )
                                    try:
                                        if float(raw_amt) <= 0:
                                            continue
                                    except (TypeError, ValueError):
                                        continue
                                    if (row.get("type") or "") == "rejected":
                                        continue
                                    bridge_amount = self._normalize_amount(
                                        raw_amt, dst_chain_norm, new_asset,
                                    )
                                    bt = row.get("block_time") or row.get("pool_time")
                                    matched_dst_tx_hash = row.get("hash") or row.get("tx_hash")
                                    matched_dst_block_time = int(bt) if bt else None
                                    break
                            except Exception as exc:  # best-effort; leave bridge_amount=0
                                logger.warning(
                                    "Bridge time-match failed for %s on %s: %s",
                                    self._format_address(dst_address), dst_chain_norm, exc,
                                )
                    if matched_dst_tx_hash is not None:
                        dst_tx_hash = matched_dst_tx_hash
                    if matched_dst_block_time is not None:
                        dst_block_time = matched_dst_block_time
                    if to_amount_api is not None and job.incoming_amount and not asset_changed:
                        # Aggregation-detection only makes sense when
                        # source and destination asset are the same; a
                        # cross-asset bridge trivially "differs" and
                        # would flood annotations.
                        gap = abs(bridge_amount - float(job.incoming_amount or 0.0)) / max(float(job.incoming_amount or 1.0), 1.0)
                        if gap > 0.2:
                            annotations.append({
                                "id": f"ann-{len(annotations)+1}",
                                "label": "Bridge Aggregation",
                                "related_addresses": [job.current_address, dst_address],
                                "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                                "text": f"Bridge aggregation detected - output amount ({bridge_amount}) differs from input ({job.incoming_amount}).",
                            })

                    bridge_step_amount = float(bridge_amount or 0.0)
                    bridge_raw_attr = fifo_ledger.attribute_outflow(job.current_address, bridge_step_amount)
                    # Cross-asset bridge: FIFO tracks source-asset units so
                    # the returned attribution (e.g. 0.14 ETH) is not
                    # comparable to the destination-denominated amount
                    # (e.g. 590 USDT). Treat the time-matched destination
                    # deposit as fully attributable — the bridge swap
                    # mapped the whole source inflow into this outflow.
                    if asset_changed and bridge_step_amount > 0:
                        bridge_raw_attr = bridge_step_amount
                    fifo_ledger.record_inflow(dst_address, bridge_step_amount, bridge_raw_attr)

                    step_index = len(paths[job.path_id]["steps"])
                    _add_step(job.path_id, {
                        "step_index": step_index,
                        "from": job.current_address,
                        "to": dst_address,
                        "tx_hash": dst_tx_hash,
                        "chain": dst_chain_norm,
                        "asset": new_asset,
                        "amount_estimate": bridge_step_amount,
                        "attributed_amount": bridge_raw_attr,
                        "time": int(dst_block_time) if dst_block_time else job.incoming_time,
                        "direction": "out",
                        "step_type": "bridge_transfer",
                        "service_label": classification.get("service_label") or heuristic.get("service_label") or "Bridge",
                        "protocol": bridge_info.get("protocol") or classification.get("protocol"),
                        "reasoning": (
                            f"Bridge {job.asset}→{new_asset} detected; continuing on {dst_chain_norm}."
                            if asset_changed else "Bridge detected; continuing on destination chain."
                        ),
                        "llm_decisions": (
                            [classifier_decision.model_dump()] if classifier_decision else []
                        ),
                    })
                    if asset_changed:
                        annotations.append({
                            "id": f"ann-{len(annotations)+1}",
                            "label": "Bridge Asset Swap",
                            "related_addresses": [job.current_address, dst_address],
                            "related_steps": [f"{job.path_id}:{step_index}"],
                            "text": (
                                f"Bridge swapped {job.asset} → {new_asset}; downstream "
                                f"dust threshold is re-anchored on the destination amount."
                            ),
                        })

                    if dst_token_id_resolved is not None:
                        new_token_id = dst_token_id_resolved
                    else:
                        new_token_id = await _resolve_token_id_for_chain(
                            dst_chain_norm,
                            dst_address,
                            new_asset,
                            fallback=job.token_id if not asset_changed else 0,
                        )

                    # Dust guard. When the bridge also swapped the asset,
                    # the source stolen_amount ("0.14 ETH") isn't
                    # comparable to the destination amount ("2.89 BTC"),
                    # so we re-anchor on the outgoing amount — anything
                    # that makes it across should be worth tracing until
                    # its own descendant shrinks below 1% of ITS inflow.
                    dust_anchor = stolen_amount
                    if asset_changed and bridge_step_amount > 0:
                        dust_anchor = bridge_step_amount
                    # Skip dust guard entirely when:
                    #   1. The bridge response lacked a destination
                    #      amount (Bridgers shape with no time-match
                    #      hit): 0 vs anything would always trip.
                    #   2. The bridge swapped the asset: bridge_raw_attr
                    #      above was set to bridge_step_amount so the
                    #      ratio is always 1.0 — but skip anyway so the
                    #      intent is explicit; the destination-side
                    #      HopJob will enforce its own dust rule once
                    #      real downstream txs appear.
                    have_dest_amount = to_amount_api is not None or bridge_step_amount > 0
                    bridge_dust_hit = (
                        have_dest_amount
                        and not asset_changed
                        and dust_anchor > 0
                        and min_attribution_ratio > 0.0
                        and bridge_raw_attr < dust_anchor * min_attribution_ratio
                    )
                    if bridge_dust_hit:
                        dust_pct = (bridge_raw_attr / dust_anchor) * 100.0 if dust_anchor else 0.0
                        paths[job.path_id]["stop_reason"] = (
                            f"Below dust threshold ({dust_pct:.2f}% of "
                            f"{'destination' if asset_changed else 'stolen'} amount) (bridge)"
                        )
                        if job.path_id not in dust_trimmed_paths:
                            annotations.append({
                                "id": f"ann-{len(annotations)+1}",
                                "label": "Dust Trimmed",
                                "related_addresses": [dst_address],
                                "related_steps": [f"{job.path_id}:{step_index}"],
                                "text": (
                                    f"Bridge leg trimmed at {self._format_address(dst_address)}: "
                                    f"attributed {bridge_raw_attr:.2f} < "
                                    f"{min_attribution_ratio*100:.2f}% of {dust_anchor:.2f} "
                                    f"({'destination' if asset_changed else 'stolen'} anchor)"
                                ),
                            })
                            dust_trimmed_paths.add(job.path_id)
                        return

                    hop_scheduler.push(HopJob(
                        path_id=job.path_id,
                        current_address=dst_address,
                        incoming_tx_hash=dst_tx_hash or job.incoming_tx_hash,
                        incoming_amount=bridge_step_amount,
                        incoming_time=int(dst_block_time) if dst_block_time else job.incoming_time,
                        chain=dst_chain_norm,
                        asset=new_asset,
                        token_id=int(new_token_id or (job.token_id if not asset_changed else 0) or 0),
                        hop_index=job.hop_index + 1,
                        attributed_amount=bridge_raw_attr,
                    ))
                    # Continue on destination chain
                    return

                # If tool hints at bridge but no destination, annotate and fall through
                # to outgoing-tx search instead of stopping immediately.
                if role == "bridge_service" or heuristic.get("role") == "bridge_service":
                    annotations.append({
                        "id": f"ann-{len(annotations)+1}",
                        "label": "Bridge - Destination Unknown",
                        "related_addresses": [job.current_address],
                        "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                        "text": "Bridge detected but destination chain/address unknown. Checking outgoing transactions.",
                    })
                    # Don't continue — fall through to outgoing tx search below

            if terminal:
                fifo_ledger.claim_terminal(job.attributed_amount)
                paths[job.path_id]["stop_reason"] = stop_reason or "Reached terminal entity"
                logger.info(
                    "Path %s stopped at %s (hop %d): %s | terminal=%s, role=%s, owner=%s",
                    job.path_id, self._format_address(job.current_address), job.hop_index,
                    stop_reason, terminal, role, owner,
                )
                return

            # Find next outgoing transactions (chronological accumulation)
            data_list = await _fetch_outgoing_txs(
                job.current_address,
                job.chain,
                job.incoming_time,
                job.token_id,
                incoming_amount=job.incoming_amount,
                asset=job.asset,
            )
            # If no results with specific token, retry with native token (asset conversion on exchange)
            if not data_list and job.token_id not in (None, 0):
                logger.info("No outgoing txs for token_id=%s, retrying with native token", job.token_id)
                data_list = await _fetch_outgoing_txs(
                    job.current_address,
                    job.chain,
                    job.incoming_time,
                    0,
                    incoming_amount=job.incoming_amount,
                    asset=job.asset,
                )
            # Retry with wider time window (5 min earlier) in case the
            # original filter was too tight for near-instant relays.
            if not data_list and job.incoming_time:
                wider_time = (
                    job.incoming_time - 300
                    if isinstance(job.incoming_time, (int, float))
                    else job.incoming_time
                )
                logger.info("No outgoing txs, retrying with wider time window for %s", self._format_address(job.current_address))
                data_list = await _fetch_outgoing_txs(
                    job.current_address,
                    job.chain,
                    wider_time,
                    job.token_id,
                    incoming_amount=job.incoming_amount,
                    asset=job.asset,
                )
            if not data_list:
                fifo_ledger.claim_terminal(job.attributed_amount)
                paths[job.path_id]["stop_reason"] = "Dead end - no outgoing transactions"
                logger.info("Path %s stopped at %s (hop %d): dead end — no outgoing txs", job.path_id, self._format_address(job.current_address), job.hop_index)
                return

            # --- Bridge-deposit detection (NEAR Intents / similar) ---
            # Per-swap deposit addresses (NEAR Intents) forward funds
            # into a stable bridge-owned aggregator. The aggregator is
            # what carries the brand tag in the API; the deposit itself
            # looks like a plain intermediate. Without this check the
            # trace would follow the deposit→aggregator hop and stop
            # inside the bridge's internal plumbing instead of crossing
            # chains.
            #
            # The detection is only a *signal* — we re-run
            # ``bridge_analyze`` on the FUNDING tx to confirm. Two
            # outcomes:
            #
            #   1. ``is_bridge=true`` — genuine per-swap deposit
            #      (the deposit address is bridge-managed, so the
            #      bridge tx IS the funding tx). Reclassify as
            #      ``bridge_service`` and follow the destination.
            #
            #   2. ``is_bridge=false`` — the current address is an
            #      ordinary user wallet that just happens to forward
            #      funds to a bridge service (Bridgers etc.). The
            #      bridge tx is the OUTGOING tx to the bridge contract,
            #      not the incoming tx. Fall through silently to
            #      normal outflow processing — the existing bridge
            #      handler at ``_agentic_hop_after_phase1`` line 2768
            #      will pick up the bridge contract on the next hop
            #      via ``bridge_analyze`` against the outgoing tx.
            #
            # We therefore DO NOT clobber ``role``, ``terminal``,
            # ``labels`` or the entity until the bridge_analyze
            # confirmation is in.
            if not terminal and job.incoming_tx_hash:
                deposit_brand_owner = self._detect_bridge_deposit_pattern(data_list)
                if deposit_brand_owner:
                    brand_label = (
                        deposit_brand_owner.get("name")
                        or deposit_brand_owner.get("slug")
                        or "Bridge"
                    )
                    logger.info(
                        "Bridge-deposit pattern at %s (hop %d): outflows aggregate to %r; "
                        "re-running bridge_analyze on incoming tx %s",
                        self._format_address(job.current_address), job.hop_index,
                        brand_label,
                        (job.incoming_tx_hash or "")[:16],
                    )

                    late_bridge_result = await _call_tool("bridge_analyze", {
                        "model": BRIDGE_ANALYZER_MODEL,
                        "chain": job.chain,
                        "tx_hash": job.incoming_tx_hash,
                    })
                    late_bridge_info = _parse_bridge_info(late_bridge_result)

                    if late_bridge_info and late_bridge_info.get("is_bridge"):
                        # CONFIRMED bridge deposit — now we can
                        # safely reclassify and follow the destination
                        # (or annotate destination-unknown).
                        role = "bridge_service"
                        terminal = True
                        if "Bridge" not in labels:
                            labels.append("Bridge")
                        if brand_label and brand_label not in labels:
                            labels.append(str(brand_label))
                        if not classification.get("service_label"):
                            classification["service_label"] = brand_label
                        _ensure_entity(
                            job.current_address, role, risk_score,
                            labels=labels, notes=notes,
                        )

                        dst_chain_raw = late_bridge_info.get("dst_chain")
                        dst_address = late_bridge_info.get("dst_address")
                        if dst_chain_raw and dst_address:
                            dst_chain_norm = self._normalize_chain(dst_chain_raw)
                            new_asset = (
                                (late_bridge_info.get("dst_asset") or job.asset or "").upper()
                                or job.asset
                            )
                            asset_changed = bool(
                                late_bridge_info.get("dst_asset")
                            ) and new_asset != (job.asset or "").upper()
                            to_amount_api = late_bridge_info.get("to_amount")
                            from_amount_api = late_bridge_info.get("from_amount")
                            if to_amount_api is not None:
                                bridge_amount = self._normalize_amount(
                                    to_amount_api, dst_chain_norm, new_asset,
                                )
                            elif not asset_changed:
                                raw = (
                                    from_amount_api if from_amount_api is not None
                                    else (job.incoming_amount or 0.0)
                                )
                                bridge_amount = self._normalize_amount(
                                    raw, dst_chain_norm, new_asset,
                                )
                            else:
                                # Asset swapped and no destination amount
                                # provided — fall back to the source
                                # incoming amount as a coarse approximation;
                                # the destination-side HopJob will refine
                                # via its own outflow walk.
                                bridge_amount = float(job.incoming_amount or 0.0)

                            dst_tx_hash = late_bridge_info.get("dst_tx_hash")
                            dst_block_time = late_bridge_info.get("dst_block_time")
                            synthetic_dst_hash = dst_tx_hash or f"bridge-{uuid.uuid4().hex[:16]}"

                            _ensure_entity(
                                dst_address, "intermediate", 0.0,
                                notes=f"{brand_label} cross-chain destination",
                            )

                            step_index_b = len(paths[job.path_id]["steps"])
                            bridge_attr = (
                                bridge_amount if asset_changed
                                else fifo_ledger.attribute_outflow(
                                    job.current_address, bridge_amount,
                                )
                            )
                            fifo_ledger.record_inflow(
                                dst_address, bridge_amount, bridge_attr,
                            )
                            _add_step(job.path_id, {
                                "step_index": step_index_b,
                                "from": job.current_address,
                                "to": dst_address,
                                "tx_hash": synthetic_dst_hash,
                                "chain": dst_chain_norm,
                                "asset": new_asset,
                                "amount_estimate": bridge_amount,
                                "attributed_amount": bridge_attr,
                                "time": dst_block_time,
                                "direction": "out",
                                "step_type": "bridge_transfer",
                                "service_label": brand_label,
                                "protocol": late_bridge_info.get("protocol"),
                                "reasoning": (
                                    f"{brand_label} bridge deposit detected; "
                                    f"continuing on {dst_chain_norm}."
                                ),
                                "llm_decisions": (
                                    [classifier_decision.model_dump()]
                                    if classifier_decision else []
                                ),
                            })

                            new_token_id = int(
                                late_bridge_info.get("dst_token_id") or 0
                            )
                            hop_scheduler.push(HopJob(
                                path_id=job.path_id,
                                current_address=dst_address,
                                incoming_tx_hash=synthetic_dst_hash,
                                incoming_amount=bridge_amount,
                                incoming_time=(
                                    int(dst_block_time)
                                    if isinstance(dst_block_time, (int, float))
                                    else job.incoming_time
                                ),
                                chain=dst_chain_norm,
                                asset=new_asset,
                                token_id=new_token_id,
                                hop_index=job.hop_index + 1,
                                attributed_amount=bridge_attr,
                            ))
                            return

                        # Bridge confirmed but cross-chain destination
                        # unresolved — annotate and stop the path so
                        # we don't trace the bridge's internal ledger.
                        annotations.append({
                            "id": f"ann-{len(annotations)+1}",
                            "label": f"{brand_label} - Destination Unknown",
                            "related_addresses": [job.current_address],
                            "related_steps": [
                                f"{job.path_id}:{max(len(paths[job.path_id]['steps'])-1, 0)}"
                            ],
                            "text": (
                                f"{brand_label} bridge deposit detected at "
                                f"{self._format_address(job.current_address)}; "
                                f"outflows aggregate into the {brand_label} "
                                f"treasury but bridge_analyze did not return a "
                                f"cross-chain destination. Manual investigation "
                                f"of the destination chain is needed."
                            ),
                        })
                        paths[job.path_id]["stop_reason"] = (
                            f"{brand_label} bridge deposit — destination unknown"
                        )
                        fifo_ledger.claim_terminal(job.attributed_amount)
                        return

                    # ``is_bridge=false`` on the funding tx → this is
                    # NOT a per-swap deposit, just an ordinary wallet
                    # whose outflow happens to land on a bridge brand.
                    # Fall through to normal outflow processing; the
                    # existing handler will catch the bridge contract
                    # at the next hop.
                    logger.info(
                        "Bridge-deposit signal at %s (hop %d) was not confirmed by "
                        "bridge_analyze (is_bridge=false on incoming tx) — proceeding "
                        "with normal outflow processing; the existing bridge handler "
                        "will catch %r contract on the next hop.",
                        self._format_address(job.current_address), job.hop_index,
                        brand_label,
                    )

            # --- OTC-like behavioral analysis (deferred, reuses data_list) ---
            if _run_otc:
                cex_threshold = payload.get("cex_single_cluster_threshold", 0.60)
                try:
                    withdraw_txs = data_list
                    in_txs_result = await _call_tool("all_txs", {
                        "address": job.current_address,
                        "blockchain_name": job.chain,
                        "filter": {"token_id": [job.token_id]} if job.token_id else None,
                        "limit": 50, "offset": 0,
                        "direction": "asc", "order": "time",
                        "transaction_type": "deposit",
                    })
                    in_data = in_txs_result.get("data", []) if isinstance(in_txs_result, dict) else []

                    total_in_volume = 0.0
                    counterparties: set = set()
                    earliest_time = None
                    for itx in in_data:
                        amt = self._normalize_amount(itx.get("amount_coerced") or itx.get("amount") or 0, job.chain, job.asset)
                        total_in_volume += amt
                        sender = itx.get("from") or itx.get("address")
                        if sender:
                            counterparties.add(sender)
                        bt = itx.get("block_time")
                        if bt and (earliest_time is None or int(bt) < earliest_time):
                            earliest_time = int(bt)

                    now_ts = int(time.time())
                    address_age_days = (now_ts - earliest_time) // 86400 if earliest_time else 0
                    tx_count = len(in_data) + len(withdraw_txs)

                    outbound_distribution: dict[str, float] = {}
                    for wtx in withdraw_txs[:50]:
                        w_hash = wtx.get("hash") or wtx.get("tx_hash")
                        w_amt = self._normalize_amount(wtx.get("amount_coerced") or wtx.get("amount") or 0, job.chain, job.asset)
                        if w_hash and w_hash in owner_hints:
                            dest_label = str(owner_hints[w_hash])
                        else:
                            dest_label = wtx.get("to") or "unknown"
                        outbound_distribution[dest_label] = outbound_distribution.get(dest_label, 0.0) + w_amt

                    otc_result = self._classify_otc_like(
                        total_in_volume=total_in_volume,
                        tx_count=tx_count,
                        counterparty_count=len(counterparties),
                        address_age_days=address_age_days,
                        outbound_distribution=outbound_distribution,
                    )

                    if otc_result.get("otc_like"):
                        role = "otc_service"
                        terminal = False
                        if "Potential Service / OTC-like Entity" not in labels:
                            labels.append("Potential Service / OTC-like Entity")
                        _ensure_entity(job.current_address, role, risk_score, labels=labels,
                                       notes=f"OTC-like profile: vol=${total_in_volume:,.0f}, {tx_count} txs, {address_age_days}d old")
                        annotations.append({
                            "id": f"ann-{len(annotations)+1}",
                            "label": "Potential Service / OTC-like Entity",
                            "related_addresses": [job.current_address],
                            "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                            "text": (
                                f"Address resembles an unidentified service (OTC-like). "
                                f"Volume: ${total_in_volume:,.0f}, Txs: {tx_count}, "
                                f"Counterparties: {len(counterparties)}, Age: {address_age_days}d."
                            ),
                        })
                        annotations.append({
                            "id": f"ann-{len(annotations)+1}",
                            "label": "Ownership Change Risk",
                            "related_addresses": [job.current_address],
                            "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                            "text": "Ownership change risk: funds may have changed hands at this service-like entity.",
                        })

                        dom_share = otc_result.get("dominant_cex_share", 0.0)
                        dom_cex = otc_result.get("dominant_cex")
                        if dom_share >= cex_threshold and dom_cex:
                            annotations.append({
                                "id": f"ann-{len(annotations)+1}",
                                "label": "CEX Concentration",
                                "related_addresses": [job.current_address],
                                "related_steps": [f"{job.path_id}:{len(paths[job.path_id]['steps'])-1}"],
                                "text": (
                                    f"CEX concentration: {dom_share*100:.1f}% of outflows go to {dom_cex}. "
                                    f"Tracing through to CEX endpoint."
                                ),
                            })
                except Exception as exc:
                    logger.warning(f"OTC-like analysis failed for {job.current_address}: {exc}")

            selector_result = None
            hop_selector_decision: DecisionRef | None = None
            selected_hashes = self._accumulate_hashes(
                data_list, job.incoming_amount, job.chain, asset=job.asset,
                hop_index=job.hop_index,
            )
            used_accumulation = bool(selected_hashes)
            if not selected_hashes and data_list:
                selector_context = {
                    "chain": job.chain,
                    "asset": job.asset,
                    "incoming_amount": job.incoming_amount,
                    "incoming_time": job.incoming_time,
                    "txs": data_list,
                }
                selector_result, hop_selector_decision = await self._run_selector(selector_context)
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
            # Track per-recipient state so that multiple outgoing txs from the
            # same source to the same destination (common for repeated CEX
            # deposits, staged payouts, etc.) are ALL captured as separate
            # edges/steps in the graph, while still queueing only a single
            # downstream HopJob per unique recipient. The queued HopJob's
            # incoming_amount is aggregated across repeats so the recipient's
            # outflow accumulation targets the true aggregate inflow.
            recipient_state: dict[str, dict[str, Any]] = {}
            used_base_path = False
            # Snapshot the base path's step history BEFORE we touch it in
            # this hop. When multiple outgoing txs are selected, the first
            # one writes into ``base_path_id`` and subsequent ones fork
            # via ``_copy_path`` — but a naive copy-at-fork-time inherits
            # the first step too, so a non-dust sibling would drag a
            # dust step into its rendered history and visualization would
            # render the dust edge via the surviving path. Forking from
            # this frozen snapshot keeps each branch clean.
            base_steps_snapshot = [dict(s) for s in paths[base_path_id]["steps"]]
            base_description = paths[base_path_id].get("description")
            # Pre-resolve all transfers for this hop's selected hashes in
            # parallel. Each _resolve_transfer is a pure read (1-2 HTTP calls)
            # with no shared-state mutation, so gathering them is safe. The
            # mutation loop that follows stays sequential so FIFO attribution
            # and path_counter bookkeeping remain deterministic.
            _hop_prefetched = await asyncio.gather(*(
                _resolve_transfer(
                    sel_hash, job.chain,
                    address_hint=job.current_address,
                    token_id_val=job.token_id,
                    expected_from=job.current_address,
                    asset_hint=job.asset,
                )
                for sel_hash in selected_hashes
            )) if selected_hashes else []
            for _idx, (sel_hash, transfer) in enumerate(zip(selected_hashes, _hop_prefetched, strict=True)):
                if not transfer or not transfer.get("to"):
                    continue
                to_addr = transfer["to"]
                edge_key = (sel_hash, job.current_address, to_addr)
                if edge_key in path_seen_hashes.get(job.path_id, set()):
                    continue
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

                step_amount = float(amount or 0.0)
                raw_attributed = fifo_ledger.attribute_outflow(job.current_address, step_amount)
                fifo_ledger.record_inflow(to_addr, step_amount, raw_attributed)

                existing_state = recipient_state.get(to_addr)
                if existing_state is not None:
                    # Repeated send to a recipient we've already recorded this
                    # hop — reuse its path so repeated txs stack as sibling
                    # steps on the same branch instead of forking.
                    path_id = existing_state["path_id"]
                elif not used_base_path:
                    path_id = base_path_id
                    used_base_path = True
                else:
                    path_counter += 1
                    path_id = str(path_counter)
                    # Fork from the FROZEN snapshot, not from the live
                    # ``base_path_id`` whose steps now include the first
                    # iteration's (possibly dust-trimmed) addition.
                    paths[path_id] = {
                        "path_id": path_id,
                        "description": base_description,
                        "steps": [dict(s) for s in base_steps_snapshot],
                        "stop_reason": None,
                    }

                step_index = len(paths[path_id]["steps"])
                _step_decisions = [
                    d.model_dump()
                    for d in (classifier_decision, hop_selector_decision)
                    if d is not None
                ]
                _add_step(path_id, {
                    "step_index": step_index,
                    "from": job.current_address,
                    "to": to_addr,
                    "tx_hash": sel_hash,
                    "chain": job.chain,
                    "asset": job.asset,
                    "amount_estimate": step_amount,
                    "attributed_amount": raw_attributed,
                    "time": block_time,
                    "direction": "out",
                    "step_type": "direct_transfer",
                    "service_label": classification.get("service_label"),
                    "protocol": classification.get("protocol"),
                    "reasoning": (selector_result or {}).get("reasoning") or "Selected by hop selector.",
                    "llm_decisions": _step_decisions,
                })

                self._collect_token_transfer_data(
                    json.dumps({"data": [{"input": {"address": transfer.get("from")}, "output": {"address": transfer.get("to"), "owner": transfer.get("output_owner")}, "amount": transfer.get("amount", 0), "block_time": transfer.get("block_time"), "token_id": transfer.get("token_id", 0)}]}, ensure_ascii=False),
                    {"tx_hash": sel_hash, "blockchain_name": job.chain},
                    all_txs_map,
                    risk_map,
                    txs_collected,
                    tx_list_collected,
                    txs_seen,
                )

                # Dust guard: the per-step filter is about OUTFLOW size
                # relative to the stolen amount (operator rule: "don't
                # chase anything under 1% of stolen funds"). We compare
                # ``step_amount`` directly, NOT ``raw_attributed``.
                # Attribution is FIFO-diluted — when an address holds a
                # mix of stolen + non-stolen inflows, its theft-share
                # ratio is <100% and the FIFO-returned attribution for a
                # legit-sized outflow can dip below the stolen*ratio
                # threshold even when the outflow itself is plainly
                # above it (e.g. 663 USDT out of a 60k stolen pool =
                # 1.1%, but FIFO attributes only 230 because 208k
                # non-theft dollars sat in the queue, diluting the
                # theft-share ratio). Dust-trimming those drops the
                # hop unfairly and cuts the trace early.
                dust_hit = (
                    stolen_amount > 0
                    and min_attribution_ratio > 0.0
                    and step_amount < stolen_amount * min_attribution_ratio
                )

                if dust_hit:
                    dust_pct = (raw_attributed / stolen_amount) * 100.0 if stolen_amount else 0.0
                    # Dust-aggregate onto an already-active path must NOT
                    # mark that path "completed". When ``existing_state`` is
                    # set the path_id was taken from a prior non-dust
                    # sibling that pushed a real HopJob — the path still has
                    # live downstream work. Setting ``stop_reason`` here
                    # (a) inflates ``_completed_paths_count()`` and causes
                    # the scheduler to exit before popping the pending
                    # HopJob, and (b) misleads downstream renderers into
                    # thinking the leg terminated at dust when it actually
                    # continued. Record the dust annotation only.
                    is_aggregate_onto_alive = existing_state is not None
                    if not is_aggregate_onto_alive:
                        paths[path_id]["stop_reason"] = (
                            f"Below dust threshold ({dust_pct:.2f}% of stolen amount)"
                        )
                    # Only mark the recipient as seen on FORKED paths — not on
                    # the incoming (base) path itself. When the first sibling
                    # iteration reuses ``base_path_id`` (``used_base_path`` is
                    # flipped only after that first iter), a naive
                    # ``path_seen_addresses[path_id].add(to_addr)`` pollutes
                    # the parent path's history. The loop-detection check at
                    # the top of the loop (``to_addr in
                    # path_seen_addresses[job.path_id]``) then blocks every
                    # LATER sibling from going to the same recipient —
                    # including the large legitimate outflow the dust
                    # iteration happened to land on first (e.g. a 10 USDT
                    # dust → TN6c followed by a 220 100 USDT real leg to
                    # TN6c would silently drop the 220k).
                    if path_id != job.path_id and not is_aggregate_onto_alive:
                        path_seen_addresses.setdefault(path_id, set()).add(to_addr)
                    if path_id not in dust_trimmed_paths and not is_aggregate_onto_alive:
                        annotations.append({
                            "id": f"ann-{len(annotations)+1}",
                            "label": "Dust Trimmed",
                            "related_addresses": [to_addr],
                            "related_steps": [f"{path_id}:{step_index}"],
                            "text": (
                                f"Branch trimmed at {self._format_address(to_addr)}: "
                                f"attributed {raw_attributed:.2f} < "
                                f"{min_attribution_ratio*100:.2f}% of stolen {stolen_amount:.2f}"
                            ),
                        })
                        dust_trimmed_paths.add(path_id)
                    took_step = True
                elif existing_state is None:
                    new_job = HopJob(
                        path_id=path_id,
                        current_address=to_addr,
                        incoming_tx_hash=sel_hash,
                        incoming_amount=step_amount,
                        incoming_time=block_time,
                        chain=job.chain,
                        asset=job.asset,
                        token_id=int(token_id or 0),
                        hop_index=job.hop_index + 1,
                        attributed_amount=raw_attributed,
                    )
                    hop_scheduler.push(new_job)
                    recipient_state[to_addr] = {
                        "hop_job": new_job,
                        "path_id": path_id,
                        "total_amount": step_amount,
                        "total_attributed": raw_attributed,
                        "earliest_time": block_time,
                    }
                else:
                    # Aggregate repeat send into the queued HopJob so the
                    # recipient's outflow accumulation (driven by
                    # incoming_amount) reflects the real total inflow.
                    existing_state["total_amount"] += step_amount
                    existing_state["total_attributed"] += raw_attributed
                    hop_job = existing_state["hop_job"]
                    hop_job.incoming_amount = existing_state["total_amount"]
                    hop_job.attributed_amount = existing_state["total_attributed"]
                    if block_time is not None:
                        try:
                            bt_int = int(block_time)
                            cur = existing_state["earliest_time"]
                            cur_int = int(cur) if cur is not None else None
                            if cur_int is None or bt_int < cur_int:
                                existing_state["earliest_time"] = block_time
                                hop_job.incoming_time = block_time
                        except (TypeError, ValueError):
                            pass
                took_step = True

            if not took_step:
                fifo_ledger.claim_terminal(job.attributed_amount)
                paths[job.path_id]["stop_reason"] = "Loop detected - no new transactions"
                logger.info("Path %s stopped at %s (hop %d): no new transactions after selection", job.path_id, self._format_address(job.current_address), job.hop_index)

        # Parallel Phase 1 AND Phase 2 across survivors of the current batch.
        # Phase 2 mutates shared state (path_counter, fifo_ledger, paths dict)
        # but all such mutations happen in synchronous code blocks without
        # awaits between them, which makes them atomic under the asyncio GIL
        # contract — concurrent tasks only interleave at await points, and the
        # order-sensitive pairs (attribute_outflow → record_inflow, path_counter
        # increment → _copy_path) are each a single sync block.
        # Default is ON; set AGENT_PARALLEL_HOPS=0 to opt out.
        _parallel_hops = os.getenv("AGENT_PARALLEL_HOPS", "1").lower() in ("1", "true", "yes")
        _hop_fanout = max(1, min(16, int(os.getenv("AGENT_HOP_FANOUT", "5"))))

        while hop_scheduler.should_continue(_completed_paths_count()):
            _batch_size = min(_hop_fanout, len(hop_scheduler)) if _parallel_hops else 1
            job_batch = [hop_scheduler.pop() for _ in range(_batch_size)]
            survivors: list[HopJob] = []
            for job in job_batch:
                if fifo_ledger.cap_exceeded and not cap_annotation_emitted:
                    cap_annotation_emitted = True
                    annotations.append({
                        "id": f"ann-{len(annotations)+1}",
                        "label": "Cap Reached",
                        "related_addresses": [job.current_address],
                        "related_steps": [f"{job.path_id}:{max(len(paths[job.path_id]['steps'])-1, 0)}"],
                        "text": (
                            f"Total traced amount ({fifo_ledger.total_traced:,.2f}) reached the cap "
                            f"({fifo_ledger.cap:,.2f} = stolen {fifo_ledger.stolen_amount:,.2f} + "
                            f"{fifo_ledger.tolerance*100:.1f}% tolerance). Further hops are classified "
                            f"for labeling but not attributed."
                        ),
                    })
                    logger.info(
                        "FIFO cap reached at %s (hop %d, path %s); continuing traversal with zero-attribution terminals",
                        self._format_address(job.current_address), job.hop_index, job.path_id,
                    )

                if job.hop_index > max_hops:
                    fifo_ledger.claim_terminal(job.attributed_amount)
                    if paths[job.path_id]["stop_reason"] is None:
                        paths[job.path_id]["stop_reason"] = "Max hop limit reached"
                    logger.info("Path %s stopped at %s (hop %d): max hops", job.path_id, self._format_address(job.current_address), job.hop_index)
                    continue

                if job.current_address in path_seen_addresses.get(job.path_id, set()):
                    fifo_ledger.claim_terminal(job.attributed_amount)
                    paths[job.path_id]["stop_reason"] = "Loop detected - address revisited"
                    logger.info("Path %s stopped at %s (hop %d): loop detected", job.path_id, self._format_address(job.current_address), job.hop_index)
                    continue

                incoming_ref = float(job.incoming_amount or 0.0)
                dust_threshold = incoming_ref * 0.001 if incoming_ref > 0 else 0.0
                if dust_threshold > 0 and job.attributed_amount < dust_threshold:
                    fifo_ledger.claim_terminal(job.attributed_amount)
                    if paths[job.path_id]["stop_reason"] is None:
                        paths[job.path_id]["stop_reason"] = "Dust amount — below 0.1% of hop inflow"
                    logger.info(
                        "Path %s stopped at %s (hop %d): dust (attributed=%.6g, incoming=%.6g)",
                        job.path_id,
                        self._format_address(job.current_address),
                        job.hop_index,
                        job.attributed_amount,
                        incoming_ref,
                    )
                    continue

                path_seen_addresses.setdefault(job.path_id, set()).add(job.current_address)

                if on_progress:
                    await on_progress(f"Analyzing hop {job.hop_index + 1}...")

                survivors.append(job)

            if not survivors:
                continue

            if _parallel_hops and len(survivors) > 1:
                _p1_list = await asyncio.gather(*(_agentic_phase1_for_job(j) for j in survivors))
                await asyncio.gather(*(
                    _agentic_hop_after_phase1(job, _p1[0], _p1[1], _p1[2])
                    for job, _p1 in zip(survivors, _p1_list, strict=True)
                ))
            else:
                _p1_list = [await _agentic_phase1_for_job(j) for j in survivors]
                for job, _p1 in zip(survivors, _p1_list, strict=True):
                    _ga, _ge, _ebr = _p1
                    await _agentic_hop_after_phase1(job, _ga, _ge, _ebr)

        if hop_scheduler.exhausted and len(hop_scheduler) > 0:
            logger.warning(
                "HopScheduler hit iteration safety net (%d iters, %d jobs still queued). "
                "Consider raising max_paths / max_hops for this case.",
                hop_scheduler.iterations, len(hop_scheduler),
            )

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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("TXS_ARRAY=%s", json.dumps(self.last_txs, ensure_ascii=False))
            logger.debug("TXLIST_ARRAY=%s", json.dumps(self.last_tx_list, ensure_ascii=False))

        return {
            "case_meta": case_meta,
            "paths": list(paths.values()),
            "entities": list(entities.values()),
            "annotations": annotations,
            "trace_stats": {
                "initial_amount_estimate": initial_amount,
                "explored_paths": len(paths),
                "terminated_reason": "All paths reached terminal entities or dead ends",
                "total_traced_amount": fifo_ledger.total_traced if fifo_ledger.stolen_amount > 0 else None,
                "stolen_amount": fifo_ledger.stolen_amount if fifo_ledger.stolen_amount > 0 else None,
                "fifo_audit_log": fifo_ledger.audit_log,
            },
        }

    def _collect_token_transfer_data(
        self, tool_result: str, arguments: dict[str, Any],
        all_txs_map: dict, risk_map: dict,
        txs_collected: list, tx_list_collected: list, txs_seen: set
    ):
        """Helper to collect token transfer data for visualization."""
        try:
            if len(tx_list_collected) >= MAX_TX_LIST:
                return
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

                is_utxo = chain in self._SATOSHI_CHAINS
                tx_type = "tx" if is_utxo else "txEth"
                tx_path = None if is_utxo else "0"

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
                        "path": tx_path,
                        "type": tx_type
                    })

                if tx_hash and from_addr and to_addr:
                    riskscore_from = risk_map.get(from_addr, 0.0)
                    riskscore_to = risk_map.get(to_addr, 0.0)
                    if riskscore_from == 0.0 and isinstance(input_data, dict):
                        riskscore_from = float(input_data.get("riskscore") or 0.0)
                    if riskscore_to == 0.0 and isinstance(output_data, dict):
                        riskscore_to = float(output_data.get("riskscore") or 0.0)

                    output_owner = output_data.get("owner") if isinstance(output_data, dict) else None
                    input_owner = input_data.get("owner") if isinstance(input_data, dict) else None

                    amount_val = float(amount_raw) if amount_raw is not None else 0.0
                    if amount_val == 0.0 and transfer.get("amount_coerced") is not None and chain == "trx":
                        amount_val = float(transfer.get("amount_coerced") or 0.0) * 1e6

                    fiat_rate = transfer.get("fiat_rate") or transfer.get("fiatRate") or 1.0

                    out_entry: dict[str, Any] = {"address": to_addr, "riskscore": riskscore_to}
                    if output_owner:
                        out_entry["owner"] = output_owner
                    inp_entry: dict[str, Any] = {"address": from_addr, "riskscore": riskscore_from}
                    if input_owner:
                        inp_entry["owner"] = input_owner

                    tx_list_collected.append({
                        "inputs": [inp_entry],
                        "outputs": [out_entry],
                        "hash": tx_hash,
                        "fiatRate": fiat_rate,
                        "addressesCount": 2,
                        "amount": amount_val,
                        "currency": chain,
                        "tokenId": token_id,
                        "poolTime": block_time,
                        "date": block_time,
                        "path": tx_path,
                        "type": tx_type
                    })
        except Exception:
            pass

    def _collect_utxo_tx_data(
        self,
        tx_hash: str,
        chain: str,
        utxo_outputs: list[dict[str, Any]],
        risk_map: dict,
        txs_collected: list,
        tx_list_collected: list,
        txs_seen: set,
    ):
        """Collect full UTXO transaction data for visualization.

        Creates a single txs entry and a single txList entry that contain
        ALL inputs and outputs of the UTXO transaction.
        """
        if not utxo_outputs or tx_hash in txs_seen:
            return
        if len(tx_list_collected) >= MAX_TX_LIST:
            return

        first = utxo_outputs[0]
        from_addr = first.get("from")
        block_time = first.get("block_time") or 0
        token_id = first.get("token_id") or 0
        fiat_rate = first.get("_fiat_rate") or first.get("fiat_rate") or 1.0
        raw_inputs = first.get("_raw_inputs", [])
        raw_outputs = first.get("_raw_outputs", [])
        total_in = first.get("_total_in") or sum(float(o.get("amount", 0)) for o in utxo_outputs)

        idx = len(txs_collected)
        txs_seen.add(tx_hash)

        txs_collected.append({
            "currency": chain,
            "descriptor": f"{tx_hash}-{chain}-{token_id}-null",
            "hash": tx_hash,
            "token_id": token_id,
            "x": 100 + idx * 40,
            "y": 100 + idx * 40,
            "color": "#D15AE4",
            "path": None,
            "type": "tx"
        })

        # Build full inputs array
        inputs_list = []
        if isinstance(raw_inputs, list):
            for inp in raw_inputs:
                if not isinstance(inp, dict):
                    continue
                addr = inp.get("address")
                if addr:
                    rs = risk_map.get(addr, float(inp.get("riskscore") or 0.0))
                    inputs_list.append({
                        "address": addr,
                        "amount": inp.get("amount"),
                        "owner": inp.get("owner"),
                        "riskscore": rs,
                    })
        if not inputs_list and from_addr:
            inputs_list = [{"address": from_addr, "riskscore": risk_map.get(from_addr, 0.0)}]

        # Build full outputs array
        outputs_list = []
        if isinstance(raw_outputs, list):
            for out in raw_outputs:
                if not isinstance(out, dict):
                    continue
                addr = out.get("address")
                if addr:
                    rs = risk_map.get(addr, float(out.get("riskscore") or 0.0))
                    outputs_list.append({
                        "address": addr,
                        "amount": out.get("amount"),
                        "owner": out.get("owner"),
                        "pos": out.get("pos"),
                        "riskscore": rs,
                        "next": out.get("next"),
                    })
        if not outputs_list:
            for uout in utxo_outputs:
                to_addr = uout.get("to")
                if to_addr:
                    outputs_list.append({
                        "address": to_addr,
                        "amount": uout.get("amount"),
                        "riskscore": risk_map.get(to_addr, float(uout.get("output_riskscore") or 0.0)),
                    })

        unique_addrs = set()
        for inp in inputs_list:
            if inp.get("address"):
                unique_addrs.add(inp["address"])
        for out in outputs_list:
            if out.get("address"):
                unique_addrs.add(out["address"])

        tx_list_collected.append({
            "inputs": inputs_list,
            "outputs": outputs_list,
            "hash": tx_hash,
            "fiatRate": fiat_rate,
            "addressesCount": len(unique_addrs),
            "amount": total_in,
            "fee": first.get("_fee"),
            "currency": chain,
            "tokenId": token_id,
            "poolTime": block_time,
            "date": block_time,
            "path": None,
            "type": "tx"
        })

    @staticmethod
    def _cap_visualization_tx_lists(
        tx_list: list[dict[str, Any]] | None,
        txs: list[dict[str, Any]] | None,
        max_items: int = MAX_TX_LIST,
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        """Shrink tx_list/txs before building the visualization payload (MCP POST size)."""
        if not tx_list or len(tx_list) <= max_items:
            return tx_list, txs
        ordered = sorted(
            tx_list,
            key=lambda e: int(e.get("date") or e.get("poolTime") or 0),
        )
        capped_list = ordered[-max_items:]
        keep_hashes = {e.get("hash") for e in capped_list if e.get("hash")}
        txs_out = [t for t in (txs or []) if t.get("hash") in keep_hashes][:max_items] if txs else txs
        return capped_list, txs_out

    # ─── Main trace entry point ───────────────────────────────────────────────

    async def trace(
        self, config: TracerConfig,
        on_progress: Callable[[str], Awaitable[None]] | None = None
    ) -> TraceResult:
        """Run a trace."""
        case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_id = f"trace-{uuid.uuid4().hex[:12]}"

        if on_progress:
            await on_progress("Analyzing transaction context...")

        # Normalize chain early so all downstream calls use the SAILS currency code.
        config.blockchain_name = self._normalize_chain(config.blockchain_name)

        # Extract victim from tx_hash when possible
        token_id_hint: int | None = None
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
            token_id=token_id_hint,
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
                "stolen_amount": config.stolen_amount,
                "traced_amount_tolerance": config.traced_amount_tolerance,
                "min_path_attribution_ratio": config.min_path_attribution_ratio,
            },
            "cex_single_cluster_threshold": config.cex_single_cluster_threshold,
            "stolen_amount": config.stolen_amount or 0.0,
            "traced_amount_tolerance": config.traced_amount_tolerance,
            "min_path_attribution_ratio": config.min_path_attribution_ratio,
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
            address_info = getattr(self, "last_address_info", None)
            tx_list, txs = self._cap_visualization_tx_lists(
                tx_list if isinstance(tx_list, list) else None,
                txs if isinstance(txs, list) else None,
            )
            viz_payload = generate_visualization_payload(trace_result, tx_list=tx_list, txs=txs, address_info=address_info)
        except Exception as exc:
            logger.warning(f"⚠️ Visualization payload generation failed: {exc}")
            return

        client = self._get_client()
        save_fn = getattr(client, "save_and_share_visualization", None)
        if not callable(save_fn):
            logger.warning("⚠️ Visualization save/share not supported by client.")
            return

        save_input = {
            "title": viz_payload.get("title"),
            "type": viz_payload.get("type", "address"),
            "payload": viz_payload.get("payload", {}),
            "helpers": viz_payload.get("helpers", {}),
            "extras": viz_payload.get("extras", {}),
        }
        try:
            span_input = json.dumps({
                "title": save_input.get("title"),
                "type": save_input.get("type"),
                "has_payload": bool(save_input.get("payload")),
            }, ensure_ascii=False)
            span_input = span_input[:2000] if len(span_input) > 2000 else span_input
            with function_span("save_visualization", input=span_input) as tool_span:
                try:
                    result = await asyncio.wait_for(save_fn(save_input), timeout=30.0)
                    try:
                        if hasattr(tool_span, "span_data"):
                            tool_span.span_data.output = self._compact_tool_result("save-visualization", result)
                    except Exception:
                        pass
                except Exception as exc:
                    try:
                        tool_span.set_error({"message": str(exc), "data": {"tool": "save_visualization"}})
                        if hasattr(tool_span, "span_data"):
                            tool_span.span_data.output = {"error": str(exc)}
                    except Exception:
                        pass
                    raise
        except Exception as exc:
            logger.warning(
                "⚠️ Visualization save/share failed: %s: %s; title=%s, type=%s",
                type(exc).__name__, exc,
                save_input.get("title"), save_input.get("type"),
            )
            return

        logger.info("Visualization save/share returned result type=%s", type(result).__name__)

        def _deep_find(obj: Any, keys: set) -> str | None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in keys and isinstance(v, str) and v:
                        return v
                    found = _deep_find(v, keys)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = _deep_find(item, keys)
                    if found:
                        return found
            return None

        share_url = None
        hash_value = None
        if isinstance(result, dict):
            share_url = (
                result.get("share_url") or
                result.get("shareUrl") or
                result.get("shareURL")
            )
            if not share_url:
                share_obj = result.get("share_result") if isinstance(result.get("share_result"), dict) else {}
                share_url = share_obj.get("share_url") or share_obj.get("shareUrl") or share_obj.get("url")
            if not share_url:
                share_url = _deep_find(result, {"share_url", "shareUrl", "shareURL", "share_link", "shareLink"})
            if not share_url:
                candidate = _deep_find(result, {"url"})
                if candidate and "/api/" not in candidate:
                    share_url = candidate

            if not hash_value:
                hash_value = (
                    result.get("hash") or
                    result.get("id") or
                    result.get("_id") or
                    result.get("data", {}).get("hash") or
                    result.get("data", {}).get("id") or
                    result.get("data", {}).get("_id") or
                    result.get("data", {}).get("payload", {}).get("hash")
                )
            if not hash_value:
                hash_value = _deep_find(result, {"hash", "id", "_id"})

        if not share_url and hash_value:
            base_url = (
                os.getenv("VISUALIZATION_BASE_URL") or
                os.getenv("NEXT_PUBLIC_WEB_URL") or
                os.getenv("NEXT_PUBLIC_APP_URL") or
                os.getenv("NEXT_PUBLIC_API_URL") or
                ""
            ).rstrip("/")
            template = os.getenv("VISUALIZATION_URL_TEMPLATE", "")
            if template:
                try:
                    share_url = template.format(base=base_url, hash=hash_value)
                except Exception:
                    pass
            if not share_url:
                if base_url:
                    share_url = f"{base_url}/ai/{hash_value}"
                else:
                    share_url = f"/ai/{hash_value}"

        if share_url:
            trace_result.visualization_url = share_url
            logger.info(f"✅ Visualization saved/shared: {share_url}")
        else:
            preview = ""
            try:
                preview = json.dumps(result, ensure_ascii=False)[:800]
            except Exception:
                preview = str(result)[:800]
            logger.warning(
                "⚠️ Visualization saved but no URL extracted (share_url=%s, hash=%s). result_preview=%s",
                share_url, hash_value, preview,
            )

    @abstractmethod
    def _get_client(self):
        """Return the underlying client for theft_detection helpers."""
        pass

    @staticmethod
    @lru_cache(maxsize=1)
    def _chain_alias_map() -> dict[str, str]:
        path = Path(__file__).resolve().parents[1] / "currencies.json"
        mapping: dict[str, str] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = []
        for item in data if isinstance(data, list) else []:
            if item.get("token_id") != 0:
                continue
            if item.get("issuer") is not None:
                continue
            currency = item.get("currency")
            if not currency:
                continue
            currency = str(currency).lower()
            # Skip testnet currencies — they share symbols with mainnets
            # (e.g. btc_testnet has symbol "btc") and would overwrite them.
            if "testnet" in currency:
                continue
            mapping[currency] = currency
            symbol = item.get("symbol")
            if symbol:
                mapping[str(symbol).lower()] = currency
            name = item.get("name")
            if name:
                mapping[str(name).lower()] = currency
        return mapping

    @staticmethod
    def _normalize_chain(chain: str | None) -> str:
        c = (chain or "").strip().lower()
        if not c:
            return c
        manual = {
            # ``tron-mainnet`` is what we send to bridge-detector; it
            # may echo the same string back in ``dst_chain``, so map
            # both the short and the network-id form to ``trx``.
            "tron": "trx",
            "tron-mainnet": "trx",
            "trc": "trx",
            "trc20": "trx",
            "trx": "trx",
            "ethereum": "eth",
            "eth": "eth",
            "binance": "bsc",
            "bsc": "bsc",
            "bnb": "bsc",
            "bep20": "bsc",
            "poly": "matic",
            "polygon": "matic",
            "matic": "matic",
        }
        if c in manual:
            return manual[c]
        alias_map = BaseTracer._chain_alias_map()
        return alias_map.get(c, c)
