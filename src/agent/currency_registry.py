"""Currency metadata: authoritative decimals + chain-name mapping.

The AML API exposes ``/api/pro/currencies`` which returns ~12k records
(every chain × token it knows). We ship a seeded snapshot at
``src/currencies.json`` and load it into two hash-tables on first
access:

* ``(chain, token_id)`` → :class:`Currency`
* ``(chain, symbol_upper)`` → :class:`Currency`

Both are plain dicts for O(1) lookup. Callers use ``get_registry()`` and
treat the result as read-only.

``EXTERNAL_CHAIN_MAP`` normalizes chain-name strings that arrive in
bridge responses (thorchain: ``"bitcoin"``, ``"ethereum"``) into our
internal chain codes (``"btc"``, ``"eth"``).

The seeded file is the Phase 1 source. Phase 2 (separate ticket) will
replace ``_load_from_file`` with a periodic API fetch; the public
interface stays the same.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Currency:
    chain: str
    token_id: int
    symbol: str  # upper-cased
    unit: int  # decimals
    name: str = ""
    issuer: str | None = None


# External-chain name → internal chain code. Used when bridge APIs
# describe the destination chain using a human-readable string rather
# than our internal code. Keys are lower-cased when looked up.
EXTERNAL_CHAIN_MAP: dict[str, str] = {
    # UTXO
    "bitcoin": "btc", "btc": "btc",
    "litecoin": "ltc", "ltc": "ltc",
    "bitcoincash": "bch", "bitcoin_cash": "bch", "bch": "bch",
    "dogecoin": "doge", "doge": "doge",
    # EVM
    "ethereum": "eth", "eth": "eth",
    "binance": "bsc", "binance_smart_chain": "bsc", "bsc": "bsc", "bnb": "bsc",
    "polygon": "matic", "matic": "matic", "pol": "matic",
    "avalanche": "avax", "avax": "avax",
    "arbitrum": "arb", "arb": "arb",
    "optimism": "op", "op": "op",
    "base": "base",
    "fantom": "ftm", "ftm": "ftm",
    "cronos": "cro", "cro": "cro",
    "moonbeam": "glmr", "glmr": "glmr",
    "harmony": "one", "one": "one",
    # Non-EVM
    "tron": "trx", "trx": "trx",
    "solana": "sol", "sol": "sol",
    "cardano": "ada", "ada": "ada",
    "ripple": "xrp", "xrp": "xrp",
    "stellar": "xlm", "xlm": "xlm",
    "polkadot": "dot", "dot": "dot",
    "near": "near",
    "ton": "ton",
    "atom": "atom", "cosmos": "atom",
}


def normalize_external_chain(name: Any) -> str | None:
    """Return internal chain code for a free-form external chain name."""
    if not isinstance(name, str) or not name.strip():
        return None
    return EXTERNAL_CHAIN_MAP.get(name.strip().lower())


def parse_thorchain_token_prefix(token: Any) -> str | None:
    """Thorchain identifies tokens as ``chain.symbol`` (e.g. ``"eth.eth"``,
    ``"btc.btc"``, ``"bsc.bnb"``). Return the internal chain code parsed
    from the prefix, or ``None`` if the input doesn't match."""
    if not isinstance(token, str) or "." not in token:
        return None
    prefix = token.split(".", 1)[0].strip().lower()
    return EXTERNAL_CHAIN_MAP.get(prefix)


class CurrencyRegistry:
    """Read-only view over the currency records.

    Instances are built by ``get_registry()`` and cached for the process
    lifetime. Don't instantiate manually — use the factory so tests
    sharing fixtures get the same table.
    """

    __slots__ = ("_by_id", "_by_symbol")

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self._by_id: dict[tuple[str, int], Currency] = {}
        self._by_symbol: dict[tuple[str, str], Currency] = {}
        for raw in records:
            chain = str(raw.get("currency") or "").strip().lower()
            if not chain:
                continue
            try:
                token_id = int(raw.get("token_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            try:
                unit = int(raw.get("unit", 0) or 0)
            except (TypeError, ValueError):
                unit = 0
            symbol = str(raw.get("symbol") or "").strip().upper()
            currency = Currency(
                chain=chain,
                token_id=token_id,
                symbol=symbol,
                unit=unit,
                name=str(raw.get("name") or ""),
                issuer=(raw.get("issuer") or None),
            )
            # (chain, token_id) — unique in the registry.
            self._by_id.setdefault((chain, token_id), currency)
            # (chain, symbol) — first-seen wins so native tokens (token_id=0)
            # registered first aren't shadowed by later wrapped variants.
            if symbol:
                self._by_symbol.setdefault((chain, symbol), currency)

    def __len__(self) -> int:
        return len(self._by_id)

    def lookup(self, chain: str, token_id: int) -> Currency | None:
        if chain is None:
            return None
        return self._by_id.get((chain.strip().lower(), int(token_id or 0)))

    def lookup_by_symbol(self, chain: str, symbol: str) -> Currency | None:
        if not chain or not symbol:
            return None
        return self._by_symbol.get((chain.strip().lower(), symbol.strip().upper()))

    def native_unit(self, chain: str) -> int | None:
        """Unit (decimals) of the chain's native coin — token_id=0."""
        rec = self.lookup(chain, 0)
        return rec.unit if rec else None


_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "currencies.json"


def _load_from_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("currencies snapshot missing at %s; registry will be empty", path)
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error("currencies snapshot %s is malformed: %s", path, e)
        return []
    if not isinstance(raw, list):
        logger.error("currencies snapshot %s: expected a list, got %s", path, type(raw).__name__)
        return []
    return raw


@lru_cache(maxsize=1)
def get_registry() -> CurrencyRegistry:
    """Return the process-wide registry. First call loads from the
    seeded JSON snapshot; subsequent calls are ~free.

    Phase 2 swaps this body for an API-backed loader. Callers keep
    ``get_registry().lookup(...)`` unchanged.
    """
    records = _load_from_file(_SNAPSHOT_PATH)
    reg = CurrencyRegistry(records)
    logger.info("currency registry loaded: %d records from %s", len(reg), _SNAPSHOT_PATH.name)
    return reg
