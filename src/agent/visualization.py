import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from agent.models import Entity, TraceResult

logger = logging.getLogger(__name__)

VIZ_MODULE_VERSION = "token_id-fix-2"
logger.info(
    "visualization.py loaded — VIZ_MODULE_VERSION=%s "
    "(edge_token_id + descriptor-parse fallback). "
    "If you don't see this line on startup, you're running a stale module.",
    VIZ_MODULE_VERSION,
)


def _token_id_from_descriptor(desc: str | None) -> int | None:
    """Parse the ``token_id`` out of a tx descriptor of the shape
    ``{hash}-{chain}-{token_id}-{idx}`` (the format written by
    ``base_tracer._collect_token_transfer_data``). Returns ``None`` if the
    descriptor does not match, so callers can fall back safely.

    This gives the viz a ground-truth source for the per-tx token that doesn't
    depend on whether ``tx_list``/``txs`` happened to carry ``tokenId``: even
    if both of those arrays get filtered/stripped upstream, the descriptor
    string alone is enough to render the edge correctly (e.g. a USDT hop whose
    descriptor is ``0x5438d5cb…-eth-94252-7`` resolves to ``94252``).
    """
    if not desc or not isinstance(desc, str):
        return None
    parts = desc.split("-")
    if len(parts) < 4:
        return None
    try:
        return int(parts[-2])
    except (TypeError, ValueError):
        return None


def _lookup_currency(chain: str, token_id: int) -> dict[str, Any] | None:
    """Return the currencies.json record for ``(chain, token_id)`` or None.

    Thin adapter over :mod:`agent.currency_registry` — the registry owns
    the hash-table, this function shapes the result as the legacy dict
    callers already expect.
    """
    if not chain:
        return None
    from .currency_registry import get_registry
    rec = get_registry().lookup(chain, token_id)
    if rec is None:
        return None
    return {
        "currency": rec.chain,
        "issuer": rec.issuer,
        "name": rec.name,
        "symbol": rec.symbol,
        "token_id": rec.token_id,
        "unit": rec.unit,
    }


# Fallback tables used when the token isn't in currencies.json. Kept in
# sync with _EVM_NATIVE_CHAINS defaults in base_tracer; BTC-family is 8.
# Registry wins whenever it has an entry, so these only fire for chains
# we haven't yet indexed.
#
# Chain *codes* match the rest of the codebase (``bsc``, not ``bnb``);
# ``bnb`` is the native asset *symbol* on BSC, not a chain code.
_NATIVE_UNIT_MAP = {"btc": 8, "bch": 8, "ltc": 8, "eth": 9, "bsc": 9, "matic": 9}
_TOKEN_UNIT_MAP = {
    "USDT": 6, "USDC": 6, "DAI": 18, "BUSD": 18,
    "WETH": 18, "WBTC": 8, "LINK": 18, "UNI": 18,
    "AAVE": 18, "GRT": 18, "MTL": 8,
}
_CURRENCY_NAMES = {
    "eth": "Ethereum", "btc": "Bitcoin", "trx": "TRON",
    "bsc": "BNB Chain", "matic": "Polygon", "bch": "Bitcoin Cash",
    "ltc": "Litecoin", "sol": "Solana",
}


def _build_currency_info(
    chain: str, token_id: int, asset_hint: str = ""
) -> dict[str, Any]:
    """Produce a currencyInfo entry for ``(chain, token_id)``.

    Prefers the canonical record from ``currencies.json`` so the
    name/symbol/unit/issuer actually match the on-chain token (otherwise a
    USDT transfer inside an "ETH" case would end up labelled as ETH with
    unit 6). Falls back to the hardcoded maps only when the DB doesn't have
    an entry.
    """
    rec = _lookup_currency(chain, token_id)
    if rec:
        symbol = rec.get("symbol") or chain
        # Native asset symbols are emitted lower-case in the platform
        # payload (``"bnb"`` on bsc, ``"trx"`` on trx, ``"eth"`` on eth)
        # while non-native token symbols stay upper-case (``"USDT"``).
        # The currency_registry uppercases everything on load, so undo
        # that here for token_id=0.
        if token_id == 0 and isinstance(symbol, str):
            symbol = symbol.lower()
        return {
            "currency": chain,
            "issuer": rec.get("issuer"),
            "name": rec.get("name") or chain,
            "symbol": symbol,
            "token_id": token_id,
            "unit": rec.get("unit", 6),
        }

    is_native = token_id == 0
    asset_upper = (asset_hint or "").upper()
    if is_native:
        unit = _NATIVE_UNIT_MAP.get(chain, 6)
        name = _CURRENCY_NAMES.get(chain, chain)
        symbol = chain
    else:
        unit = _TOKEN_UNIT_MAP.get(asset_upper, 6)
        if asset_upper == "USDT":
            name, symbol = "Tether USD", "USDT"
        else:
            name = asset_hint or chain
            symbol = asset_upper or chain
    return {
        "currency": chain,
        "issuer": None,
        "name": name,
        "symbol": symbol,
        "token_id": token_id,
        "unit": unit,
    }

def _normalize_chain(chain: str) -> str:
    c = (chain or "").lower()
    if c in {"tron", "trc", "trc20", "trx"}:
        return "trx"
    if c in {"ethereum", "eth"}:
        return "eth"
    # Chain code is "bsc" everywhere else in the codebase
    # (models.py, base_tracer._normalize_chain, currency_registry).
    # ``bnb`` is the native asset *symbol* on BSC, not a chain code —
    # mapping the chain to "bnb" here breaks the (chain, token_id)
    # lookup against currencies.json (which keys BSC tokens under
    # "bsc"), so the frontend ends up with "currency: bnb",
    # symbol="bnb", unit=6 for USDT, etc.
    if c in {"binance", "bsc", "bnb", "bep20"}:
        return "bsc"
    return c

def _normalize_tx_descriptor(desc: str, chain: str, token_id: int | None) -> str:
    if not desc:
        return desc
    parts = desc.split("-")
    if len(parts) < 4:
        return desc
    suffix = parts[-1]
    token_part = parts[-2]
    hash_part = "-".join(parts[:-3])
    token_val = str(token_id) if token_id is not None else token_part
    return f"{hash_part}-{chain}-{token_val}-{suffix}"

def _get_descriptor(address: str, chain: str, token_id: int = 0) -> str:
    """Generate a descriptor string for nodes/edges."""
    # Standard address descriptor
    return f"{address}-{chain}-{token_id}"


def _parse_address_descriptor(desc: str) -> tuple[str, str, int] | None:
    """Parse ``address-chain-token_id`` descriptors.

    Address parts may contain dashes (rare but possible in non-EVM chains), so
    we split from the right and treat the last two segments as chain/token_id.
    Returns ``None`` when the descriptor doesn't match an address node format.
    """
    if not desc or not isinstance(desc, str):
        return None
    parts = desc.rsplit("-", 2)
    if len(parts) != 3:
        return None
    addr, chain, token_raw = parts
    if not addr or not chain:
        return None
    try:
        token_id = int(token_raw)
    except (TypeError, ValueError):
        return None
    return addr, chain, token_id

def _get_timestamp(t: Any) -> int:
    if hasattr(t, 'timestamp'):
        return int(t.timestamp())
    if isinstance(t, (int, float)):
        return int(t)
    if isinstance(t, str):
        # Try numeric string
        try:
            return int(float(t))
        except (ValueError, OverflowError):
            pass
        # Try ISO date/time
        try:
            return int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError):
            return 0
    return 0

def _is_service(entity: Entity | None) -> bool:
    """Check if entity should be visualized as a service comment block."""
    if not entity:
        return False
    # Roles that are services
    SERVICE_ROLES = {"cex_deposit", "bridge_service", "otc_service", "unidentified_service"}
    return entity.role in SERVICE_ROLES

def _compute_positions(
    nodes: set[str],
    edges: list[tuple[str, str]],
    victim_address: str,
    service_descriptors: set[str],
    node_weights: dict[str, float],
    all_steps: list | None = None,
    get_node_descriptor_fn=None,
    token_id_map: dict | None = None,
) -> dict[str, dict[str, float]]:
    """
    Each trace-path becomes a horizontal row:
        item ─ tx ─ item ─ tx ─ item   (same Y, equal X spacing)
    Rows are stacked top-to-bottom for each fan-out.

    When two paths share a prefix (e.g. victim → perpetrator) the shared
    nodes keep their first-assigned position so the layout stays compact.
    """
    X_START = 353.5
    X_ITEM_GAP = 571          # distance between two adjacent address nodes
    Y_ROW_GAP = 185           # vertical distance between rows

    positions: dict[str, dict[str, float]] = {}
    current_row_y = X_START   # first row y (reusing X_START value for symmetry)

    if not all_steps or get_node_descriptor_fn is None:
        # Fallback: simple BFS layout if steps aren't provided
        adj: dict[str, list[str]] = defaultdict(list)
        in_deg: dict[str, int] = defaultdict(int)
        for u, v in edges:
            adj[u].append(v)
            in_deg[v] += 1
            if u not in in_deg:
                in_deg[u] = 0

        roots = [n for n in nodes if victim_address.lower() in n.lower() and n not in service_descriptors]
        if not roots:
            roots = [n for n in nodes if in_deg[n] == 0]
        if not roots and nodes:
            roots = [next(iter(nodes))]

        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for r in roots:
            queue.append((r, 0))
            visited[r] = 0
        while queue:
            u, lvl = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited[v] = lvl + 1
                    queue.append((v, lvl + 1))
        for n in nodes:
            if n not in visited:
                visited[n] = 0

        levels: dict[int, list[str]] = defaultdict(list)
        for n, lvl in visited.items():
            levels[lvl].append(n)
        y = X_START
        for lvl in sorted(levels):
            lvl_nodes = sorted(levels[lvl], key=lambda n: (-node_weights.get(n, 0.0), n))
            for nd in lvl_nodes:
                positions[nd] = {"x": X_START + lvl * X_ITEM_GAP, "y": y}
                y += Y_ROW_GAP
        return positions

    # ── Build unique paths as address-descriptor chains ──────────────
    # Gather chains of address descriptors per path
    path_chains: list[list[str]] = []
    seen_chain_tuples: set[tuple[str, ...]] = set()

    _tid_map = token_id_map or {}

    def _resolve_token_id(asset: str, chain: str) -> int:
        return _tid_map.get((chain, asset), _get_token_id(asset, chain))

    # Build adjacency among address descriptors from the steps
    adj_steps: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx, step in enumerate(all_steps):
        chain = _normalize_chain(step.chain)
        asset = (step.asset or "").upper()
        token_id = _resolve_token_id(asset, chain)
        src = get_node_descriptor_fn(step.from_address, chain, token_id)
        dst = get_node_descriptor_fn(step.to_address, chain, token_id)
        adj_steps[src].append((dst, idx))

    # Find roots (nodes with no incoming from other steps)
    all_srcs: set[str] = set()
    all_dsts: set[str] = set()
    for step in all_steps:
        chain = _normalize_chain(step.chain)
        asset = (step.asset or "").upper()
        token_id = _resolve_token_id(asset, chain)
        all_srcs.add(get_node_descriptor_fn(step.from_address, chain, token_id))
        all_dsts.add(get_node_descriptor_fn(step.to_address, chain, token_id))

    root_descs = all_srcs - all_dsts
    if not root_descs:
        root_descs = {next(iter(all_srcs))} if all_srcs else set()

    # DFS from roots to enumerate all leaf-ending paths (cycle-safe)
    def _dfs(node: str, current_path: list[str], visited_in_path: set[str]):
        nexts = adj_steps.get(node, [])
        unvisited = [(dst, idx) for dst, idx in nexts if dst not in visited_in_path]
        if not unvisited:
            if len(current_path) > 1:
                chain_tuple = tuple(current_path)
                if chain_tuple not in seen_chain_tuples:
                    seen_chain_tuples.add(chain_tuple)
                    path_chains.append(list(current_path))
            return
        for dst, _idx in unvisited:
            _dfs(dst, current_path + [dst], visited_in_path | {dst})

    for root in sorted(root_descs):
        _dfs(root, [root], {root})

    # Sort paths: longer / higher-weight paths first
    def _path_weight(p: list[str]) -> float:
        return sum(node_weights.get(n, 0.0) for n in p)
    path_chains.sort(key=lambda p: (-_path_weight(p), -len(p)))

    # ── Assign positions row by row ──────────────────────────────────
    # First path gets the baseline row; every subsequent path that
    # introduces new nodes gets a fresh row so labels never overlap.
    current_row_y = 311.25
    is_first_path = True

    for chain_descs in path_chains:
        new_descs = [d for d in chain_descs if d not in positions]
        if not new_descs:
            continue  # entire path already positioned

        if not is_first_path:
            current_row_y += Y_ROW_GAP
        is_first_path = False

        for col, desc in enumerate(chain_descs):
            if desc not in positions:
                positions[desc] = {
                    "x": X_START + col * X_ITEM_GAP,
                    "y": current_row_y,
                }

    # Position any remaining nodes not reached by path walks
    for n in nodes:
        if n not in positions:
            current_row_y += Y_ROW_GAP
            positions[n] = {"x": X_START, "y": current_row_y}

    return positions

def _get_token_id(asset: str, chain: str) -> int:
    """Resolve a deterministic on-chain token_id for ``(chain, asset)``.

    Order: native-symbol shortcut → currencies.json registry lookup → 0.

    Used as a fallback when ``token_id_map`` (built from the supplied
    ``tx_list`` / ``txs``) doesn't have the (chain, asset) pair. The
    previous implementation used Python's randomized ``hash()`` to
    fabricate an ID in [1, 1000], which produced non-deterministic IDs
    like ``431`` on every run and never matched the platform's
    canonical token_ids (e.g. BSC USDT = 9). Returning 0 on miss keeps
    the descriptor lane stable instead of inventing a phantom token.
    """
    if not asset:
        return 0

    asset_upper = asset.upper()
    chain_upper = chain.upper()

    # Native asset symbols on common chains.
    if asset_upper == chain_upper or asset_upper in {
        "ETH", "BTC", "TRX", "SOL", "MATIC", "BNB"
    }:
        return 0

    # Look up the canonical token_id from the seeded currency registry.
    # This handles BSC USDT (token_id=9), ETH USDT (token_id=94252),
    # TRX USDT (token_id=9), and every other token we ship a record for
    # — without depending on a non-deterministic Python hash.
    try:
        from .currency_registry import get_registry
        rec = get_registry().lookup_by_symbol(chain, asset_upper)
        if rec is not None:
            return rec.token_id
    except Exception:  # noqa: BLE001 — registry must never break the viz
        logger.debug("currency_registry lookup failed for (%s, %s)", chain, asset)

    return 0

def generate_visualization_payload(
    trace_result: TraceResult,
    title: str | None = None,
    tx_list: list[dict[str, Any]] | None = None,
    txs: list[dict[str, Any]] | None = None,
    address_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Generate visualization payload from TraceResult.
    Format matches the expected structure (corrected.json).
    """
    import time

    logger.info("🔧 Starting visualization payload generation...")
    logger.debug(f"Case ID: {trace_result.case_meta.case_id}, Victim: {trace_result.case_meta.victim_address}")

    # 1. Prepare data structures
    items = []
    connects = []
    txs_output = []
    comments = []
    currency_info = {}
    tx_list_inputs = []
    if tx_list:
        for tx in tx_list:
            norm = dict(tx)
            norm_chain = _normalize_chain(norm.get("currency"))
            norm["currency"] = norm_chain
            if "tokenId" in norm and norm["tokenId"] is not None:
                try:
                    norm["tokenId"] = int(norm["tokenId"])
                except (ValueError, TypeError):
                    logger.warning("Could not convert tokenId to int: %s", norm["tokenId"])
            if "token_id" in norm and norm["token_id"] is not None:
                try:
                    norm["token_id"] = int(norm["token_id"])
                except (ValueError, TypeError):
                    logger.warning("Could not convert token_id to int: %s", norm["token_id"])
            tx_list_inputs.append(norm)
    use_provided_tx_list = bool(tx_list_inputs)

    fiat_rate_by_hash: dict[str, float] = {}
    provided_tx_hashes: set[str] = set()
    # Raw (on-chain base-unit) amount keyed by tx hash, harvested from the
    # incoming tx_list. Used to populate ``connects[].data.amount`` so each
    # edge is self-sufficient and the frontend doesn't need to fall back to
    # a (sometimes failing) hash→txList join to know how much was moved.
    amount_by_hash: dict[str, Any] = {}
    amount_by_hash_pair: dict[tuple[str, str, str], Any] = {}
    amount_by_hash_pair_token: dict[tuple[str, str, str, int], Any] = {}
    # Actual on-chain token_id keyed by tx hash, harvested from tx_list/txs.
    # The Step model carries only the *trace-level* asset symbol (e.g. "ETH"
    # for a stolen-ETH case), so a USDT hop downstream still has
    # ``step.asset == "ETH"`` and the naive ``token_id_map[(chain,asset)]``
    # lookup returns 0. Feeding the real per-hash token_id into the edge
    # payload lets the frontend render "413.759798 USDT" instead of "NaN".
    token_id_by_hash: dict[str, int] = {}
    # Preserve per-hash/per-leg path metadata from tx_list so tx nodes can
    # resolve the same transfer branch the sidebar shows (critical for
    # multi-transfer same-hash EVM txs; otherwise frontend falls back to
    # Zero Address when hash/path don't match).
    tx_path_by_hash: dict[str, str] = {}
    tx_path_by_hash_pair: dict[tuple[str, str, str], str] = {}
    tx_path_by_hash_pair_token: dict[tuple[str, str, str, int], str] = {}
    for tx in tx_list_inputs:
        h = tx.get("hash")
        if h:
            fiat_rate_by_hash[h] = float(tx.get("fiatRate") or tx.get("fiat_rate") or 1.0)
            provided_tx_hashes.add(h)
            amt = tx.get("amount")
            if amt is not None:
                amount_by_hash[h] = amt
            tid = tx.get("tokenId")
            if tid is None:
                tid = tx.get("token_id")
            if tid is not None:
                try:
                    tid_int = int(tid)
                    existing_tid = token_id_by_hash.get(h)
                    # Keep the most-informative token id when both a real tx
                    # row and a synthetic bridge fallback row share one hash.
                    # Synthetic rows often carry token_id=0; they must not
                    # clobber an already-discovered non-native token id.
                    if existing_tid is None or not (existing_tid != 0 and tid_int == 0):
                        token_id_by_hash[h] = tid_int
                except (TypeError, ValueError):
                    pass
            path_val = tx.get("path")
            if path_val is not None:
                path_str = str(path_val)
                tx_path_by_hash.setdefault(h, path_str)
                ins = tx.get("inputs") or []
                outs = tx.get("outputs") or []
                if ins and outs:
                    in_addr = ins[0].get("address")
                    out_addr = outs[0].get("address")
                    if in_addr and out_addr:
                        tx_path_by_hash_pair.setdefault((h, in_addr, out_addr), path_str)
                        if amt is not None:
                            amount_by_hash_pair.setdefault((h, in_addr, out_addr), amt)
                        if tid is not None:
                            try:
                                tid_int = int(tid)
                                tx_path_by_hash_pair_token.setdefault(
                                    (h, in_addr, out_addr, tid_int),
                                    path_str,
                                )
                                if amt is not None:
                                    amount_by_hash_pair_token.setdefault(
                                        (h, in_addr, out_addr, tid_int),
                                        amt,
                                    )
                            except (TypeError, ValueError):
                                pass
        # Backfill display-name fields on entries coming from
        # ``_collect_token_transfer_data`` (they only carry the short
        # chain code). The frontend sidebar reads ``blockchain`` /
        # ``blockchain_name`` for the "Blockchain: …" row; without
        # these, it renders the "MOCK DATA" sentinel.
        if not tx.get("blockchain"):
            tx_chain = _normalize_chain(tx.get("currency") or "")
            tx["blockchain"] = _CURRENCY_NAMES.get(tx_chain, tx_chain.upper() if tx_chain else "")
        if not tx.get("blockchain_name"):
            tx["blockchain_name"] = tx.get("blockchain") or ""
    # Also harvest token_id from the pre-built ``txs`` list (populated by
    # ``_collect_token_transfer_data`` in base_tracer). That path is the one
    # that actually fires for in-trace resolved token hops, so it covers
    # cases where tx_list was filtered/capped.
    if txs:
        for tx in txs:
            h = tx.get("hash")
            if not h or h in token_id_by_hash:
                continue
            tid = tx.get("token_id")
            if tid is None:
                tid = tx.get("tokenId")
            if tid is not None:
                try:
                    token_id_by_hash[h] = int(tid)
                except (TypeError, ValueError):
                    pass

    address_to_entity = {e.address: e for e in trace_result.entities}

    # Build risk-score and owner lookups from txList (more reliable than
    # entity data which may be 0/empty for addresses added by the postprocessor).
    risk_from_txlist: dict[str, float] = {}
    owner_from_txlist: dict[str, dict] = {}
    for tx in tx_list_inputs:
        for inp in (tx.get("inputs") or []):
            addr = inp.get("address")
            rs = inp.get("riskscore")
            if addr and rs and (rs > risk_from_txlist.get(addr, 0.0)):
                risk_from_txlist[addr] = float(rs)
            if addr and inp.get("owner") and addr not in owner_from_txlist:
                owner_from_txlist[addr] = inp["owner"]
        for out in (tx.get("outputs") or []):
            addr = out.get("address")
            rs = out.get("riskscore")
            if addr and rs and (rs > risk_from_txlist.get(addr, 0.0)):
                risk_from_txlist[addr] = float(rs)
            if addr and out.get("owner") and addr not in owner_from_txlist:
                owner_from_txlist[addr] = out["owner"]

    # Build additional owner lookups from addressInfo (get_address API results)
    address_info_data = address_info or {}
    for addr, chains_data in address_info_data.items():
        if addr in owner_from_txlist:
            continue
        for _chain_key, chain_data in (chains_data.items() if isinstance(chains_data, dict) else []):
            if not isinstance(chain_data, dict):
                continue
            ai_owner = chain_data.get("owner")
            if ai_owner and isinstance(ai_owner, dict) and ai_owner.get("name"):
                owner_from_txlist[addr] = ai_owner
                break

    # Labels that are meta-annotations, not real owner/service names.
    _META_LABELS = {"High Risk", "Suspected Perpetrator", "Exchange", "Bridge", "DEX"}

    service_comment_map = {}
    ren_counter = 0
    # Postprocess (``trace_postprocess.ensure_entity``) rekeys entities by
    # ``(address, chain)`` and auto-adds an ``intermediate`` Entity for each
    # chain an address touches. On a cross-chain bridge hop that means the
    # bridge contract gets TWO Entity rows: ``(Bridgers, eth, bridge_service)``
    # from the tracer and ``(Bridgers, trx, intermediate)`` from postprocess.
    # The second one must not clobber or duplicate the bridge's label, so we
    # only consult the authoritative entity per address here.
    _comment_worthy_roles = {
        "victim", "perpetrator", "bridge_service",
        "cex_deposit", "otc_service", "unidentified_service",
    }
    _role_priority = {
        "victim": 5, "perpetrator": 4,
        "bridge_service": 4, "cex_deposit": 4,
        "otc_service": 4, "unidentified_service": 4,
        "intermediate": 1,
    }
    _best_entity_by_addr: dict[str, Entity] = {}
    for entity in trace_result.entities:
        cur = _best_entity_by_addr.get(entity.address)
        if cur is None or _role_priority.get(entity.role, 0) > _role_priority.get(cur.role, 0):
            _best_entity_by_addr[entity.address] = entity
    for address, entity in _best_entity_by_addr.items():
        if entity.role in _comment_worthy_roles:
            service_comment_map[address] = f"«ren»{ren_counter}"
            ren_counter += 1

    # --- Pass 1: Build Graph Topology & Weights ---
    node_descriptors = set()
    service_descriptors = set(service_comment_map.values())
    edges = []

    token_id_map = {} # (chain, asset) -> int
    # Collect every (chain, token_id) actually observed in tx_list so the
    # currencyInfo block below can emit accurate name/symbol/unit/issuer
    # entries for each token (not just whichever one the case_meta hints at).
    observed_token_ids: set[tuple[str, int]] = set()
    if tx_list_inputs:
        try:
            case_hint = (trace_result.case_meta.asset_symbol or "").upper()
        except AttributeError:
            case_hint = ""
        for tx in tx_list_inputs:
            chain = _normalize_chain(tx.get("currency"))
            token_id = tx.get("tokenId")
            if token_id is None:
                token_id = tx.get("token_id")
            if not chain or token_id is None:
                continue
            tid = int(token_id)
            observed_token_ids.add((chain, tid))
            if tid == 0:
                native_sym = chain.upper()
                token_id_map.setdefault((chain, native_sym), 0)
                token_id_map.setdefault((chain, ""), 0)
                # The case hint usually matches the native symbol for a
                # native-asset case (e.g. "ETH" on eth) — set it too so
                # lookups by step.asset="ETH" keep returning 0 even when a
                # token-transfer tx appears later.
                if case_hint:
                    token_id_map.setdefault((chain, case_hint), 0)
            else:
                # Look up the *actual* asset symbol for this (chain, tid)
                # from the canonical currency DB instead of trusting the
                # case-level asset_symbol hint. Otherwise, a single USDT
                # transfer (tid=94252) in an "ETH" case would wrongly map
                # ("eth","ETH") → 94252 and drag every native edge with it.
                rec = _lookup_currency(chain, tid)
                sym = (rec.get("symbol") if rec else None) or ""
                sym = str(sym).upper()
                if sym:
                    token_id_map.setdefault((chain, sym), tid)
    node_weights = defaultdict(float) # descriptor -> total volume

    def get_node_descriptor(address: str, chain: str, token_id: int) -> str:
        return _get_descriptor(address, chain, token_id)

    # Collect unique steps across all paths.  Paths share common prefixes so
    # the same (from, to, tx_hash) triple can appear many times; we keep only
    # the first occurrence to avoid duplicate edges in the visualization.
    #
    # Drop the terminal step of any path trimmed below the dust threshold —
    # operator feedback is that those edges add clutter without decision
    # value (they represent <1% of stolen funds going to some random
    # address). The step is still present in ``trace_result.paths`` for
    # API consumers and the "Dust Trimmed" annotation still describes
    # what happened; only the visualization hides it.
    _step_keys_seen: set[tuple[str, str, str]] = set()
    all_steps = []
    for path in trace_result.paths:
        stop_reason = (path.stop_reason or "").lower()
        is_dust_path = stop_reason.startswith("below dust threshold")
        steps_to_render = path.steps
        if is_dust_path and path.steps:
            # Last step is the one into the dust recipient — drop it.
            steps_to_render = path.steps[:-1]
        for step in steps_to_render:
            key = (step.from_address, step.to_address, step.tx_hash or "")
            if key not in _step_keys_seen:
                _step_keys_seen.add(key)
                all_steps.append(step)

    # Only keep descriptor mappings from pre-collected txs; positions will
    # be recomputed from the node layout so we do NOT copy them into txs_output.
    tx_desc_by_hash = {}
    if txs:
        for tx in txs:
            norm = dict(tx)
            chain_n = _normalize_chain(norm.get("currency"))
            norm["currency"] = chain_n
            tid = norm.get("token_id")
            if tid is None:
                tid = norm.get("tokenId")
            if tid is not None:
                try:
                    tid = int(tid)
                except (ValueError, TypeError):
                    logger.warning("Could not convert token_id to int in txs: %s", tid)
                norm["token_id"] = tid
            desc = norm.get("descriptor")
            if desc:
                norm["descriptor"] = _normalize_tx_descriptor(desc, chain_n, tid)
            tx_hash = norm.get("hash")
            tx_desc = norm.get("descriptor")
            if tx_hash and tx_desc:
                tx_desc_by_hash[tx_hash] = tx_desc
                # Belt-and-suspenders: also harvest the token_id straight out of
                # the descriptor string. ``_collect_token_transfer_data`` bakes
                # the real token_id into the descriptor as the second-to-last
                # dash-separated segment, so this works even if ``tx_list`` and
                # ``txs`` both fail to carry ``tokenId`` for the hash.
                desc_tid = _token_id_from_descriptor(tx_desc)
                if desc_tid is not None and tx_hash not in token_id_by_hash:
                    token_id_by_hash[tx_hash] = desc_tid
    tx_desc_seen: set[str] = set()
    # For UTXO multi-output seed txs, a single on-chain tx (one tx_hash)
    # produces many steps with distinct (from, to) pairs. Each step emits
    # a `src→tx` and a `tx→tgt` edge; without deduping by (src, tx_hash)
    # we'd repeat the same `src→tx` edge once per output.
    #
    # We key on tx_hash (not tx_desc) because some steps synthesize a
    # per-index descriptor when ``tx_desc_by_hash`` is missing an entry
    # — the hash itself is the real aggregation key.
    src_to_tx_seen: set[tuple[str, str]] = set()

    # Pre-count how many rendered steps reference each tx_hash. A tx_hash
    # that appears in exactly one step is an account-model transfer (one
    # input, one output, one step), and ``amount_by_hash`` is already in
    # the API's base-unit — we trust it as-is and skip our own scaling
    # (avoids double-scaling ETH gwei → wei).
    # Multi-output UTXO seeds appear in multiple steps with the same hash
    # but different ``to`` addresses; those need per-output scaling.
    step_count_by_hash: dict[str, int] = {}
    for s in all_steps:
        if s.tx_hash:
            step_count_by_hash[s.tx_hash] = step_count_by_hash.get(s.tx_hash, 0) + 1

    for step in all_steps:
        chain = _normalize_chain(step.chain)
        asset = (step.asset or "").upper()
        key = (chain, asset)
        if key not in token_id_map:
            token_id_map[key] = _get_token_id(asset, chain)

        token_id = token_id_map[key]

        src = get_node_descriptor(step.from_address, chain, token_id)
        dst = get_node_descriptor(step.to_address, chain, token_id)

        node_descriptors.add(src)
        node_descriptors.add(dst)
        edges.append((src, dst))

        # Accumulate weight
        val = step.amount_estimate or 0.0
        node_weights[src] += val
        node_weights[dst] += val

    # Log graph topology
    logger.info(f"📊 Graph topology: {len(node_descriptors)} nodes, {len(edges)} edges")
    if token_id_by_hash:
        _nonzero = {h: t for h, t in token_id_by_hash.items() if t}
        if _nonzero:
            logger.info(
                f"🧮 Per-hash token_id overrides (edge/tx will advertise the real token): "
                f"{ {h[:10]+'…': t for h, t in _nonzero.items()} }"
            )
    logger.debug(f"Nodes: {list(node_descriptors)}")
    logger.debug(f"Token ID Map: {token_id_map}")

    # --- Pass 2: Compute Layout ---
    positions = _compute_positions(
        node_descriptors, edges, trace_result.case_meta.victim_address,
        service_descriptors, node_weights,
        all_steps=all_steps, get_node_descriptor_fn=get_node_descriptor,
        token_id_map=token_id_map,
    )

    # --- Pass 3: Generate Items & Comments ---
    added_descriptors = set()

    # Track addresses per chain/token for autoTxs grouping if needed,
    # but autoTxs is per-address.

    def add_node_or_comment(address: str, chain: str, token_id: int):
        descriptor = get_node_descriptor(address, chain, token_id)
        if descriptor in added_descriptors:
            return

        pos = positions.get(descriptor, {"x": 0, "y": 0})
        entity = address_to_entity.get(address)

        risk_score = (entity.risk_score if entity else 0.0) or risk_from_txlist.get(address, 0.0)

        owner = None
        if entity and entity.labels:
            real_labels = [lb for lb in entity.labels if lb not in _META_LABELS]
            if real_labels:
                owner = {
                    "id": 0,
                    "name": real_labels[0],
                    "slug": real_labels[0],
                    "type": "exchange_licensed" if "exchange" in (entity.role or "") else "unknown",
                    "subtype": None
                }
        if not owner and address in owner_from_txlist:
            txl_owner = owner_from_txlist[address]
            if isinstance(txl_owner, dict):
                owner = {
                    "id": txl_owner.get("id", 0),
                    "name": txl_owner.get("name", ""),
                    "slug": txl_owner.get("slug", txl_owner.get("name", "")),
                    "type": txl_owner.get("type", "unknown"),
                    "subtype": txl_owner.get("subtype"),
                }
            elif isinstance(txl_owner, str) and txl_owner:
                owner = {"id": 0, "name": txl_owner, "slug": txl_owner, "type": "unknown", "subtype": None}

        items.append({
            "address": address,
            "descriptor": descriptor,
            "x": pos["x"],
            "y": pos["y"],
            "extend": {
                "currency": chain,
                "token_id": token_id,
                "owner": owner,
                "riskScore": risk_score,
                "type": "address"
            },
            "type": "address",
            "isManuallyMoved": False
        })
        added_descriptors.add(descriptor)

    # --- Pre-fill Currency Info with Native ---
    # Ensure the case's native asset is present even if no step produced it
    # yet. Keyed by ``(chain, token_id)`` so a cross-chain trace keeps both
    # native entries (e.g. both eth:0 and trx:0) instead of one clobbering
    # the other.
    blockchain = _normalize_chain(trace_result.case_meta.blockchain_name)
    if blockchain:
        key = (blockchain, 0)
        if key not in currency_info:
            currency_info[key] = _build_currency_info(blockchain, 0)

    # Also pre-fill entries for every (chain, token_id) observed in the
    # tx_list so downstream consumers can resolve amounts/units even when
    # the token never appears as a step's asset (e.g. destination-side
    # tokens from bridge transfers).
    for chain_obs, tid_obs in observed_token_ids:
        key = (chain_obs, tid_obs)
        if key not in currency_info:
            currency_info[key] = _build_currency_info(chain_obs, tid_obs)

    # --- Pass 4: Generate Edges & Txs ---
    # Prepare for autoTxs: map address -> list of (step_index, type, hash, path)
    address_activity = defaultdict(list)

    # Disambiguate tx midpoints in two overlapping failure modes:
    #
    #   (a) Multiple distinct txs share the same (source, target) pair.
    #       Every such tx lands on ``(src+tgt)/2`` and the frontend
    #       stacks the amount labels on top of each other ("10k USDT)SDT"
    #       in the rendered graph) while drawing every edge over the
    #       same straight line.
    #
    #   (b) Completely unrelated pairs happen to share a midpoint. The
    #       layout places addresses on a regular X/Y grid, so two pairs
    #       whose src-columns and tgt-columns match AND whose y-averages
    #       coincide produce overlapping tx-nodes even though the
    #       endpoints differ (e.g. THWY8p→TC3rAMtu and TLdbvt→TYE5mN
    #       both collapse onto ``(2352, 496.25)`` in the TRX case).
    #
    # A single collision map keyed on the rounded midpoint coordinate
    # handles both cases: each tx that lands on an already-occupied
    # coordinate fans out ``±_TX_COLLISION_STEP_PX`` along Y in an
    # alternating pattern until a free slot is found.
    #
    # The step is wider than a label's rendered height (amount chips are
    # roughly 28–34px tall in the frontend, so 68px keeps two stacked
    # labels clearly separated with breathing room). The collision map
    # is checked against BOTH the raw midpoint and each candidate offset
    # position — this keeps a single overcrowded column from walking all
    # of its offset siblings into one neighbour's slot.
    _TX_COLLISION_STEP_PX = 68
    _tx_slot_taken: set[tuple[float, float]] = set()

    # Operators frequently turn on the platform's "merged tx mode" to
    # collapse repeated transactions between the same pair of addresses
    # into one fat edge. The expected payload shape:
    #
    #   * one ``mergedEdge`` entry in ``connects`` per unique (from, to)
    #     pair, with ``id="{from}-{to}"`` and address-descriptors as
    #     source/target;
    #   * one ``mergedTx`` entry in ``txs`` per unique pair, with
    #     ``descriptor=hash="{from}{to}"`` (no chain suffix, just the
    #     concatenated address pair);
    #   * each individual ``txEth``/``tx`` entry carries a
    #     ``parentNode="{from}{to}"`` linking it to the hub.
    #
    # When the frontend sees ``helpers.isMergedTxMode = true`` it hides
    # the per-tx connectors and shows the consolidated mergedEdge
    # instead. Multiple txs between the same pair (e.g. four
    # TLHPDaLrq → TN6cEuxV transfers) collapse into a single visual
    # arrow with a tx-count badge.
    edge_pairs: dict[tuple[str, str], dict[str, Any]] = {}

    def _resolve_tx_position(mid_x: float, mid_y: float) -> tuple[float, float]:
        """Return an unclaimed ``(x, y)`` near ``(mid_x, mid_y)``.

        The first caller on a given midpoint gets it verbatim; each
        subsequent caller tries ``±68``, ``±136``, … along Y until an
        empty slot is found. Slots are tracked in ``_tx_slot_taken`` so
        a later collision at a neighbouring pair can't silently land on
        an already-shifted sibling of ours.
        """
        # Round to 1-px bins so float noise doesn't hide a collision.
        base_x = round(mid_x, 1)
        base_y = round(mid_y, 1)
        if (base_x, base_y) not in _tx_slot_taken:
            _tx_slot_taken.add((base_x, base_y))
            return mid_x, mid_y
        # Walk outward: +68, -68, +136, -136, …
        step = _TX_COLLISION_STEP_PX
        rank = 1
        while True:
            for sign in (1, -1):
                candidate = (base_x, round(base_y + sign * rank * step, 1))
                if candidate not in _tx_slot_taken:
                    _tx_slot_taken.add(candidate)
                    return mid_x, mid_y + sign * rank * step
            rank += 1
            if rank > 40:
                # Runaway guard — 40 rank × 68px = 2720px vertical span.
                # Any real graph that outgrows this has bigger issues
                # than overlap; give up and reuse the midpoint.
                return mid_x, mid_y

    for i_step, step in enumerate(all_steps):
        chain = _normalize_chain(step.chain)
        asset = (step.asset or "").upper()
        # Address-view token_id: used to KEY the address node, so all hops on
        # a chain share one node per address regardless of which token moved
        # (e.g. an ETH trace that touches USDT keeps the deposit address on
        # the same ``-eth-0`` lane as its upstream ETH hop).
        node_token_id = token_id_map.get((chain, asset), 0)
        # Edge/tx token_id: reflects the *actual* asset that moved in this
        # specific tx. step.asset is propagated from the trace-level asset
        # and is often stale ("ETH" for a USDT hop), so prefer the real
        # token_id harvested from tx_list/txs by hash. Without this, the
        # edge payload advertises ``token_id=0, currency=eth`` for a USDT
        # transfer, the frontend can't join it to the USDT ``txList`` row,
        # and the edge label renders as "NaN".
        # Resolution order for the per-edge token_id (most → least trusted):
        #   1. token_id_by_hash harvested from ``tx_list``/``txs`` args.
        #   2. The pre-built descriptor in ``tx_desc_by_hash`` — its
        #      ``…-{chain}-{token_id}-{idx}`` suffix is authoritative because
        #      ``base_tracer._collect_token_transfer_data`` copies the real
        #      token_id into it.
        #   3. The chain-native token_id derived from ``step.asset``.
        edge_token_id = node_token_id
        if step.tx_hash:
            if step.tx_hash in token_id_by_hash:
                edge_token_id = token_id_by_hash[step.tx_hash]
            else:
                desc_tid = _token_id_from_descriptor(tx_desc_by_hash.get(step.tx_hash))
                if desc_tid is not None:
                    edge_token_id = desc_tid
                    token_id_by_hash[step.tx_hash] = desc_tid

        src_desc = get_node_descriptor(step.from_address, chain, node_token_id)
        tgt_desc = get_node_descriptor(step.to_address, chain, node_token_id)

        add_node_or_comment(step.from_address, chain, node_token_id)
        add_node_or_comment(step.to_address, chain, node_token_id)

        src_pos = positions.get(src_desc, {"x": 0, "y": 0})
        tgt_pos = positions.get(tgt_desc, {"x": 0, "y": 0})

        # Basic edge color
        edge_color = "#EC292C"

        tx_hash = step.tx_hash or f"tx-{uuid.uuid4().hex}"
        # The tx descriptor must match whatever was pre-baked in
        # ``tx_desc_by_hash`` (populated upstream by
        # ``_collect_token_transfer_data``) so the edge's ``target`` and the
        # ``txs[].descriptor`` line up; fall back to the real token_id when
        # no pre-built descriptor exists.
        tx_desc = tx_desc_by_hash.get(step.tx_hash) or f"{tx_hash}-{chain}-{edge_token_id}-{i_step}"

        mid_x = (src_pos["x"] + tgt_pos["x"]) / 2
        mid_y = (src_pos["y"] + tgt_pos["y"]) / 2
        # Displace the tx node off any midpoint another tx has already
        # claimed. Catches both same-pair repeats (b36a + 376c both on
        # TLHPDaL→TN6c) and reverse-direction collisions (TN6c→TLHPDaL
        # shares the same midpoint as TLHPDaL→TN6c when both endpoints
        # sit on the same Y row) plus accidental cross-pair hits where
        # unrelated src/tgt pairs happen to average to the same point.
        mid_x, mid_y = _resolve_tx_position(mid_x, mid_y)

        _UTXO_CHAINS = {"btc", "bch", "ltc"}
        is_utxo = chain in _UTXO_CHAINS
        tx_type = "tx" if is_utxo else "txEth"
        if is_utxo:
            tx_path = None
        else:
            tx_path = None
            if step.tx_hash:
                tx_path = tx_path_by_hash_pair_token.get(
                    (step.tx_hash, step.from_address, step.to_address, int(edge_token_id))
                )
                if tx_path is None:
                    tx_path = tx_path_by_hash_pair.get(
                        (step.tx_hash, step.from_address, step.to_address)
                    )
                if tx_path is None:
                    tx_path = tx_path_by_hash.get(step.tx_hash)
            if tx_path is None:
                tx_path = "0"
        if step.tx_hash and not is_utxo:
            # A single account-model tx hash can carry many transfer paths.
            # Encode path into tx descriptor so each leg gets its own node and
            # frontend joins tx node <-> txList row deterministically.
            tx_desc = f"{tx_hash}-{chain}-{edge_token_id}-{tx_path}"

        # Track this (from, to) pair for the merged-tx pass below.
        # The first step on a pair locks in the src/tgt descriptors and
        # color; subsequent steps on the same pair (multi-tx between
        # the same two addresses) all collapse into the same hub.
        pair_key = (step.from_address, step.to_address)
        if pair_key not in edge_pairs:
            edge_pairs[pair_key] = {
                "src_desc": src_desc,
                "tgt_desc": tgt_desc,
                "color": edge_color,
            }
        parent_descriptor = f"{step.from_address}{step.to_address}"

        if tx_desc not in tx_desc_seen:
            txs_output.append({
                "currency": chain,
                "descriptor": tx_desc,
                "hash": tx_hash,
                "token_id": edge_token_id,
                "x": mid_x,
                "y": mid_y,
                "color": edge_color,
                "path": tx_path,
                "type": tx_type,
                "parentNode": parent_descriptor,
            })
            tx_desc_seen.add(tx_desc)

        fiat_rate = fiat_rate_by_hash.get(step.tx_hash, 1.0) if step.tx_hash else 1.0

        # Edge amounts split between the src→tx and tx→tgt legs to handle
        # UTXO multi-output seeds correctly:
        #
        #   * src→tx : total tx amount (aggregate across all outputs). We
        #              prefer ``amount_by_hash`` because it reflects the
        #              real on-chain tx value.
        #   * tx→tgt : per-output share. We prefer ``step.amount_estimate``
        #              so each output edge carries its own size. Without
        #              this, a 7-output UTXO seed with 36.5M sat total
        #              would draw 7 edges of 36.5M each instead of their
        #              individual outputs (17.9M + 2M + 0.7M + …).
        #
        # For account-model chains both values coincide, so the split
        # is a no-op there. Unit scaling uses ``edge_token_id`` (the tx's
        # real token) so a 413.759798 USDT estimate becomes 413759798
        # (×10^6), not 413759798000 (×10^9, which is what would happen
        # if we used the native ETH unit for a USDT hop).
        rec = _lookup_currency(chain, edge_token_id)
        unit = (rec.get("unit") if rec else None)
        if unit is None:
            unit = _NATIVE_UNIT_MAP.get(chain, 6) if edge_token_id == 0 else 6

        per_step_amount: Any = None
        if step.amount_estimate is not None:
            try:
                per_step_amount = int(float(step.amount_estimate) * (10 ** int(unit)))
            except (TypeError, ValueError, OverflowError):
                per_step_amount = None

        total_tx_amount: Any = None
        if step.tx_hash:
            total_tx_amount = amount_by_hash_pair_token.get(
                (step.tx_hash, step.from_address, step.to_address, int(edge_token_id))
            )
            if total_tx_amount is None:
                total_tx_amount = amount_by_hash_pair.get(
                    (step.tx_hash, step.from_address, step.to_address)
                )
            if total_tx_amount is None and step.tx_hash in amount_by_hash:
                total_tx_amount = amount_by_hash[step.tx_hash]

        # Account-model txs (single step per tx_hash) trust amount_by_hash
        # because the API's base-unit representation is canonical. We
        # never do our own unit scaling for them — otherwise values that
        # are already in wei/sun/sat get multiplied a second time (the
        # ETH gwei → wei regression we're patching here).
        #
        # Only UTXO multi-output seeds (same tx_hash, multiple distinct
        # (from, to) steps) need per-output scaling, because
        # ``amount_by_hash`` holds the tx aggregate but each step carries
        # its own output-share in display units.
        is_multi_output = (
            step.tx_hash is not None
            and step_count_by_hash.get(step.tx_hash, 0) > 1
        )

        if total_tx_amount is not None and not is_multi_output:
            # Account-model: trust the API value verbatim, no scaling.
            src_edge_amount = total_tx_amount
            tgt_edge_amount = total_tx_amount
        else:
            # UTXO multi-output (or account tx with no amount_by_hash
            # entry — rare, e.g. synthetic bridge-destination hashes).
            src_edge_amount = total_tx_amount if total_tx_amount is not None else per_step_amount
            tgt_edge_amount = per_step_amount if per_step_amount is not None else total_tx_amount
        # Back-compat alias: the txList-fallback block below references
        # ``edge_amount`` as a single post-resolution amount for the step.
        # Use the per-output share (what the edge actually carries).
        edge_amount = tgt_edge_amount

        # UTXO multi-output txs intentionally fan out into multiple ``tx_desc``
        # values (…-0, …-1, …-2) so each output branch has its own tx→tgt edge.
        # But the input side must stay singular: one on-chain tx has one input
        # edge from the source address. Dedup by tx_hash for UTXO and by
        # descriptor elsewhere.
        src_tx_key = (src_desc, step.tx_hash if (is_utxo and step.tx_hash) else tx_desc)
        if src_tx_key not in src_to_tx_seen:
            src_to_tx_seen.add(src_tx_key)
            connects.append({
                "source": src_desc,
                "target": tx_desc,
                "data": {
                    "currency": chain,
                    "amount": src_edge_amount,
                    "fiatRate": fiat_rate,
                    "token_id": edge_token_id,
                    "color": edge_color,
                    "type": "straight",
                    "isNew": True,
                    "isNeedReverse": False,
                    "hovered": False
                }
            })
        connects.append({
            "source": tx_desc,
            "target": tgt_desc,
            "data": {
                "currency": chain,
                "amount": tgt_edge_amount,
                "fiatRate": fiat_rate,
                "token_id": edge_token_id,
                "color": edge_color,
                "type": "straight",
                "isNew": True,
                "isNeedReverse": False,
                "hovered": False
            }
        })

        # Record activity for autoTxs (keyed on the *node* token_id so each
        # chain-view of an address keeps a single activity bucket).
        address_activity[(step.from_address, chain, node_token_id)].append({
            "type": "out",
            "hash": step.tx_hash,
            "index": i_step,
            "path": tx_path
        })
        address_activity[(step.to_address, chain, node_token_id)].append({
             "type": "in",
             "hash": step.tx_hash,
             "index": i_step,
             "path": tx_path
        })

        # Emit a txList entry when the step's tx is not already represented
        # there. Three distinct cases need coverage:
        #   1. No tx_list was provided at all → we must synthesize everything.
        #   2. step.tx_hash is None → we've generated a synthetic ``tx-…`` hash
        #      that certainly isn't in any upstream list.
        #   3. step.tx_hash is a *real* hash that the tx-collector never saw —
        #      typically bridge/swap destination txs that live on a different
        #      chain than the source-side ``all-txs`` queries. Without an
        #      entry here the frontend cannot resolve amount/currency for the
        #      edge and renders "NaN".
        is_synthetic_hash = step.tx_hash is None
        is_missing_hash = bool(step.tx_hash) and step.tx_hash not in provided_tx_hashes
        if not use_provided_tx_list or is_synthetic_hash or is_missing_hash:
            if tx_hash:
                provided_tx_hashes.add(tx_hash)
            # Human-readable blockchain name — frontend's tx-sidebar
            # looks this up directly to render "Blockchain: TRON" etc.
            # Without it, the card falls back to a sentinel ("MOCK
            # DATA") even when we have a fully valid currency/tokenId
            # resolution. Keep ``currency`` as the short chain code for
            # any downstream that joins by it.
            chain_display = _CURRENCY_NAMES.get(chain, chain.upper() if chain else "")
            tx_list_inputs.append({
                "inputs": [{"address": step.from_address, "riskscore": address_to_entity.get(step.from_address, Entity(address="",chain="",role="intermediate",risk_score=0.0)).risk_score or 0.0, "type": "address"}],
                "outputs": [{"address": step.to_address, "riskscore": address_to_entity.get(step.to_address, Entity(address="",chain="",role="intermediate",risk_score=0.0)).risk_score or 0.0, "type": "address"}],
                "hash": tx_hash,
                "fiatRate": fiat_rate,
                "addressesCount": 2,
                "amount": edge_amount if edge_amount is not None else (
                    int((step.amount_estimate or 0) * 1e6) if chain == 'trx' else step.amount_estimate
                ),
                "currency": chain,
                "tokenId": edge_token_id,
                "blockchain": chain_display,
                "blockchain_name": chain_display,
                "poolTime": _get_timestamp(step.time),
                "date": _get_timestamp(step.time),
                "path": tx_path,
                "type": tx_type,
                "reasoning": step.reasoning,
                "step_type": step.step_type,
                "service_label": step.service_label,
                "direction": step.direction
            })

        ci_key = (chain, edge_token_id)
        if ci_key not in currency_info:
            currency_info[ci_key] = _build_currency_info(
                chain=chain,
                token_id=edge_token_id,
                asset_hint=(step.asset or "").upper(),
            )

    # --- Merged-edge / merged-tx hubs (one per (from, to) pair) ---
    # Operators turn on the platform's "merged tx mode" to collapse
    # repeated transfers between the same pair of addresses into one
    # fat arrow with a count badge — this is what the human-built
    # graphs showed for ``TLHPDaLrq → TN6cEuxV`` (4 transfers
    # collapsing into a single visual edge with 30k/220k/etc. amounts
    # listed inside).
    #
    # We emit one ``mergedEdge`` connect (id="{from}-{to}",
    # source/target = address-descriptors) and one ``mergedTx`` item
    # (descriptor=hash="{from}{to}", input/output) per unique pair.
    # Every individual ``txEth``/``tx`` already carries
    # ``parentNode="{from}{to}"`` from the loop above, so the frontend
    # can group them under the hub when the toggle is on.
    for (from_addr, to_addr), info in edge_pairs.items():
        pair_id = f"{from_addr}-{to_addr}"
        parent_descriptor = f"{from_addr}{to_addr}"
        connects.append({
            "id": pair_id,
            "type": "mergedEdge",
            "source": info["src_desc"],
            "target": info["tgt_desc"],
            "data": {
                "input": from_addr,
                "output": to_addr,
                "color": info["color"],
            },
        })
        txs_output.append({
            "descriptor": parent_descriptor,
            "hash": parent_descriptor,
            "input": from_addr,
            "output": to_addr,
            "x": 0,
            "y": 0,
            "type": "mergedTx",
        })

    # --- Intra-address token-lane connectors ---
    # The same address can appear on multiple token lanes on one chain
    # (e.g. ``0xabc-eth-0`` and ``0xabc-eth-94252``). Without an explicit
    # connector, the graph can look like two unrelated components although it is
    # literally one address. Link those descriptors with a subtle dashed edge.
    addr_chain_to_descs: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for desc in added_descriptors:
        parsed = _parse_address_descriptor(desc)
        if not parsed:
            continue
        addr, desc_chain, desc_token = parsed
        addr_chain_to_descs[(addr, desc_chain)].append((desc_token, desc))

    alias_edge_seen: set[tuple[str, str]] = set()
    for (_addr, _chain), token_descs in addr_chain_to_descs.items():
        if len(token_descs) < 2:
            continue
        token_descs.sort(key=lambda t: t[0])
        for i in range(1, len(token_descs)):
            src_d = token_descs[i - 1][1]
            tgt_d = token_descs[i][1]
            key = (src_d, tgt_d)
            if key in alias_edge_seen:
                continue
            alias_edge_seen.add(key)
            connects.append({
                "source": src_d,
                "target": tgt_d,
                "data": {
                    "color": "#9AA3B2",
                    "type": "dashed",
                    "isNew": True,
                    "isNeedReverse": False,
                    "hovered": False,
                    "label": "Same address",
                },
            })

    # --- Cross-chain bridge handoff connectors ---
    # For each ``bridge_transfer`` step, the tracer emits the step on the
    # *destination* chain (chain=dst), with ``from_address`` set to the
    # bridge contract's on-source-chain address. Visualization therefore
    # creates two nodes for the same bridge contract — one on the source
    # chain (rendered from the upstream hop) and one on the destination
    # chain (rendered from this step). Without an explicit link between
    # them the two Bridgers nodes float as disconnected components on
    # the graph. Draw a dashed cross-chain connector so operators see
    # the same contract on both chains.
    bridge_handoff_seen: set[tuple[str, str]] = set()
    for step in all_steps:
        if (step.step_type or "") != "bridge_transfer":
            continue
        bridge_addr = step.from_address
        if not bridge_addr:
            continue
        dst_chain = _normalize_chain(step.chain)
        addr_prefix = f"{bridge_addr}-"
        candidates = [d for d in added_descriptors if d.startswith(addr_prefix)]
        dst_descriptors = [d for d in candidates if d.startswith(f"{bridge_addr}-{dst_chain}-")]
        other_descriptors = [d for d in candidates if not d.startswith(f"{bridge_addr}-{dst_chain}-")]
        if not dst_descriptors or not other_descriptors:
            continue
        src_d = sorted(other_descriptors)[0]
        tgt_d = sorted(dst_descriptors)[0]
        key = (src_d, tgt_d)
        if key in bridge_handoff_seen:
            continue
        bridge_handoff_seen.add(key)
        connects.append({
            "source": src_d,
            "target": tgt_d,
            "data": {
                "color": "#77869E",
                "type": "dashed",
                "isNew": True,
                "isNeedReverse": False,
                "hovered": False,
                "label": "Bridge",
            },
        })

    # --- Generate autoTxs ---
    auto_txs = []

    for (address, chain, token_id), activities in address_activity.items():
        if address in service_comment_map:
            continue

        # Sort by step index
        activities.sort(key=lambda x: x["index"])

        for i, _act in enumerate(activities):
            data_block = {}

            # Link Next
            if i < len(activities) - 1:
                next_act = activities[i+1]
                data_block["next_" + next_act["type"]] = {
                    "hash": next_act["hash"],
                    "path": next_act["path"]
                }

            # Link Prev
            if i > 0:
                prev_act = activities[i-1]
                data_block["prev_" + prev_act["type"]] = {
                    "hash": prev_act["hash"],
                    "path": prev_act["path"]
                }

            # Offset? (Random or calculated)
            data_block["offset"] = (i + 1) * 100 # Dummy offset

            auto_txs.append({
                "address": address,
                "currency": chain,
                "token_id": token_id,
                "data": data_block
            })

    payload = {
        "comments": comments,
        "connects": connects,
        "items": items,
        "transform": {"k": 1, "x": 0, "y": 0},
        "txs": txs_output
    }

    # --- Ensure terminal (leaf) addresses also get comment labels ---
    # The trace may stop before classifying the last address; detect leaf
    # nodes and add them to service_comment_map so the visualization still
    # marks the endpoint. Historically this block forcibly re-labelled every
    # intermediate leaf as ``cex_deposit`` ("Exchange deposit address"),
    # which produced false "exchange" claims on plain mule addresses the
    # classifier never actually identified as a service. We keep the role
    # honest (``intermediate`` stays ``intermediate``) and instead pick a
    # comment label based on *why* the path stopped.
    # Any address that appears as a ``from_address`` on at least one
    # rendered step has outgoing edges — it's an intermediate node, not a
    # terminal leaf, regardless of whether *some* path happens to stop
    # there. Operator feedback: labeling such nodes "Destination address"
    # is confusing because the graph clearly shows the trace continuing
    # through them. Only label true leaves (no outgoing edges anywhere).
    addrs_with_outgoing: set[str] = set()
    for _s in all_steps:
        if _s.from_address:
            addrs_with_outgoing.add(_s.from_address)

    terminal_stop_reasons: dict[str, str] = {}
    for path in trace_result.paths:
        if not path.steps:
            continue
        reason = (path.stop_reason or "").strip()
        # Mirror the dust filter above: dust-trimmed paths had their
        # terminal step removed from the render, so their dust recipient
        # is no longer a graph node. Don't resurrect it as a comment.
        if reason.lower().startswith("below dust threshold"):
            continue
        leaf = path.steps[-1].to_address
        if leaf in addrs_with_outgoing:
            # This address is a through-node on another (rendered) path.
            continue
        if leaf not in terminal_stop_reasons or reason:
            terminal_stop_reasons[leaf] = reason

    # Address types that warrant a dust-endpoint label. Operator feedback:
    # putting "Trace endpoint (dust amount)" on a plain unknown address
    # is noise — the bubble only makes sense when the dust lands on an
    # identified service, because that's the part of the story worth
    # surfacing ("dust went to <exchange/bridge/OTC>"). For a blank
    # intermediate leaf we just leave the node unlabeled.
    _SERVICE_ROLES_FOR_DUST_LABEL = {
        "cex_deposit", "bridge_service", "dex_service",
        "otc_service", "unidentified_service",
    }

    def _is_service_or_identified(addr: str) -> bool:
        entity = address_to_entity.get(addr)
        if entity is None:
            return False
        if entity.role in _SERVICE_ROLES_FOR_DUST_LABEL:
            return True
        # ``owner`` lives in the legacy label block for some entities —
        # treat any non-empty owner-like label as "identified".
        if entity.labels:
            meaningful = [lb for lb in entity.labels if lb not in _META_LABELS]
            if meaningful:
                return True
        return False

    for addr, reason in terminal_stop_reasons.items():
        if addr in service_comment_map:
            continue
        entity = address_to_entity.get(addr)
        if entity and entity.role in {"victim", "perpetrator"}:
            continue
        # Dust-endpoint label is a service-specific signal; if the leaf
        # is a plain unknown address, don't add a comment at all.
        reason_lower = (reason or "").lower()
        if "dust" in reason_lower and not _is_service_or_identified(addr):
            continue
        service_comment_map[addr] = f"«ren»{ren_counter}"
        ren_counter += 1

    def _intermediate_label_for(addr: str) -> str:
        """Pick a human-readable label for a terminal ``intermediate`` node
        using the path's stop_reason when available."""
        reason = (terminal_stop_reasons.get(addr) or "").lower()
        if not reason:
            return "Destination address"
        if "dead end" in reason or "no outgoing" in reason:
            return "Trace endpoint\n(no outflows found)"
        if "max hop" in reason:
            return "Trace endpoint\n(hop limit)"
        if "loop" in reason:
            return "Trace endpoint\n(loop)"
        if "cap" in reason:
            return "Trace endpoint\n(cap reached)"
        if "dust" in reason:
            return "Trace endpoint\n(dust amount)"
        return "Destination address"

    # --- Add role labels as comments (victim/perp/service) ---
    role_labels = {
        "victim": "Victim's address",
        "perpetrator": "Perpetrator's address",
        "bridge_service": "Bridge service",
        "cex_deposit": "Exchange deposit address",
        "otc_service": "OTC service",
        "unidentified_service": "Suspected unidentified service",
        "intermediate": "Destination address",
    }
    _comment_emitted: set[str] = set()
    for entity in trace_result.entities:
        if entity.address not in service_comment_map:
            continue
        # One comment per address — postprocess may have created several
        # Entity rows for the same address on different chains.
        if entity.address in _comment_emitted:
            continue
        # Use the authoritative entity (highest role priority) so the label
        # reflects e.g. ``bridge_service`` rather than the auto-added
        # ``intermediate`` twin.
        entity = _best_entity_by_addr.get(entity.address, entity)
        _comment_emitted.add(entity.address)
        comment_desc = service_comment_map[entity.address]
        # Anchor the comment on a node that actually exists among the items.
        # ``entity.chain`` is frozen at trace-entry (often the seed chain),
        # and ``case_meta.asset_symbol`` picks the wrong token_id for
        # destination-chain nodes on a cross-chain bridge, which left the
        # «ren» connector dangling at (0,0) with a default-position offset.
        entity_chain_norm = _normalize_chain(entity.chain)
        addr_prefix = f"{entity.address}-"
        real_descriptors = sorted(d for d in added_descriptors if d.startswith(addr_prefix))
        own_chain_prefix = f"{entity.address}-{entity_chain_norm}-"
        matching = [d for d in real_descriptors if d.startswith(own_chain_prefix)]
        if matching:
            address_desc = matching[0]
        elif real_descriptors:
            address_desc = real_descriptors[0]
        else:
            token_id = token_id_map.get((entity_chain_norm, (trace_result.case_meta.asset_symbol or "").upper()), 0)
            address_desc = _get_descriptor(entity.address, entity_chain_norm, token_id)
        pos = positions.get(address_desc, {"x": 0, "y": 0})
        real_labels = [lb for lb in (entity.labels or []) if lb not in _META_LABELS]
        # Prefer owner name from txList (e.g. "n.exchange") over generic role label
        txl_owner = owner_from_txlist.get(entity.address)
        owner_name = None
        if txl_owner:
            owner_name = txl_owner.get("name") if isinstance(txl_owner, dict) else str(txl_owner) if txl_owner else None
        if not owner_name and real_labels:
            owner_name = real_labels[0]

        # When the path stopped because of dust AND the leaf is a real
        # service, we keep the service identity AND tack on a
        # "(dust amount)" suffix so operators can tell "dust reached
        # Binance" apart from a regular full-flow Binance deposit.
        reason_for_addr = (terminal_stop_reasons.get(entity.address) or "").lower()
        dust_suffix = "\n(dust amount)" if "dust" in reason_for_addr else ""

        if owner_name and entity.role in {"cex_deposit", "bridge_service", "otc_service", "unidentified_service"}:
            role_suffix = role_labels.get(entity.role, "")
            core = f"{owner_name}\n{role_suffix}" if role_suffix else owner_name
            label = core + dust_suffix
        elif entity.role == "intermediate" and entity.address in terminal_stop_reasons:
            # Terminal intermediate: pick a label that reflects *why* the
            # trace stopped instead of pretending it's a CEX deposit.
            label = owner_name or _intermediate_label_for(entity.address)
        else:
            base = role_labels.get(entity.role) or (owner_name if owner_name else entity.role.replace("_", " ").title())
            label = base + dust_suffix
        has_owner_prefix = owner_name and entity.role in {"cex_deposit", "bridge_service", "otc_service", "unidentified_service"}
        line_count = label.count("\n") + 1
        if line_count >= 3 or (has_owner_prefix and line_count >= 2):
            comment_h = 65
        elif line_count == 2 or entity.role in {"bridge_service", "cex_deposit", "otc_service", "unidentified_service"}:
            comment_h = 50
        else:
            comment_h = 35
        comment_w = 136
        comment_x = pos["x"] - comment_w / 2.0
        comment_y = pos["y"] - comment_h - 25
        comments.append({
            "author": "User",
            "date": time.time(),
            "descriptor": comment_desc,
            "text": label,
            "type": "comment",
            "width": comment_w,
            "height": comment_h,
            "isManuallyMoved": False,
            "typeOfComment": "comment",
            "color": "#77869E",
            "x": comment_x,
            "y": comment_y
        })
        connects.append({
            "source": comment_desc,
            "target": address_desc,
            "data": {
                "color": "#C2C6CE",
                "type": "straight",
                "hovered": False
            }
        })

    helpers = {
        "isConnectionBasedMode": False,
        # Always emit merged structures (per-pair mergedEdge + mergedTx
        # hubs are built unconditionally above), and flag the payload
        # so the frontend can pick the consolidated view by default.
        # Operators can still toggle back to the expanded per-tx view —
        # the individual ``txEth``/``tx`` connectors stay in ``connects``
        # untouched.
        "isMergedTxMode": bool(edge_pairs),
        "isFiatMode": False,
        "isShowDate": False,
        "isHelperLinesDisabled": False,
        "bridgeHistory": [],
        "addressInfo": address_info_data if address_info_data else {},
        "labels": [],
        "blockList": [],
        "autoTxs": auto_txs,
        "interactionTxsStatsList": {},
        "commentSettings": {
            "defaultType": "comment",
            "defaultLineType": "straight",
            "defaultCommentColor": "#77869E",
            "defaultTxCommentColor": "#C2C6CE",
            "defaultSymbol": "$"
        },
        "prevReportAddressData": None,
        "txList": tx_list_inputs,
        "currencyInfo": list(currency_info.values())
    }

    default_title = f"Trace: {trace_result.case_meta.description[:30]}..." if trace_result.case_meta.description else f"Trace {trace_result.case_meta.case_id}"

    # Final summary logging
    logger.info(f"✅ Visualization built: {len(items)} items, {len(connects)} connections, {len(txs_output)} transactions")
    logger.info(f"📦 Currencies: {[c['symbol'] for c in currency_info.values()]}")
    logger.debug(f"Currency Info: {list(currency_info.values())}")
    logger.debug(f"AutoTxs count: {len(auto_txs)}")

    return {
        "createdAt": int(time.time() * 1000),
        "title": title or default_title,
        "type": "address",
        "thumbnail": "",
        "hash": None,
        "extras": {},
        "payload": payload,
        "helpers": helpers
    }
