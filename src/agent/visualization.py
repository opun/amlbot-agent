import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

from agent.models import Entity, TraceResult

logger = logging.getLogger(__name__)

def _normalize_chain(chain: str) -> str:
    c = (chain or "").lower()
    if c in {"tron", "trc", "trc20", "trx"}:
        return "trx"
    if c in {"ethereum", "eth"}:
        return "eth"
    if c in {"binance", "bsc", "bnb", "bep20"}:
        return "bnb"
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
    """
    Generate a deterministic token ID for the asset.
    Returns 0 for native assets, and a hash-based ID for tokens.
    """
    if not asset:
        return 0

    asset_upper = asset.upper()
    chain_upper = chain.upper()

    if asset_upper == chain_upper or asset_upper in ["ETH", "BTC", "TRX", "SOL", "MATIC", "BNB"]:
         return 0

    # Common known tokens adjustments
    if chain_upper == "TRX" and asset_upper == "USDT":
        return 9

    return abs(hash(f"{chain}:{asset}")) % 1000 + 1

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
    for tx in tx_list_inputs:
        h = tx.get("hash")
        if h:
            fiat_rate_by_hash[h] = float(tx.get("fiatRate") or tx.get("fiat_rate") or 1.0)

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
    for entity in trace_result.entities:
        if entity.role in {"victim", "perpetrator", "bridge_service", "cex_deposit", "otc_service", "unidentified_service"}:
            service_comment_map[entity.address] = f"«ren»{ren_counter}"
            ren_counter += 1

    # --- Pass 1: Build Graph Topology & Weights ---
    node_descriptors = set()
    service_descriptors = set(service_comment_map.values())
    edges = []

    token_id_map = {} # (chain, asset) -> int
    if tx_list_inputs:
        try:
            asset_hint = (trace_result.case_meta.asset_symbol or "").upper()
        except AttributeError:
            asset_hint = ""
        for tx in tx_list_inputs:
            chain = _normalize_chain(tx.get("currency"))
            token_id = tx.get("tokenId")
            if token_id is None:
                token_id = tx.get("token_id")
            if chain and token_id is not None and asset_hint:
                token_id_map[(chain, asset_hint)] = int(token_id)
    node_weights = defaultdict(float) # descriptor -> total volume

    def get_node_descriptor(address: str, chain: str, token_id: int) -> str:
        return _get_descriptor(address, chain, token_id)

    # Collect unique steps across all paths.  Paths share common prefixes so
    # the same (from, to, tx_hash) triple can appear many times; we keep only
    # the first occurrence to avoid duplicate edges in the visualization.
    _step_keys_seen: set[tuple[str, str, str]] = set()
    all_steps = []
    for path in trace_result.paths:
        for step in path.steps:
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
    tx_desc_seen: set[str] = set()

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
    # Ensure native TRX/ETH is present
    blockchain = trace_result.case_meta.blockchain_name
    if blockchain == "trx":
         currency_info[0] = {
            "currency": "trx",
            "issuer": None,
            "name": "TRON",
            "symbol": "trx",
            "token_id": 0,
            "unit": 6
         }

    # --- Pass 4: Generate Edges & Txs ---
    # Prepare for autoTxs: map address -> list of (step_index, type, hash, path)
    address_activity = defaultdict(list)

    for i_step, step in enumerate(all_steps):
        chain = _normalize_chain(step.chain)
        asset = (step.asset or "").upper()
        token_id = token_id_map.get((chain, asset), 0)

        src_desc = get_node_descriptor(step.from_address, chain, token_id)
        tgt_desc = get_node_descriptor(step.to_address, chain, token_id)

        add_node_or_comment(step.from_address, chain, token_id)
        add_node_or_comment(step.to_address, chain, token_id)

        src_pos = positions.get(src_desc, {"x": 0, "y": 0})
        tgt_pos = positions.get(tgt_desc, {"x": 0, "y": 0})

        # Basic edge color
        edge_color = "#EC292C"

        tx_hash = step.tx_hash or f"tx-{uuid.uuid4().hex}"
        tx_desc = tx_desc_by_hash.get(step.tx_hash) or f"{tx_hash}-{chain}-{token_id}-{i_step}"

        mid_x = (src_pos["x"] + tgt_pos["x"]) / 2
        mid_y = (src_pos["y"] + tgt_pos["y"]) / 2

        _UTXO_CHAINS = {"btc", "bch", "ltc"}
        is_utxo = chain in _UTXO_CHAINS
        tx_type = "tx" if is_utxo else "txEth"
        tx_path = None if is_utxo else "0"

        if tx_desc not in tx_desc_seen:
            txs_output.append({
                "currency": chain,
                "descriptor": tx_desc,
                "hash": tx_hash,
                "token_id": token_id,
                "x": mid_x,
                "y": mid_y,
                "color": edge_color,
                "path": tx_path,
                "type": tx_type
            })
            tx_desc_seen.add(tx_desc)

        step_amount = step.amount_estimate or 0
        fiat_rate = fiat_rate_by_hash.get(step.tx_hash, 1.0) if step.tx_hash else 1.0

        connects.append({
            "source": src_desc,
            "target": tx_desc,
            "data": {
                "currency": chain,
                "amount": None,
                "fiatRate": fiat_rate,
                "token_id": token_id,
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
                "amount": None,
                "fiatRate": fiat_rate,
                "token_id": token_id,
                "color": edge_color,
                "type": "straight",
                "isNew": True,
                "isNeedReverse": False,
                "hovered": False
            }
        })

        # Record activity for autoTxs
        # For Sender (OUT)
        address_activity[(step.from_address, chain, token_id)].append({
            "type": "out",
            "hash": step.tx_hash,
            "index": i_step,
            "path": "0"
        })
        # For Receiver (IN)
        address_activity[(step.to_address, chain, token_id)].append({
             "type": "in",
             "hash": step.tx_hash,
             "index": i_step,
             "path": "0"
        })

        # Populate helper txList if not provided
        if not use_provided_tx_list:
            tx_list_inputs.append({
                "inputs": [{"address": step.from_address, "riskscore": address_to_entity.get(step.from_address, Entity(address="",chain="",role="intermediate",risk_score=0.0)).risk_score or 0.0, "type": "address"}],
                "outputs": [{"address": step.to_address, "riskscore": address_to_entity.get(step.to_address, Entity(address="",chain="",role="intermediate",risk_score=0.0)).risk_score or 0.0, "type": "address"}],
                "hash": step.tx_hash,
                "fiatRate": fiat_rate,
                "addressesCount": 2,
                "amount": int((step.amount_estimate or 0) * 1e6) if chain == 'trx' else step.amount_estimate,
                "currency": chain,
                "tokenId": token_id,
                "poolTime": _get_timestamp(step.time),
                "date": _get_timestamp(step.time),
                "path": tx_path,
                "type": tx_type,
                # Extra metadata for UI
                "reasoning": step.reasoning,
                "step_type": step.step_type,
                "service_label": step.service_label,
                "direction": step.direction
            })

        if token_id not in currency_info:
            asset_upper = (step.asset or "").upper()
            _NATIVE_UNIT_MAP = {"btc": 8, "bch": 8, "ltc": 8, "eth": 9, "bnb": 9, "matic": 9}
            _TOKEN_UNIT_MAP = {
                "USDT": 6, "USDC": 6, "DAI": 18, "BUSD": 18,
                "WETH": 18, "WBTC": 8, "LINK": 18, "UNI": 18,
                "AAVE": 18, "GRT": 18, "MTL": 8,
            }
            _CURRENCY_NAMES = {
                "eth": "Ethereum", "btc": "Bitcoin", "trx": "TRON",
                "bnb": "BNB Chain", "matic": "Polygon", "bch": "Bitcoin Cash",
                "ltc": "Litecoin", "sol": "Solana",
            }
            is_native = (token_id == 0)
            if is_native:
                unit = _NATIVE_UNIT_MAP.get(chain, 6)
            else:
                unit = _TOKEN_UNIT_MAP.get(asset_upper, 6)
            if asset_upper == "USDT":
                name = "Tether USD"
                symbol = "USDT"
            elif is_native and chain in _CURRENCY_NAMES:
                name = _CURRENCY_NAMES[chain]
                symbol = chain
            else:
                name = step.asset or chain
                symbol = asset_upper if asset_upper else chain
            currency_info[token_id] = {
                "currency": chain,
                "issuer": None,
                "name": name,
                "symbol": symbol,
                "token_id": token_id,
                "unit": unit
            }

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
    # nodes and add them to service_comment_map if missing.
    terminal_addresses = set()
    for path in trace_result.paths:
        if path.steps:
            terminal_addresses.add(path.steps[-1].to_address)

    for addr in terminal_addresses:
        if addr in service_comment_map:
            continue
        entity = address_to_entity.get(addr)
        if entity and entity.role in {"victim", "perpetrator"}:
            continue
        service_comment_map[addr] = f"«ren»{ren_counter}"
        ren_counter += 1
        if entity and entity.role == "intermediate":
            entity.role = "cex_deposit"

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
    for entity in trace_result.entities:
        if entity.address not in service_comment_map:
            continue
        comment_desc = service_comment_map[entity.address]
        token_id = token_id_map.get((_normalize_chain(entity.chain), (trace_result.case_meta.asset_symbol or "").upper()), 0)
        address_desc = _get_descriptor(entity.address, _normalize_chain(entity.chain), token_id)
        pos = positions.get(address_desc, {"x": 0, "y": 0})
        real_labels = [lb for lb in (entity.labels or []) if lb not in _META_LABELS]
        # Prefer owner name from txList (e.g. "n.exchange") over generic role label
        txl_owner = owner_from_txlist.get(entity.address)
        owner_name = None
        if txl_owner:
            owner_name = txl_owner.get("name") if isinstance(txl_owner, dict) else str(txl_owner) if txl_owner else None
        if not owner_name and real_labels:
            owner_name = real_labels[0]

        if owner_name and entity.role in {"cex_deposit", "bridge_service", "otc_service", "unidentified_service"}:
            role_suffix = role_labels.get(entity.role, "")
            label = f"{owner_name}\n{role_suffix}" if role_suffix else owner_name
        else:
            label = role_labels.get(entity.role) or (owner_name if owner_name else entity.role.replace("_", " ").title())
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
        "isMergedTxMode": False,
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
