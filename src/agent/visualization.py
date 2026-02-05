from typing import Dict, Any, List, Optional, Set, Tuple
import uuid
import logging
from collections import defaultdict, deque
from datetime import datetime
from agent.models import TraceResult, Entity, Step, TraceStats

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

def _normalize_tx_descriptor(desc: str, chain: str, token_id: Optional[int]) -> str:
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
        except Exception:
            pass
        # Try ISO date/time
        try:
            return int(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
        except Exception:
            return 0
    return 0

def _is_service(entity: Optional[Entity]) -> bool:
    """Check if entity should be visualized as a service comment block."""
    if not entity:
        return False
    # Roles that are services
    SERVICE_ROLES = {"cex_deposit", "bridge_service", "otc_service", "unidentified_service"}
    return entity.role in SERVICE_ROLES

def _compute_positions(nodes: Set[str], edges: List[Tuple[str, str]], victim_address: str, service_descriptors: Set[str], node_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Compute x,y positions for nodes using a layered graph layout.
    Nodes are sorted vertically by their weight (volume) to highlight important paths.
    """
    # Build adjacency list
    adj = defaultdict(list)
    in_degree = defaultdict(int) 
    
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
        if u not in in_degree:
            in_degree[u] = 0

    # Identify roots
    queue = deque()
    visited = {} # descriptor -> level
    
    roots = []
    # prioritizing victim
    for n in nodes:
        if victim_address.lower() in n.lower() and n not in service_descriptors:
             roots.append(n)
             
    if not roots:
         # Highest value nodes as fallback roots? or 0-degree
         roots = [n for n in nodes if in_degree[n] == 0]
    if not roots and nodes:
         roots = [next(iter(nodes))]
         
    for root in roots:
        queue.append((root, 0))
        visited[root] = 0
        
    # BFS for Layer Assignment
    max_level = 0
    while queue:
        u, level = queue.popleft()
        max_level = max(max_level, level)
        
        for v in adj[u]:
            if v not in visited:
                visited[v] = level + 1
                queue.append((v, level + 1))
            
    # Handle disconnected
    leftovers = [n for n in nodes if n not in visited]
    for n in leftovers:
        visited[n] = 0

    # Group by Level
    levels = defaultdict(list)
    for node, level in visited.items():
        levels[level].append(node)
        
    # Assign Coordinates
    positions = {}
    X_GAP = 350
    Y_GAP = 120
    
    for level, level_nodes in levels.items():
        # Sort nodes by weight (descending), then name
        # Heavier nodes (more volume) appear at the top -> or Center? 
        # Standard flow often puts main line in middle or top. 
        # Let's do Descending Weight -> Top to Bottom.
        level_nodes.sort(key=lambda n: (-node_weights.get(n, 0.0), n))
        
        count = len(level_nodes)
        start_y = -((count - 1) * Y_GAP) / 2
        
        for i, node in enumerate(level_nodes):
            positions[node] = {
                "x": level * X_GAP,
                "y": start_y + (i * Y_GAP)
            }
            
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
    title: Optional[str] = None,
    tx_list: Optional[List[Dict[str, Any]]] = None,
    txs: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
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
                except Exception:
                    pass
            if "token_id" in norm and norm["token_id"] is not None:
                try:
                    norm["token_id"] = int(norm["token_id"])
                except Exception:
                    pass
            tx_list_inputs.append(norm)
    use_provided_tx_list = bool(tx_list_inputs)
    
    address_to_entity = {e.address: e for e in trace_result.entities}
    
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
        except Exception:
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

    # We need to collect all steps to build graph
    all_steps = []
    for path in trace_result.paths:
        all_steps.extend(path.steps)

    tx_desc_by_hash = {}
    if txs:
        normalized_txs = []
        for tx in txs:
            norm = dict(tx)
            chain = _normalize_chain(norm.get("currency"))
            norm["currency"] = chain
            token_id = norm.get("token_id")
            if token_id is None:
                token_id = norm.get("tokenId")
            if token_id is not None:
                try:
                    token_id = int(token_id)
                except Exception:
                    pass
                norm["token_id"] = token_id
            desc = norm.get("descriptor")
            if desc:
                norm["descriptor"] = _normalize_tx_descriptor(desc, chain, token_id)
            normalized_txs.append(norm)
            tx_hash = norm.get("hash")
            tx_desc = norm.get("descriptor")
            if tx_hash and tx_desc:
                tx_desc_by_hash[tx_hash] = tx_desc
        txs = normalized_txs
    tx_desc_seen = set(tx_desc_by_hash.values())

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
    positions = _compute_positions(node_descriptors, edges, trace_result.case_meta.victim_address, service_descriptors, node_weights)

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
        
        risk_score = entity.risk_score if entity else 0.0
        owner = None
        if entity and entity.labels:
             owner = {
                 "id": 0,
                 "name": entity.labels[0],
                 "slug": entity.labels[0], 
                 "type": "exchange_licensed" if "exchange" in (entity.role or "") else "unknown",
                 "subtype": None
             }
             
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
            "isManuallyMoved": True
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

    if txs:
        txs_output = list(txs)

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
        y_offset = -40 if i_step % 2 == 0 else 40
        if src_desc == tgt_desc:
            y_offset = 0

        if tx_desc not in tx_desc_seen:
            txs_output.append({
                "currency": chain,
                "descriptor": tx_desc,
                "hash": step.tx_hash,
                "token_id": token_id,
                "x": mid_x, 
                "y": mid_y + y_offset,
                "color": edge_color,
                "path": "0", 
                "type": "txEth"
            })
            tx_desc_seen.add(tx_desc)

        connects.append({
            "source": src_desc,
            "target": tx_desc,
            "data": {
                "currency": chain,
                "amount": None,
                "fiatRate": 1.0, 
                "token_id": token_id,
                "color": edge_color,
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
                "fiatRate": 1.0, 
                "token_id": token_id,
                "color": edge_color,
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
                "fiatRate": 1.0,
                "addressesCount": 2,
                "amount": int((step.amount_estimate or 0) * 1e6) if chain == 'trx' else step.amount_estimate, 
                "currency": chain,
                "tokenId": token_id,
                "poolTime": _get_timestamp(step.time),
                "date": _get_timestamp(step.time),
                "path": "0",
                "type": "txEth",
                # Extra metadata for UI
                "reasoning": step.reasoning,
                "step_type": step.step_type,
                "service_label": step.service_label,
                "direction": step.direction
            })
        
        if token_id not in currency_info:
            asset_upper = (step.asset or "").upper()
            currency_info[token_id] = {
                "currency": chain,
                "issuer": None, 
                "name": "Tether USD" if asset_upper == "USDT" else step.asset,
                "symbol": asset_upper if asset_upper else step.asset,
                "token_id": token_id,
                "unit": 6 
            }

    # --- Generate autoTxs ---
    auto_txs = []
    
    for (address, chain, token_id), activities in address_activity.items():
        if address in service_comment_map: continue # Skip service nodes for autoTxs?
        
        # Sort by step index
        activities.sort(key=lambda x: x["index"])
        
        for i, act in enumerate(activities):
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

    # --- Add role labels as comments (victim/perp/service) ---
    role_labels = {
        "victim": "Victim's address",
        "perpetrator": "Perpetrator's address",
        "bridge_service": "Bridge service",
        "cex_deposit": "Exchange deposit address",
        "otc_service": "OTC service",
        "unidentified_service": "Suspected unidentified service"
    }
    for entity in trace_result.entities:
        if entity.address not in service_comment_map:
            continue
        comment_desc = service_comment_map[entity.address]
        token_id = token_id_map.get((_normalize_chain(entity.chain), (trace_result.case_meta.asset_symbol or "").upper()), 0)
        address_desc = _get_descriptor(entity.address, _normalize_chain(entity.chain), token_id)
        pos = positions.get(address_desc, {"x": 0, "y": 0})
        label = role_labels.get(entity.role) or (entity.labels[0] if entity.labels else entity.role.replace("_", " ").title())
        comments.append({
            "author": "User",
            "date": time.time(),
            "descriptor": comment_desc,
            "text": label,
            "type": "comment",
            "width": 126,
            "height": 50 if entity.role in {"bridge_service", "cex_deposit", "otc_service", "unidentified_service"} else 35,
            "isManuallyMoved": True,
            "typeOfComment": "comment",
            "color": "#77869E",
            "x": pos["x"] + 90,
            "y": pos["y"] - 80
        })
        connects.append({
            "source": comment_desc,
            "target": address_desc,
            "data": {
                "color": "#C2C6CE",
                "type": "smoothstep",
                "hovered": False
            }
        })
    
    # Hardcoded helpers from example
    helpers = {
        "isConnectionBasedMode": False,
        "isMergedTxMode": False,
        "isFiatMode": False,
        "isShowDate": False,
        "isHelperLinesDisabled": False,
        "labels": [],
        "blockList": [],
        "autoTxs": auto_txs,
        "interactionTxsStatsList": {},
        "commentSettings": {
            "defaultType": "comment",
            "defaultLineType": "smoothstep",
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
