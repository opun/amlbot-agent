from typing import Dict, Any, List, Optional, Set, Tuple
import uuid
import logging
from collections import defaultdict, deque
from datetime import datetime
from agent.models import TraceResult, Entity, Step, TraceStats

logger = logging.getLogger(__name__)

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

def generate_visualization_payload(trace_result: TraceResult, title: Optional[str] = None) -> Dict[str, Any]:
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
    txs = []
    comments = []
    currency_info = {}
    tx_list_inputs = [] # Use separate list for helpers.txList
    
    address_to_entity = {e.address: e for e in trace_result.entities}
    
    service_address_map = {} 
    ren_counter = 0
    for entity in trace_result.entities:
        if _is_service(entity):
            service_address_map[entity.address] = f"«ren»{ren_counter}"
            ren_counter += 1

    # --- Pass 1: Build Graph Topology & Weights ---
    node_descriptors = set()
    service_descriptors = set(service_address_map.values())
    edges = []
    
    token_id_map = {} # (chain, asset) -> int
    node_weights = defaultdict(float) # descriptor -> total volume
    
    def get_node_descriptor(address: str, chain: str, token_id: int) -> str:
        if address in service_address_map:
            return service_address_map[address]
        return _get_descriptor(address, chain, token_id)

    # We need to collect all steps to build graph
    all_steps = []
    for path in trace_result.paths:
        all_steps.extend(path.steps)

    for step in all_steps:
        key = (step.chain, step.asset)
        if key not in token_id_map:
            token_id_map[key] = _get_token_id(step.asset, step.chain)
        
        token_id = token_id_map[key]
        
        src = get_node_descriptor(step.from_address, step.chain, token_id)
        dst = get_node_descriptor(step.to_address, step.chain, token_id)
        
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
        
        if descriptor in service_descriptors:
            label = "Unknown Service"
            if entity and entity.labels:
                label = entity.labels[0]
            elif entity and entity.role:
                label = entity.role.replace("_", " ").title()
            
            comments.append({
                "author": "User", 
                "date": time.time(),
                "descriptor": descriptor,
                "text": label,
                "type": "comment",
                "width": 126,
                "height": 50, 
                "isManuallyMoved": True,
                "typeOfComment": "comment",
                "color": "#77869E",
                "x": pos["x"],
                "y": pos["y"]
            })
        else:
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

    for i_step, step in enumerate(all_steps):
        chain = step.chain
        token_id = token_id_map.get((chain, step.asset), 0)
        
        src_desc = get_node_descriptor(step.from_address, chain, token_id)
        tgt_desc = get_node_descriptor(step.to_address, chain, token_id)
        
        add_node_or_comment(step.from_address, chain, token_id)
        add_node_or_comment(step.to_address, chain, token_id)
        
        is_service_connection = (src_desc in service_descriptors) or (tgt_desc in service_descriptors)
        
        src_pos = positions.get(src_desc, {"x": 0, "y": 0})
        tgt_pos = positions.get(tgt_desc, {"x": 0, "y": 0})
        
        # Basic edge color
        edge_color = "#EC292C" 
        
        if is_service_connection:
            connects.append({
                "source": src_desc,
                "target": tgt_desc,
                "data": {
                    "color": "#C2C6CE", 
                    "type": "smoothstep",
                    "hovered": False
                }
            })
        else:
            connects.append({
                "source": src_desc,
                "target": tgt_desc,
                "data": {
                    "currency": chain,
                    "amount": step.amount_estimate,
                    "fiatRate": 1.0, 
                    "token_id": token_id,
                    "color": edge_color,
                    "isNew": True,
                    "isNeedReverse": False,
                    "hovered": False
                }
            })
            
            tx_hash = step.tx_hash or f"tx-{uuid.uuid4().hex}"
            tx_desc = f"{tx_hash}-{chain}-{token_id}-{i_step}"
            
            mid_x = (src_pos["x"] + tgt_pos["x"]) / 2
            mid_y = (src_pos["y"] + tgt_pos["y"]) / 2
            y_offset = -40 if i_step % 2 == 0 else 40
            if src_desc == tgt_desc: y_offset = 0
            
            txs.append({
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

            # Populate helper txList
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
            currency_info[token_id] = {
                "currency": chain,
                "issuer": None, 
                "name": "Tether USD" if token_id == 9 else step.asset,
                "symbol": "USDT" if token_id == 9 else step.asset,
                "token_id": token_id,
                "unit": 6 
            }

    # --- Generate autoTxs ---
    auto_txs = []
    
    for (address, chain, token_id), activities in address_activity.items():
        if address in service_address_map: continue # Skip service nodes for autoTxs?
        
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
        "txs": txs
    }
    
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
    logger.info(f"✅ Visualization built: {len(items)} items, {len(connects)} connections, {len(txs)} transactions")
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
