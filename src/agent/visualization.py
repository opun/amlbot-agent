from typing import Dict, Any, List, Optional, Set, Tuple
import uuid
import logging
from collections import defaultdict, deque
from agent.models import TraceResult, Entity, Step, TraceStats

logger = logging.getLogger(__name__)

def _get_descriptor(address: str, chain: str, token_id: int = 0) -> str:
    """Generate a descriptor string for nodes/edges."""
    # Standard address descriptor
    return f"{address}-{chain}-{token_id}"

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

def generate_visualization_payload(trace_result: TraceResult, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate visualization payload from TraceResult.
    Format matches the expected structure (corrected.json).
    """
    
    # 1. Prepare data structures
    items = []
    connects = []
    txs = []
    comments = []
    currency_info = {}
    tx_list = []
    
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
            # We want to use the same descriptor for the same address/chain combo?
            # Service map is address -> descriptor. 
            return service_address_map[address]
        return _get_descriptor(address, chain, token_id)

    for path in trace_result.paths:
        for step in path.steps:
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
            # We treat amount valid if > 0
            val = step.amount_estimate or 0.0
            node_weights[src] += val
            node_weights[dst] += val
            
    # --- Pass 2: Compute Layout ---
    positions = _compute_positions(node_descriptors, edges, trace_result.case_meta.victim_address, service_descriptors, node_weights)

    # --- Pass 3: Generate Items & Comments ---
    added_descriptors = set()
    
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
                "author": "System",
                "date": 1764320753, 
                "descriptor": descriptor,
                "text": label,
                "type": "comment",
                "width": 126,
                "height": 50,
                "isManuallyMoved": False,
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
                "isManuallyMoved": False 
            })
        added_descriptors.add(descriptor)

    # --- Pass 4: Generate Edges & Txs ---
    for path in trace_result.paths:
        for i_step, step in enumerate(path.steps):
            chain = step.chain
            token_id = token_id_map.get((chain, step.asset), 0)
            
            src_desc = get_node_descriptor(step.from_address, chain, token_id)
            tgt_desc = get_node_descriptor(step.to_address, chain, token_id)
            
            add_node_or_comment(step.from_address, chain, token_id)
            add_node_or_comment(step.to_address, chain, token_id)
            
            is_service_connection = (src_desc in service_descriptors) or (tgt_desc in service_descriptors)
            
            src_pos = positions.get(src_desc, {"x": 0, "y": 0})
            tgt_pos = positions.get(tgt_desc, {"x": 0, "y": 0})
            mid_x = (src_pos["x"] + tgt_pos["x"]) / 2
            mid_y = (src_pos["y"] + tgt_pos["y"]) / 2
            y_offset = -40 if i_step % 2 == 0 else 40
            
            if is_service_connection:
                connects.append({
                    "source": src_desc,
                    "target": tgt_desc,
                    "data": {
                        "color": "#29793F",
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
                        "token_id": token_id,
                        "color": "#EC292C",
                        "isNew": True,
                        "fiatRate": 1.0, 
                        "isNeedReverse": False,
                        "hovered": False
                    }
                })
                
                tx_hash = step.tx_hash or f"tx-{uuid.uuid4().hex}"
                tx_desc = f"{tx_hash}-{chain}-{token_id}-{i_step}"
                
                txs.append({
                    "currency": chain,
                    "descriptor": tx_desc,
                    "hash": step.tx_hash,
                    "token_id": token_id,
                    "x": mid_x, 
                    "y": mid_y + y_offset,
                    "color": "#EC292C",
                    "path": "0",
                    "parentNode": f"{src_desc}{tgt_desc}", 
                    "type": "txEth"
                })
                
            # Rich metadata for TxList
            tx_list.append({
                "inputs": [{"address": step.from_address, "riskscore": 0.0}], 
                "outputs": [{"address": step.to_address, "riskscore": 0.0}],
                "hash": step.tx_hash,
                "fiatRate": 1.0,
                "addressesCount": 2,
                "amount": step.amount_estimate,
                "currency": chain,
                "tokenId": token_id,
                "poolTime": int(step.time.timestamp()) if hasattr(step.time, 'timestamp') else 0,
                "date": int(step.time.timestamp()) if hasattr(step.time, 'timestamp') else 0,
                "path": "0",
                "type": "txEth",
                # Extra metadata for UI
                "reasoning": step.reasoning,
                "step_type": step.step_type,
                "service_label": step.service_label,
                "direction": step.direction
            })
            
            if chain not in currency_info:
                currency_info[chain] = {
                    "currency": chain,
                    "issuer": None,
                    "name": step.asset,
                    "symbol": step.asset,
                    "token_id": token_id,
                    "unit": 6 
                }

    payload = {
        "comments": comments,
        "connects": connects,
        "items": items,
        "transform": {"k": 1, "x": 0, "y": 0},
        "txs": txs
    }
    
    helpers = {
        "currencyInfo": list(currency_info.values()),
        "txList": tx_list
    }
    
    default_title = f"Trace: {trace_result.case_meta.description[:30]}..." if trace_result.case_meta.description else f"Trace {trace_result.case_meta.case_id}"
    
    return {
        "title": title or default_title,
        "type": "address",
        "extras": {
             "trace_id": trace_result.case_meta.trace_id,
             "victim_address": trace_result.case_meta.victim_address
        },
        "payload": payload,
        "helpers": helpers
    }

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
         
    return abs(hash(f"{chain}:{asset}")) % 1000 + 1
