import asyncio
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from agents import gen_trace_id

# Setup logger for tracer
logger = logging.getLogger("tracer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[TRACE] %(message)s'))
    logger.addHandler(handler)

from agent.models import (
    TracerConfig, TraceResult, CaseMeta, Path, Step, Entity, Annotation, TraceStats
)
from agent.mcp_client import MCPClient
from agent.theft_detection import (
    infer_asset_symbol,
    fetch_candidate_theft_txs,
    choose_theft_tx,
    infer_approx_date_from_description,
    extract_victim_from_tx_hash
)
from agent.classification import AddressClassifier

class CryptoTracer:
    def __init__(self, client: MCPClient):
        self.client = client
        self.classifier = AddressClassifier(client)

    async def _resolve_to_address(self, from_addr: str, tx_hash: str, chain: str, token_id: int = 0, path_str: str = "0") -> Optional[str]:
        """Helper to resolve destination address using get-transaction."""
        try:
            detail = await self.client.get_transaction(from_addr, tx_hash, chain, token_id, path_str)

            # Handle text-wrapped JSON response
            if isinstance(detail, dict) and "text" in detail and isinstance(detail["text"], str):
                try:
                    import json
                    parsed_detail = json.loads(detail["text"])
                    detail = parsed_detail
                    logger.debug("  Successfully parsed nested JSON from text field")
                except json.JSONDecodeError:
                    logger.debug("  Failed to parse text field as JSON")

            # Handle response structure: could be {"success": true, "data": {...}} or just {"data": {...}}
            if detail.get("success") is True and "data" in detail:
                data = detail["data"]
            else:
                data = detail.get("data", detail)

            output = data.get("output", {})
            address = output.get("address")
            if address:
                logger.debug(f"  Resolved destination: {address}")
            else:
                logger.debug(f"  WARNING: get-transaction returned no output.address")
                logger.debug(f"    Response structure: success={detail.get('success')}, has_data={'data' in detail}")
                logger.debug(f"    Data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                logger.debug(f"    Output keys: {list(output.keys()) if isinstance(output, dict) else 'not a dict'}")
            return address
        except Exception as e:
            logger.debug(f"  ERROR resolving address: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"  Traceback: {traceback.format_exc()}")
            return None

    async def trace(self, config: TracerConfig) -> TraceResult:
        # 0. Setup
        case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        # Generate Trace ID for OpenAI Tracing
        trace_id = gen_trace_id()

        # Mode 2: Extract victim_address from tx_hash if needed
        if not config.victim_address and config.tx_hash:
            logger.debug(f"Extracting victim address from tx_hash: {config.tx_hash}")
            victim_addr, extracted_token_id, extracted_asset, block_time = await extract_victim_from_tx_hash(
                config.tx_hash, config.blockchain_name, self.client
            )
            config.victim_address = victim_addr
            if extracted_token_id is not None:
                # Store extracted token_id for later use
                config._extracted_token_id = extracted_token_id
            if extracted_asset and not config.asset_symbol:
                config.asset_symbol = extracted_asset
            if block_time and not config.approx_date:
                # Convert block_time to date string
                try:
                    dt = datetime.fromtimestamp(block_time)
                    config.approx_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        # Ensure victim_address is set
        if not config.victim_address:
            raise ValueError("victim_address is required. Either provide it directly or provide tx_hash to extract it.")

        # Auto-detect approx date if missing
        if not config.approx_date and config.description:
            config.approx_date = infer_approx_date_from_description(config.description)

        # Auto-detect asset if missing
        asset_symbol, token_id = await infer_asset_symbol(config, self.client)
        config.asset_symbol = asset_symbol # Update config with detected asset

        # Use extracted token_id if available
        if hasattr(config, '_extracted_token_id') and config._extracted_token_id is not None:
            token_id = config._extracted_token_id

        case_meta = CaseMeta(
            case_id=case_id,
            trace_id=trace_id,
            description=config.description or "",
            victim_address=config.victim_address,
            blockchain_name=config.blockchain_name,
            chains=[config.blockchain_name],
            asset_symbol=asset_symbol,
            approx_date=config.approx_date
        )

        entities: Dict[Tuple[str, str], Entity] = {}
        paths: List[Path] = []
        annotations: List[Annotation] = []

        # Helper to add entity
        def add_entity(e: Entity):
            entities[(e.address, e.chain)] = e

        # 1. Theft Selection
        candidates = await fetch_candidate_theft_txs(config, asset_symbol, token_id, self.client)

        if not candidates:
            # No candidates
            add_entity(Entity(
                address=config.victim_address,
                chain=config.blockchain_name,
                role="victim",
                labels=["Victim"],
                notes="No outgoing theft tx found."
            ))
            annotations.append(Annotation(
                id="no-theft",
                label="No Theft",
                related_addresses=[config.victim_address],
                related_steps=[],
                text="No outgoing transaction found matching criteria."
            ))
            return TraceResult(
                case_meta=case_meta,
                paths=[],
                entities=list(entities.values()),
                annotations=annotations,
                trace_stats=TraceStats(
                    initial_amount_estimate=0,
                    max_hops=config.max_hops,
                    max_branches_per_hop=config.max_branches_per_hop,
                    min_amount_ratio=config.min_amount_ratio,
                    explored_paths=0,
                    terminated_reason="no_candidate_tx"
                )
            )

        chosen_tx = choose_theft_tx(candidates, config.approx_date)
        initial_amount = chosen_tx["amount"]

        # Update token_id to use the actual transaction's token_id from chosen theft tx
        # This ensures we filter subsequent transactions by the correct token_id
        token_id = chosen_tx.get("token_id", token_id)
        logger.debug(f"Using token_id from chosen theft tx: {token_id} (tx: {chosen_tx['tx_hash'][:16]}...)")

        # Resolve 'to' address for chosen tx
        logger.debug(f"Resolving destination address for theft tx {chosen_tx['tx_hash'][:16]}...")
        to_address = await self._resolve_to_address(
            config.victim_address,
            chosen_tx["tx_hash"],
            config.blockchain_name,
            chosen_tx.get("token_id", token_id), # CHANGED: Use chosen_tx token_id
            chosen_tx.get("path", "0")
        )

        if not to_address:
            logger.debug(f"FAILED to resolve destination address - returning with empty paths")
            annotations.append(Annotation(
                id="resolve-fail",
                label="Error",
                related_addresses=[config.victim_address],
                related_steps=[],
                text=f"Could not resolve destination for theft tx {chosen_tx['tx_hash']}"
            ))
            return TraceResult(
                case_meta=case_meta,
                paths=[],
                entities=list(entities.values()),
                annotations=annotations,
                trace_stats=TraceStats(
                    initial_amount_estimate=initial_amount,
                    max_hops=config.max_hops,
                    max_branches_per_hop=config.max_branches_per_hop,
                    min_amount_ratio=config.min_amount_ratio,
                    explored_paths=0,
                    terminated_reason="resolve_fail"
                )
            )

        # 2. Initial Path
        path_id = "path-1"
        first_step = Step(
            step_index=0,
            from_address=config.victim_address,
            to_address=to_address,
            tx_hash=chosen_tx["tx_hash"],
            chain=config.blockchain_name,
            asset=asset_symbol,
            amount_estimate=initial_amount,
            time=chosen_tx.get("time"),
            direction="out",
            step_type="direct_transfer"
        )

        paths.append(Path(path_id=path_id, description="Main flow", steps=[first_step]))

        add_entity(Entity(
            address=config.victim_address,
            chain=config.blockchain_name,
            role="victim",
            labels=["Victim"],
            notes="Source of funds"
        ))

        # First hop entity (will be re-classified in loop)
        add_entity(Entity(
            address=to_address,
            chain=config.blockchain_name,
            role="intermediate",
            labels=[],
            notes="First hop"
        ))

        annotations.append(Annotation(
            id="victim-theft",
            label="Theft",
            related_addresses=[config.victim_address, to_address],
            related_steps=[f"{path_id}:0"],
            text=f"Funds ({initial_amount} {asset_symbol}) moved to {to_address}."
        ))

        # 3. Multi-Hop Loop
        logger.debug(f"")
        logger.debug(f"{'='*60}")
        logger.debug(f"Starting multi-hop trace from {to_address}")
        logger.debug(f"  Initial path created: {path_id} with {len(paths[0].steps)} step(s)")
        logger.debug(f"  Queue initialized with 1 path to explore")
        logger.debug(f"{'='*60}")

        queue = [(paths[0], first_step)]
        explored_paths = 0

        while queue:
            current_path, last_step = queue.pop(0)
            explored_paths += 1

            if len(current_path.steps) >= config.max_hops:
                logger.debug(f"  MAX HOPS REACHED ({config.max_hops}) - Terminating path")
                continue

            current_addr = last_step.to_address
            current_chain = last_step.chain
            current_amt = last_step.amount_estimate

            # 3.1 Classify
            logger.debug(f"")
            logger.debug(f"{'='*60}")
            logger.debug(f"Processing: {current_addr}")
            logger.debug(f"  Path: {current_path.path_id}, Step: {last_step.step_index}")
            logger.debug(f"  Amount received: {current_amt} {asset_symbol}")
            logger.debug(f"{'='*60}")

            context = {
                "amount_in": current_amt,
                "hop_index": last_step.step_index,
                "is_victim": False
            }
            entity = await self.classifier.classify(current_addr, current_chain, asset_symbol, context)
            add_entity(entity)
            logger.debug(f"  Classification: role={entity.role}, labels={entity.labels}")

            # 3.2 Check Bridge
            if entity.role == "bridge_service":
                # Check if it's a supported bridge protocol
                supported_bridges = ["layerzero", "wormhole", "multichain", "allbridge", "bridge", "generic"]
                # Check labels (which contain platform names) for supported protocols
                is_supported = any(b in l.lower() for l in entity.labels for b in supported_bridges)

                if is_supported:
                    logger.debug(f"  Bridge detected ({entity.labels}), attempting analysis...")
                    try:
                        bridge_res = await self.client.bridge_analyzer(current_chain, last_step.tx_hash)
                        res_data = bridge_res.get("result", bridge_res)

                        if res_data.get("is_bridge"):
                            logger.debug(f"  Bridge analysis SUCCESS: {res_data.get('protocol')} -> {res_data.get('dst_chain')}")

                            # Update current step info
                            last_step.step_type = "bridge_in"
                            last_step.protocol = res_data.get("protocol")
                            last_step.service_label = f"Bridge to {res_data.get('dst_chain')}"

                            dst_chain = res_data.get("dst_chain")
                            dst_addr = res_data.get("destination_address")

                            if dst_chain and dst_addr:
                                # We found the destination!
                                # Note: We might need to fetch incoming txs on dst chain to get exact time/hash,
                                # but for now let's assume we can continue trace from here.

                                # However, bridge_analyzer usually gives us the OUTGOING tx on the other side?
                                # Or just the address?
                                # If we just have address, we should search for the incoming tx there.

                                dst_filter = {
                                    "delta_coerced": [{">=": 0.0001}],
                                    "amount_coerced": [{">": 0}]
                                }
                                # Pass direction and order explicitly
                                dst_txs_res = await self.client.all_txs(
                                    dst_addr,
                                    dst_chain,
                                    dst_filter,
                                    limit=20,
                                    direction="asc",
                                    order="time"
                                )
                                dst_txs = dst_txs_res.get("data", [])

                                bridge_out_tx = None
                                bridge_out_time = None
                                best_diff = float('inf')

                                for dtx in dst_txs:
                                    amt = abs(float(dtx.get("delta_coerced") or dtx.get("amount_coerced") or 0))
                                    diff = abs(amt - current_amt)
                                    if diff < best_diff and diff / current_amt < 0.1:
                                        best_diff = diff
                                        bridge_out_tx = dtx.get("hash")
                                        bridge_out_time = dtx.get("block_time")

                                # If we didn't find a matching tx, we might still want to continue from the address?
                                # Let's assume yes, using current time if needed, but better to find the tx.

                                new_step = Step(
                                    step_index=last_step.step_index + 1,
                                    from_address=current_addr,
                                    to_address=dst_addr,
                                    tx_hash=bridge_out_tx, # Might be None if not found yet
                                    chain=dst_chain,
                                    asset=asset_symbol, # Asset might change? For now assume same symbol
                                    amount_estimate=current_amt,
                                    time=bridge_out_time or last_step.time, # Fallback to previous time
                                    direction="in",
                                    step_type="bridge_out",
                                    protocol=res_data.get("protocol")
                                )
                                current_path.steps.append(new_step)

                                add_entity(Entity(
                                    address=dst_addr,
                                    chain=dst_chain,
                                    role="intermediate",
                                    labels=["Bridge Receiver"],
                                    notes=f"Received from {current_chain} via {res_data.get('protocol')}"
                                ))

                                queue.append((current_path, new_step))
                                continue # Successfully processed bridge, skip termination
                        else:
                            logger.debug(f"  Bridge analysis returned is_bridge=False")
                    except Exception as e:
                        logger.debug(f"  Bridge analysis failed: {e}")

            # 3.3 Check Termination
            # If it was a bridge service but analysis failed or wasn't supported, it falls through here.
            if entity.role in ["cex_deposit", "otc_service", "unidentified_service", "bridge_service"]:
                logger.debug(f"  ENDPOINT REACHED - Terminating path")
                logger.debug(f"    Reason: Address classified as '{entity.role}'")

                # If bridge/swap, mention it explicitly in annotation
                note = f"Funds reached endpoint: {entity.role}"
                if entity.role == "bridge_service":
                    note = f"Funds reached Bridge/Swap/DEX service: {', '.join(entity.labels)}"

                annotations.append(Annotation(
                    id=f"term-{uuid.uuid4().hex[:4]}",
                    label="Endpoint",
                    related_addresses=[current_addr],
                    related_steps=[f"{current_path.path_id}:{last_step.step_index}"],
                    text=note
                ))
                continue

            # 3.4 Find Next Hops
            logger.debug(f"=== Finding next hops for {current_addr} (step {last_step.step_index}) ===")
            logger.debug(f"  Current amount to trace: {current_amt} {asset_symbol}")
            logger.debug(f"  Last step time (raw): {last_step.time}")

            out_filter = {
                "amount_coerced": [{">": 0}]
            }
            if last_step.time:
                 try:
                     if isinstance(last_step.time, str):
                         dt = datetime.fromisoformat(last_step.time.replace("Z", "+00:00"))
                         ts = int(dt.timestamp())
                         logger.debug(f"  Parsed time from ISO string: {last_step.time} -> timestamp {ts}")
                     else:
                         ts = int(last_step.time) if last_step.time else 0
                         logger.debug(f"  Using raw timestamp: {ts}")
                     if ts > 0:
                         out_filter["block_time"] = {">=": ts}
                         # Convert timestamp to readable date for logging
                         readable_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                         logger.debug(f"  Filter: block_time >= {ts} ({readable_time})")
                         logger.debug(f"  Reason: Only looking for outgoing txs AFTER funds arrived")
                 except Exception as e:
                     logger.debug(f"  Failed to parse time: {e}")

            logger.debug(f"  Full filter: {out_filter}")
            logger.debug(f"  Calling all_txs for withdrawals...")

            # Pass direction and order explicitly
            out_res = await self.client.all_txs(
                current_addr,
                current_chain,
                out_filter,
                limit=50,
                direction="asc",
                order="time",
                transaction_type="withdrawal"
            )
            out_txs = out_res.get("data", [])
            logger.debug(f"  API returned {len(out_txs)} withdrawal transactions")

            candidates = []
            threshold = initial_amount * config.min_amount_ratio
            logger.debug(f"  Filtering txs: threshold = {threshold:.2f} (initial_amount {initial_amount} * min_ratio {config.min_amount_ratio})")
            if token_id != 0:
                logger.debug(f"  Token filter: only accepting token_id = {token_id}")

            skipped_small = 0
            skipped_token = 0

            for tx in out_txs:
                amt = abs(float(tx.get("delta_coerced") or tx.get("amount_coerced") or 0))
                if amt < threshold:
                    skipped_small += 1
                    continue

                tx_token_id = tx.get("token_id", 0)
                if token_id != 0 and tx_token_id != token_id:
                    skipped_token += 1
                    continue

                # Optimization: Just store raw candidate info, don't fetch details yet
                candidates.append({
                    "tx_hash": tx.get("hash"),
                    "amount": amt,
                    "time": tx.get("block_time"),
                    "path": tx.get("path", "0"),
                    "token_id": tx_token_id,
                    "to": None
                })

            logger.debug(f"  Filtering results: {len(candidates)} candidates passed")
            logger.debug(f"    - Skipped {skipped_small} txs (amount < {threshold:.2f})")
            logger.debug(f"    - Skipped {skipped_token} txs (wrong token_id, expected {token_id})")

            # Sort and Pick Top Branches
            # Sort by time (ascending) to capture flow in chronological order
            candidates.sort(key=lambda x: x.get("time", 0))

            if candidates:
                logger.debug(f"  Candidates sorted by time (chronological):")
                for i, c in enumerate(candidates[:5]):  # Show first 5
                    c_time = datetime.fromtimestamp(c["time"]).strftime('%Y-%m-%d %H:%M:%S') if c["time"] else "N/A"
                    logger.debug(f"    [{i}] {c['amount']:.2f} {asset_symbol} @ {c_time} | tx: {c['tx_hash'][:16]}...")
                if len(candidates) > 5:
                    logger.debug(f"    ... and {len(candidates) - 5} more")

            # GROUPING & SELECTION (Lazy Resolution)
            # We resolve addresses one by one and stop once we cover the target amount
            logger.debug(f"  Resolving and selecting hops to cover target amount: {current_amt:.2f} {asset_symbol}")

            grouped_hops = {} # Key: (to_address, token_id) -> data
            accumulated_out = 0.0
            target_amount = current_amt
            resolve_failed = 0

            for cand in candidates:
                # Check if we've already covered the amount
                # Note: We do this check BEFORE processing the current candidate to be safe,
                # but strictly we check after adding. However, to avoid over-fetching, if we are already full, stop.
                if accumulated_out >= target_amount:
                    logger.debug(f"    >> Accumulated {accumulated_out:.2f} >= target {target_amount:.2f}, stopping resolution")
                    break

                to_addr = await self._resolve_to_address(
                    current_addr,
                    cand["tx_hash"],
                    current_chain,
                    cand.get("token_id", token_id),
                    cand.get("path", "0")
                )

                if not to_addr:
                    resolve_failed += 1
                    continue

                accumulated_out += cand["amount"]

                key = (to_addr, cand.get("token_id", token_id))
                if key not in grouped_hops:
                    grouped_hops[key] = {
                        "to_address": to_addr,
                        "token_id": cand.get("token_id", token_id),
                        "total_amount": 0.0,
                        "max_time": 0,
                        "tx_hashes": [],
                        "first_tx_hash": cand["tx_hash"]
                    }

                group = grouped_hops[key]
                group["total_amount"] += cand["amount"]
                group["tx_hashes"].append(cand["tx_hash"])
                if cand.get("time", 0) > group["max_time"]:
                    group["max_time"] = cand.get("time", 0)

                logger.debug(f"    + Resolved {to_addr[:16]}... ({cand['amount']:.2f}) -> accumulated: {accumulated_out:.2f}")

            if resolve_failed > 0:
                logger.debug(f"    Failed to resolve {resolve_failed} destination addresses")

            # Convert to list and sort by time (for consistent output order)
            next_hops = list(grouped_hops.values())
            next_hops.sort(key=lambda x: x["max_time"])

            if not next_hops:
                logger.debug(f"  NO NEXT HOPS FOUND - marking as dead end")
                logger.debug(f"    Reason: No outgoing transactions met the criteria")
                annotations.append(Annotation(
                    id=f"deadend-{uuid.uuid4().hex[:4]}",
                    label="Dead End",
                    related_addresses=[current_addr],
                    related_steps=[f"{current_path.path_id}:{last_step.step_index}"],
                    text="No significant outgoing flow found."
                ))
                continue

            logger.debug(f"  FINAL SELECTION: {len(next_hops)} hop(s) covering {accumulated_out:.2f} {asset_symbol}")

            # Process Chosen Hops
            logger.debug(f"  Processing {len(next_hops)} selected hop(s)...")
            for i, hop in enumerate(next_hops):
                to_addr = hop["to_address"]

                # CYCLE DETECTION
                existing_addresses = set()
                for s in current_path.steps:
                    existing_addresses.add(s.from_address.lower())
                    existing_addresses.add(s.to_address.lower())

                if to_addr.lower() in existing_addresses:
                    logger.debug(f"    [{i}] SKIPPED - Cycle detected: {to_addr[:16]}... already in path")
                    continue

                logger.debug(f"    [{i}] Creating step to {to_addr[:16]}... ({hop['total_amount']:.2f} {asset_symbol})")

                if i == 0:
                    p = current_path
                else:
                    import copy
                    new_steps = copy.deepcopy(current_path.steps)
                    p = Path(
                        path_id=f"{current_path.path_id}-{i}",
                        description=f"Branch from step {last_step.step_index}",
                        steps=new_steps
                    )
                    paths.append(p)

                new_step = Step(
                    step_index=last_step.step_index + 1,
                    from_address=current_addr,
                    to_address=to_addr,
                    tx_hash=hop["first_tx_hash"], # Use first hash as representative
                    chain=current_chain,
                    asset=asset_symbol,
                    amount_estimate=hop["total_amount"],
                    time=hop["max_time"],
                    direction="out",
                    step_type="direct_transfer"
                )
                p.steps.append(new_step)

                add_entity(Entity(
                    address=to_addr,
                    chain=current_chain,
                    role="intermediate",
                    labels=[],
                    notes=""
                ))

                queue.append((p, new_step))

        logger.debug(f"")
        logger.debug(f"{'='*60}")
        logger.debug(f"Trace completed")
        logger.debug(f"  Total paths: {len(paths)}")
        logger.debug(f"  Total entities: {len(entities)}")
        logger.debug(f"  Total annotations: {len(annotations)}")
        for i, path in enumerate(paths):
            logger.debug(f"    Path {i+1} ({path.path_id}): {len(path.steps)} step(s)")
        logger.debug(f"{'='*60}")

        return TraceResult(
            case_meta=case_meta,
            paths=paths,
            entities=list(entities.values()),
            annotations=annotations,
            trace_stats=TraceStats(
                initial_amount_estimate=initial_amount,
                max_hops=config.max_hops,
                max_branches_per_hop=config.max_branches_per_hop,
                min_amount_ratio=config.min_amount_ratio,
                explored_paths=len(paths),
                terminated_reason="completed"
            )
        )
