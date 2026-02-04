import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from agent.models import (
    TracerConfig,
    TraceResult,
    CaseMeta,
    Path,
    Step,
    Entity,
    Annotation,
    TraceStats,
)
from agent.theft_detection import (
    infer_asset_symbol,
    infer_approx_date_from_description,
    extract_victim_from_tx_hash,
)
from agent.mcp_client import MCPClient
from agent.mcp_http_client import MCPHTTPClient

logger = logging.getLogger("deterministic_tracer")

AnyMCPClient = Union[MCPClient, MCPHTTPClient]

ALLOWED_STEP_TYPES = {
    "direct_transfer",
    "bridge_in",
    "bridge_out",
    "bridge_transfer",
    "bridge_arrival",
    "service_deposit",
    "internal_transfer",
}


@dataclass
class TxCandidate:
    tx_hash: str
    amount: float
    block_time: Optional[int]
    recipient: Optional[str]
    asset: Optional[str]
    token_id: Optional[int]


def _unwrap_tool_result(result: Any) -> Any:
    if isinstance(result, dict) and "text" in result and isinstance(result["text"], str):
        try:
            return json.loads(result["text"])
        except json.JSONDecodeError:
            return result
    if isinstance(result, dict) and "raw_output" in result and isinstance(result["raw_output"], str):
        try:
            return json.loads(result["raw_output"])
        except json.JSONDecodeError:
            return result
    return result


def _extract_list(result: Any, keys: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    for key in keys:
        value = result.get(key)
        if isinstance(value, list):
            return value
    return []


def _parse_amount(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return 0.0
    return 0.0


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _extract_owner_and_risk(aml_result: Any) -> Tuple[float, Dict[str, float], List[str], Optional[str]]:
    data = _unwrap_tool_result(aml_result)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    risk_score = 0.0
    signals: Dict[str, float] = {}
    owner_name = None
    labels: List[str] = []

    if isinstance(data, dict):
        riskscore = data.get("riskscore") or data.get("risk_score") or {}
        if isinstance(riskscore, dict):
            risk_score = _parse_amount(riskscore.get("value"))
            signals = riskscore.get("signals") or {}
        else:
            risk_score = _parse_amount(riskscore)

        owner = data.get("owner") or {}
        if isinstance(owner, dict):
            owner_name = owner.get("name") or owner.get("title")
            if owner_name:
                labels.append(str(owner_name))
            subtype = owner.get("subtype")
            if subtype:
                labels.append(str(subtype))

    return risk_score, signals, labels, owner_name


def _extract_service_platforms(extra_result: Any) -> List[str]:
    data = _unwrap_tool_result(extra_result)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        data = data["data"]
    services = {}
    if isinstance(data, dict):
        services = data.get("services") or {}
    use_platform = services.get("use_platform") if isinstance(services, dict) else None
    if isinstance(use_platform, list):
        return [str(x) for x in use_platform]
    return []


def _extract_transfer_fields(transfer: Dict[str, Any], chain: str) -> Tuple[Optional[str], Optional[str], float, Optional[int], Optional[int], Optional[str]]:
    input_data = transfer.get("input") or {}
    output_data = transfer.get("output") or {}

    if isinstance(input_data, dict):
        sender = input_data.get("address") or input_data.get("from")
    else:
        sender = transfer.get("from")

    if isinstance(output_data, dict):
        recipient = output_data.get("address") or output_data.get("to")
    else:
        recipient = transfer.get("to")

    amount = _parse_amount(
        transfer.get("amount")
        or transfer.get("amount_coerced")
        or transfer.get("value")
        or output_data.get("amount")
        if isinstance(output_data, dict)
        else None
    )

    token_id = transfer.get("token_id") or transfer.get("tokenId")
    token_id = _parse_int(token_id) or 0

    block_time = _parse_int(transfer.get("block_time") or transfer.get("time"))

    asset = (
        transfer.get("asset")
        or transfer.get("symbol")
        or transfer.get("token_symbol")
    )
    if not asset and isinstance(transfer.get("token"), dict):
        asset = transfer["token"].get("symbol") or transfer["token"].get("name")
    if not asset:
        asset = chain.upper()

    return sender, recipient, amount, token_id, block_time, asset


def _matches_keywords(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)


def _detect_role(
    owner_name: Optional[str],
    labels: List[str],
    service_platforms: List[str],
    hop_index: int,
    is_victim: bool,
    risk_score: float,
) -> Tuple[str, bool, Optional[str], Optional[str]]:
    owner_text = " ".join([x for x in [owner_name] + labels if x])
    services_text = " ".join(service_platforms)

    bridge_keywords = [
        "bridge", "swap", "dex", "uniswap", "sushiswap", "pancakeswap", "curve",
        "layerzero", "stargate", "allbridge", "wormhole", "router", "bridgers",
        "multichain", "synapse", "hop", "across"
    ]
    cex_keywords = [
        "exchange", "binance", "coinbase", "kraken", "huobi", "okx", "kucoin",
        "bitfinex", "mxc", "gate.io", "poloniex", "bybit"
    ]
    mixer_keywords = ["mixer", "tornado", "blender", "sinbad"]
    otc_keywords = ["otc"]

    combined = f"{owner_text} {services_text}".strip().lower()

    if is_victim:
        return "victim", False, None, None

    if _matches_keywords(combined, mixer_keywords):
        return "unidentified_service", True, "Mixer", None
    if _matches_keywords(combined, otc_keywords):
        return "otc_service", True, "OTC", None
    if _matches_keywords(combined, cex_keywords):
        return "cex_deposit", True, owner_name or (service_platforms[0] if service_platforms else "CEX"), None
    if _matches_keywords(combined, bridge_keywords):
        return "bridge_service", True, service_platforms[0] if service_platforms else owner_name, service_platforms[0] if service_platforms else None

    # Default
    if hop_index == 0:
        return "perpetrator", False, None, None
    return "intermediate", False, None, None


def _parse_date_to_timestamp(date_str: str) -> Optional[int]:
    try:
        dt = datetime.fromisoformat(date_str)
        return int(dt.timestamp())
    except Exception:
        return None


class RuleBasedTracer:
    def __init__(self, client: AnyMCPClient, max_hops: int = 8, max_paths: int = 20):
        self.client = client
        self.max_hops = max_hops
        self.max_paths = max_paths

    async def _get_outgoing_txs(
        self, address: str, chain: str, token_id: Optional[int], start_time: Optional[int]
    ) -> List[Dict[str, Any]]:
        filter_criteria: Dict[str, Any] = {}
        if start_time is not None:
            filter_criteria["time"] = {">=": start_time}
        if token_id is not None:
            filter_criteria["token_id"] = [token_id]

        result = await self.client.all_txs(
            address,
            chain,
            filter_criteria=filter_criteria if filter_criteria else None,
            limit=50,
            offset=0,
            direction="asc",
            order="time",
            transaction_type="withdrawal",
        )
        result = _unwrap_tool_result(result)
        txs = _extract_list(result, ["data", "result", "transactions", "txs", "items"])
        return txs

    async def _resolve_transfer_from_tx(self, tx_hash: str, chain: str) -> Optional[TxCandidate]:
        transfer_result = await self.client.token_transfers(tx_hash, chain)
        transfer_result = _unwrap_tool_result(transfer_result)
        transfers = _extract_list(transfer_result, ["data", "result", "transfers", "items"])
        if not transfers:
            return None

        # Choose transfer with largest amount
        best = None
        best_amount = -1.0
        for tr in transfers:
            sender, recipient, amount, token_id, block_time, asset = _extract_transfer_fields(tr, chain)
            if amount > best_amount and recipient:
                best_amount = amount
                best = (sender, recipient, amount, token_id, block_time, asset)

        if not best:
            return None

        _, recipient, amount, token_id, block_time, asset = best
        return TxCandidate(
            tx_hash=tx_hash,
            amount=amount,
            block_time=block_time,
            recipient=recipient,
            asset=asset,
            token_id=token_id,
        )

    async def _classify_address(
        self,
        address: str,
        chain: str,
        asset_symbol: str,
        hop_index: int,
        is_victim: bool,
    ) -> Tuple[Entity, bool, Optional[str], Optional[str], float]:
        aml_result = await self.client.get_address(chain, address)
        extra_result = await self.client.get_extra_address_info(address, asset_symbol)

        risk_score, signals, labels, owner_name = _extract_owner_and_risk(aml_result)
        service_platforms = _extract_service_platforms(extra_result)

        role, is_terminal, service_label, protocol = _detect_role(
            owner_name, labels, service_platforms, hop_index, is_victim, risk_score
        )

        # Add service platform labels
        for svc in service_platforms:
            if svc not in labels:
                labels.append(svc)

        if risk_score > 0.75 and "High Risk" not in labels:
            labels.append("High Risk")

        entity = Entity(
            address=address,
            chain=chain,
            role=role,
            risk_score=risk_score,
            riskscore_signals=signals or {},
            labels=labels,
            notes=None,
        )
        return entity, is_terminal, service_label, protocol, risk_score

    async def _resolve_token_id_for_address(
        self, chain: str, address: str, asset_symbol: str
    ) -> Optional[int]:
        try:
            stats = await self.client.token_stats(chain, address)
            stats = _unwrap_tool_result(stats)
            tokens = stats.get("tokens") if isinstance(stats, dict) else None
            if isinstance(tokens, list):
                for t in tokens:
                    if str(t.get("symbol", "")).upper() == asset_symbol.upper():
                        return _parse_int(t.get("token_id")) or 0
        except Exception:
            return None
        return None

    async def trace(self, config: TracerConfig) -> TraceResult:
        # Setup meta
        if not config.approx_date and config.description:
            config.approx_date = infer_approx_date_from_description(config.description)

        if not config.victim_address and config.tx_hash:
            victim_addr, extracted_token_id, extracted_asset, block_time = await extract_victim_from_tx_hash(
                config.tx_hash, config.blockchain_name, self.client
            )
            config.victim_address = victim_addr
            if extracted_asset and not config.asset_symbol:
                config.asset_symbol = extracted_asset
            if block_time and not config.approx_date:
                try:
                    dt = datetime.fromtimestamp(block_time)
                    config.approx_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        if not config.victim_address:
            raise ValueError("victim_address is required. Provide tx_hash or wallet address.")

        asset_symbol, detected_token_id = await infer_asset_symbol(config, self.client)
        config.asset_symbol = asset_symbol
        token_id = detected_token_id

        case_meta = CaseMeta(
            case_id=f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{datetime.now().microsecond}",
            trace_id=None,
            description=config.description or "",
            victim_address=config.victim_address,
            blockchain_name=config.blockchain_name,
            chains=[config.blockchain_name],
            asset_symbol=asset_symbol,
            approx_date=config.approx_date,
        )

        annotations: List[Annotation] = []
        entities: Dict[Tuple[str, str], Entity] = {}
        paths: List[Path] = []

        initial_amount = 0.0
        initial_time: Optional[int] = None
        initial_recipient: Optional[str] = None
        initial_tx_hash: Optional[str] = None
        initial_chain = config.blockchain_name

        if config.tx_hash:
            candidate = await self._resolve_transfer_from_tx(config.tx_hash, config.blockchain_name)
            if not candidate or not candidate.recipient:
                raise ValueError("Failed to resolve recipient from tx_hash via token-transfers.")
            initial_tx_hash = config.tx_hash
            initial_amount = candidate.amount
            initial_time = candidate.block_time
            initial_recipient = candidate.recipient
            if candidate.asset:
                asset_symbol = candidate.asset
                config.asset_symbol = candidate.asset
            if candidate.token_id is not None:
                token_id = candidate.token_id
        else:
            # Select theft transaction from victim outgoing
            txs = await self._get_outgoing_txs(config.victim_address, config.blockchain_name, token_id, None)
            if not txs:
                raise ValueError("No outgoing transactions found for victim address.")

            # Prefer txs near approx_date if provided
            filtered = txs
            if config.approx_date:
                approx_ts = _parse_date_to_timestamp(config.approx_date)
                if approx_ts:
                    window_start = approx_ts - int(timedelta(days=7).total_seconds())
                    window_end = approx_ts + int(timedelta(days=7).total_seconds())
                    filtered = [
                        t for t in txs
                        if _parse_int(t.get("block_time")) is not None
                        and window_start <= _parse_int(t.get("block_time")) <= window_end
                    ] or txs

            # Choose largest outgoing
            def tx_amount(t: Dict[str, Any]) -> float:
                return _parse_amount(t.get("amount") or t.get("amount_coerced") or t.get("value"))

            selected = max(filtered, key=tx_amount)
            initial_tx_hash = selected.get("hash") or selected.get("tx_hash")
            initial_time = _parse_int(selected.get("block_time"))
            initial_amount = tx_amount(selected)

            if not initial_tx_hash:
                raise ValueError("Could not determine theft transaction hash.")

            # Resolve recipient via token-transfers
            candidate = await self._resolve_transfer_from_tx(initial_tx_hash, config.blockchain_name)
            if not candidate or not candidate.recipient:
                raise ValueError("Failed to resolve recipient from theft tx.")
            initial_recipient = candidate.recipient
            if candidate.asset:
                asset_symbol = candidate.asset
                config.asset_symbol = candidate.asset
            if candidate.token_id is not None:
                token_id = candidate.token_id

        if not initial_recipient:
            raise ValueError("Could not determine initial recipient for trace.")

        initial_step = Step(
            step_index=0,
            from_address=config.victim_address,
            to_address=initial_recipient,
            tx_hash=initial_tx_hash,
            chain=config.blockchain_name,
            asset=asset_symbol,
            amount_estimate=initial_amount,
            time=initial_time,
            direction="out",
            step_type="direct_transfer",
            service_label=None,
            protocol=None,
            reasoning="Initial theft transfer identified from input data."
        )

        # Seed DFS stack
        stack = [
            {
                "path_id": "1",
                "steps": [initial_step],
                "current_address": initial_recipient,
                "incoming_amount": initial_amount,
                "incoming_time": initial_time,
                "chain": initial_chain,
                "token_id": token_id,
                "hop_index": 0,
                "incoming_tx_hash": initial_tx_hash,
                "visited": {(config.victim_address, initial_chain), (initial_recipient, initial_chain)},
            }
        ]
        path_counter = 1
        annotation_counter = 1
        chain_set = {config.blockchain_name}

        # Ensure victim entity
        victim_entity = Entity(
            address=config.victim_address,
            chain=config.blockchain_name,
            role="victim",
            risk_score=0.0,
            riskscore_signals={},
            labels=["Victim"],
            notes=None,
        )
        entities[(config.victim_address, config.blockchain_name)] = victim_entity

        while stack and len(paths) < self.max_paths:
            state = stack.pop()
            steps: List[Step] = state["steps"]
            current_address: str = state["current_address"]
            incoming_amount: float = state["incoming_amount"] or 0.0
            incoming_time: Optional[int] = state["incoming_time"]
            chain: str = state["chain"]
            token_id = state["token_id"]
            hop_index = state["hop_index"]
            incoming_tx_hash = state.get("incoming_tx_hash")
            visited = set(state["visited"])

            # Classify current address
            entity, is_terminal, service_label, protocol, risk_score = await self._classify_address(
                current_address,
                chain,
                asset_symbol,
                hop_index,
                is_victim=(current_address == config.victim_address),
            )
            entities[(current_address, chain)] = entity

            if risk_score > 0.75:
                annotations.append(Annotation(
                    id=f"ann-{annotation_counter}",
                    label="High Risk",
                    related_addresses=[current_address],
                    related_steps=[f"{state['path_id']}:{len(steps)-1}"],
                    text=f"Passed through high-risk address (score: {risk_score:.2f}) - continuing trace."
                ))
                annotation_counter += 1

            # Stop if max hops
            if hop_index >= self.max_hops:
                paths.append(Path(
                    path_id=state["path_id"],
                    description="Deterministic trace path",
                    steps=steps,
                    stop_reason="Max hops reached",
                ))
                continue

            # Bridge handling
            if entity.role == "bridge_service":
                bridge_info = None
                if incoming_tx_hash:
                    try:
                        if hasattr(self.client, "bridge_analyze"):
                            bridge_info = await self.client.bridge_analyze(chain, incoming_tx_hash)
                        else:
                            bridge_info = await self.client.bridge_analyzer(chain, incoming_tx_hash)
                        bridge_info = _unwrap_tool_result(bridge_info)
                    except Exception:
                        bridge_info = None

                dst_chain = None
                dst_addr = None
                amount_out = None
                if isinstance(bridge_info, dict):
                    if bridge_info.get("is_bridge"):
                        dst_chain = bridge_info.get("dst_chain") or bridge_info.get("dest_chain")
                        dst_addr = bridge_info.get("destination_address") or bridge_info.get("dst_address")
                        amount_out = _parse_amount(bridge_info.get("amount_out") or bridge_info.get("amount"))

                if dst_chain and dst_addr:
                    chain_set.add(dst_chain)
                    bridge_amount = amount_out if amount_out else incoming_amount
                    if amount_out and incoming_amount:
                        gap = abs(amount_out - incoming_amount) / max(incoming_amount, 1.0)
                        if gap > 0.2:
                            annotations.append(Annotation(
                                id=f"ann-{annotation_counter}",
                                label="Bridge Aggregation",
                                related_addresses=[current_address, dst_addr],
                                related_steps=[f"{state['path_id']}:{len(steps)-1}"],
                                text=f"Bridge aggregation detected - output amount ({amount_out}) differs from input ({incoming_amount})."
                            ))
                            annotation_counter += 1

                    step = Step(
                        step_index=len(steps),
                        from_address=current_address,
                        to_address=dst_addr,
                        tx_hash=bridge_info.get("dst_tx_hash") if isinstance(bridge_info, dict) else None,
                        chain=dst_chain,
                        asset=asset_symbol,
                        amount_estimate=bridge_amount,
                        time=_parse_int(bridge_info.get("dst_block_time") if isinstance(bridge_info, dict) else None) or incoming_time,
                        direction="out",
                        step_type="bridge_transfer",
                        service_label=service_label,
                        protocol=protocol,
                        reasoning="Bridge detected; continuing on destination chain."
                    )
                    steps.append(step)

                    # Continue on destination chain
                    new_token_id = await self._resolve_token_id_for_address(dst_chain, dst_addr, asset_symbol)
                    stack.append({
                        "path_id": state["path_id"],
                        "steps": steps,
                        "current_address": dst_addr,
                        "incoming_amount": bridge_amount,
                        "incoming_time": step.time if isinstance(step.time, int) else incoming_time,
                        "chain": dst_chain,
                        "token_id": new_token_id or token_id,
                        "hop_index": hop_index + 1,
                        "incoming_tx_hash": step.tx_hash,
                        "visited": visited | {(dst_addr, dst_chain)},
                    })
                    continue

                # No destination known -> stop
                paths.append(Path(
                    path_id=state["path_id"],
                    description="Deterministic trace path",
                    steps=steps,
                    stop_reason="Reached bridge service - destination unknown",
                ))
                continue

            # Terminal handling
            if is_terminal and entity.role in {"cex_deposit", "otc_service", "unidentified_service"}:
                stop = "Reached CEX deposit" if entity.role == "cex_deposit" else (
                    "Reached OTC service" if entity.role == "otc_service" else "Reached mixer/service"
                )
                paths.append(Path(
                    path_id=state["path_id"],
                    description="Deterministic trace path",
                    steps=steps,
                    stop_reason=stop,
                ))
                continue

            # Fetch outgoing txs
            txs = await self._get_outgoing_txs(current_address, chain, token_id, incoming_time)
            if not txs:
                paths.append(Path(
                    path_id=state["path_id"],
                    description="Deterministic trace path",
                    steps=steps,
                    stop_reason="Dead end - no outgoing transactions",
                ))
                continue

            # Resolve txs and build candidates
            candidates: List[TxCandidate] = []
            for tx in txs:
                tx_hash = tx.get("hash") or tx.get("tx_hash")
                if not tx_hash:
                    continue
                candidate = await self._resolve_transfer_from_tx(tx_hash, chain)
                if not candidate or not candidate.recipient:
                    continue
                if candidate.amount == 0:
                    candidate.amount = _parse_amount(tx.get("amount") or tx.get("amount_coerced") or tx.get("value"))
                if candidate.block_time is None:
                    candidate.block_time = _parse_int(tx.get("block_time"))
                candidates.append(candidate)

            # Sort chronologically
            candidates.sort(key=lambda c: c.block_time or 0)

            if not candidates:
                paths.append(Path(
                    path_id=state["path_id"],
                    description="Deterministic trace path",
                    steps=steps,
                    stop_reason="Dead end - no resolvable outgoing transactions",
                ))
                continue

            # Chronological accumulation
            selected: List[TxCandidate] = []
            accumulated = 0.0
            for cand in candidates:
                selected.append(cand)
                accumulated += cand.amount
                if incoming_amount > 0:
                    gap = (incoming_amount - accumulated) / incoming_amount
                    if accumulated >= incoming_amount or gap <= 0.015:
                        break
                else:
                    break

            # Branch for selected txs
            for i, cand in enumerate(selected):
                recipient = cand.recipient
                if not recipient:
                    continue

                step_type = "direct_transfer"
                step = Step(
                    step_index=len(steps),
                    from_address=current_address,
                    to_address=recipient,
                    tx_hash=cand.tx_hash,
                    chain=chain,
                    asset=asset_symbol,
                    amount_estimate=cand.amount,
                    time=cand.block_time,
                    direction="out",
                    step_type=step_type,
                    service_label=None,
                    protocol=None,
                    reasoning=(
                        f"Tx selected chronologically. Accumulated: {accumulated:.2f} "
                        f"of {incoming_amount:.2f}."
                    ),
                )

                # Detect cycles
                if (recipient, chain) in visited:
                    annotations.append(Annotation(
                        id=f"ann-{annotation_counter}",
                        label="Cycle Detected",
                        related_addresses=[recipient],
                        related_steps=[f"{state['path_id']}:{len(steps)}"],
                        text="Cycle detected - address already seen in this path. Stopping to avoid infinite loop."
                    ))
                    annotation_counter += 1

                    paths.append(Path(
                        path_id=state["path_id"] if i == 0 else f"{state['path_id']}.{i+1}",
                        description="Deterministic trace path",
                        steps=steps + [step],
                        stop_reason="Cycle detected - stopped",
                    ))
                    continue

                next_state = {
                    "path_id": state["path_id"] if i == 0 else f"{state['path_id']}.{i+1}",
                    "steps": steps + [step],
                    "current_address": recipient,
                    "incoming_amount": cand.amount,
                    "incoming_time": cand.block_time,
                    "chain": chain,
                    "token_id": cand.token_id or token_id,
                    "hop_index": hop_index + 1,
                    "incoming_tx_hash": cand.tx_hash,
                    "visited": visited | {(recipient, chain)},
                }
                stack.append(next_state)

        # Final TraceResult
        trace_stats = TraceStats(
            initial_amount_estimate=initial_amount,
            explored_paths=len(paths),
            terminated_reason="All paths reached terminal entities or dead ends",
        )

        # Flatten entities
        entity_list = list(entities.values())

        # Update case_meta chains
        case_meta.chains = sorted(chain_set)

        return TraceResult(
            case_meta=case_meta,
            paths=paths,
            entities=entity_list,
            annotations=annotations,
            trace_stats=trace_stats,
            visualization_url=None,
        )
