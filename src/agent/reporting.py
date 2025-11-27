from agent.models import TraceResult

def _get_path_stop_reason(path, annotations) -> str:
    """Get the stop reason for a path from annotations."""
    last_step_ref = f"{path.path_id}:{len(path.steps) - 1}" if path.steps else None

    for ann in annotations:
        if last_step_ref and last_step_ref in ann.related_steps:
            return ann.text

    return "Max hops reached or trace completed"


def build_summary_text(trace_result: TraceResult) -> str:
    """
    Generate a human-readable summary of the trace.
    """
    meta = trace_result.case_meta
    stats = trace_result.trace_stats

    summary = f"Case {meta.case_id}: Tracing {meta.asset_symbol} on {meta.blockchain_name}.\n"
    summary += f"Victim: {meta.victim_address}\n"
    if meta.approx_date:
        summary += f"Approximate Date: {meta.approx_date}\n"

    summary += f"\nTrace Stats:\n- Explored Paths: {stats.explored_paths}\n"
    summary += f"- Initial Amount: {stats.initial_amount_estimate}\n"

    if not trace_result.paths:
        summary += "\nNo paths found. No outgoing theft transaction identified."
        return summary

    summary += "\nFindings:\n"
    for path in trace_result.paths:
        summary += f"\nPath {path.path_id}:\n"
        for step in path.steps:
            summary += f"  Step {step.step_index}: {step.from_address} -> {step.to_address} "
            summary += f"({step.amount_estimate} {step.asset}) via {step.step_type}\n"
            if step.tx_hash:
                summary += f"    Tx: {step.tx_hash}\n"
            if step.service_label:
                summary += f"    Service: {step.service_label}\n"

        # Add stop reason for this path
        stop_reason = _get_path_stop_reason(path, trace_result.annotations)
        summary += f"  >> Path stopped: {stop_reason}\n"

    # Identify endpoints
    endpoints = [e for e in trace_result.entities if e.role in ["cex_deposit", "otc_service", "unidentified_service"]]
    if endpoints:
        summary += "\nIdentified Endpoints:\n"
        for e in endpoints:
            reason = e.notes if e.notes else "Based on risk signals"
            risk_info = f" (Risk Score: {e.risk_score})" if e.risk_score is not None else ""
            summary += f"- {e.address} ({e.role}): {', '.join(e.labels)}\n"
            summary += f"  Reason: {reason}{risk_info}\n"

    return summary

def build_graph(trace_result: TraceResult) -> dict:
    """
    Build a graph JSON object for visualization.
    """
    nodes = []
    for entity in trace_result.entities:
        nodes.append({
            "id": f"{entity.address}-{entity.chain}",
            "type": "address",
            "address": entity.address,
            "chain": entity.chain,
            "role": entity.role,
            "labels": entity.labels,
            "metadata": {
                "risk_score": entity.risk_score,
                "notes": entity.notes
            }
        })

    edges = []
    for path in trace_result.paths:
        for step in path.steps:
            edges.append({
                "id": f"edge-{path.path_id}-{step.step_index}",
                "from": f"{step.from_address}-{step.chain}",
                "to": f"{step.to_address}-{step.chain}",
                "tx_hash": step.tx_hash,
                "relation": "flow",
                "chain": step.chain,
                "asset": step.asset,
                "amount_estimate": step.amount_estimate,
                "step_type": step.step_type
            })

    # Build path summaries with stop reasons
    path_summaries = []
    for path in trace_result.paths:
        stop_reason = _get_path_stop_reason(path, trace_result.annotations)
        last_step = path.steps[-1] if path.steps else None
        path_summaries.append({
            "path_id": path.path_id,
            "description": path.description,
            "total_steps": len(path.steps),
            "last_step_index": last_step.step_index if last_step else None,
            "last_address": last_step.to_address if last_step else None,
            "stop_reason": stop_reason
        })

    return {
        "case_meta": trace_result.case_meta.model_dump(),
        "nodes": nodes,
        "edges": edges,
        "paths": path_summaries,
        "comments": [a.model_dump() for a in trace_result.annotations]
    }

def build_report(trace_result: TraceResult) -> dict:
    return {
        "summary_text": build_summary_text(trace_result),
        "graph": build_graph(trace_result)
    }
