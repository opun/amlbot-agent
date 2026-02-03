from agent.models import TraceResult


def _wrap_text(text: str, max_width: int = 70) -> list[str]:
    """Word-wrap text to fit within max_width characters."""
    if not text:
        return []

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + (1 if current_line else 0) <= max_width:
            current_line.append(word)
            current_length += word_length + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length

    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [text]


def _get_path_stop_reason(path, annotations) -> str:
    """Get the stop reason for a path from annotations or path.stop_reason."""
    # First check if stop_reason is set on the path itself
    if hasattr(path, 'stop_reason') and path.stop_reason:
        return path.stop_reason

    # Fallback to checking annotations
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
            # Add reasoning if available
            if hasattr(step, 'reasoning') and step.reasoning:
                summary += f"    Reasoning: {step.reasoning}\n"

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
            edge_data = {
                "id": f"edge-{path.path_id}-{step.step_index}",
                "from": f"{step.from_address}-{step.chain}",
                "to": f"{step.to_address}-{step.chain}",
                "tx_hash": step.tx_hash,
                "relation": "flow",
                "chain": step.chain,
                "asset": step.asset,
                "amount_estimate": step.amount_estimate,
                "step_type": step.step_type
            }
            # Add reasoning if available
            if hasattr(step, 'reasoning') and step.reasoning:
                edge_data["reasoning"] = step.reasoning
            edges.append(edge_data)

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

def build_mermaid_graph(trace_result: TraceResult) -> str:
    """
    Generate a Mermaid flowchart diagram of the trace.
    """
    import re

    mermaid = ["flowchart TD"]

    # Define styling with explicit text colors for readability
    mermaid.append("    %% Styling")
    mermaid.append("    classDef victim fill:#ffcccc,stroke:#ff0000,stroke-width:2px,color:#000000,font-weight:bold")
    mermaid.append("    classDef perpetrator fill:#ff9999,stroke:#cc0000,stroke-width:2px,color:#000000,font-weight:bold")
    mermaid.append("    classDef service fill:#ccffcc,stroke:#00cc00,stroke-width:2px,color:#000000,font-weight:bold")
    mermaid.append("    classDef unknown fill:#d1d5db,stroke:#4b5563,stroke-width:2px,color:#111827,font-weight:600")

    # Nodes - use simple counter-based IDs to avoid any special character issues
    added_nodes = {}  # address -> node_id mapping
    node_counter = 0

    # Define a helper to get or create a valid node id
    # Mermaid node IDs must start with a letter and only contain alphanumeric + underscore
    def get_node_id(addr):
        if addr in added_nodes:
            return added_nodes[addr]

        nonlocal node_counter
        # Use simple counter-based ID to ensure it's always valid
        node_id = f"N{node_counter}"
        node_counter += 1
        added_nodes[addr] = node_id
        return node_id

    # Helper to escape label text for Mermaid
    def escape_label(text):
        # Escape quotes and special characters
        return text.replace('"', '&quot;').replace('\n', '<br/>')

    for entity in trace_result.entities:
        if entity.address in added_nodes:
            continue

        node_id = get_node_id(entity.address)
        label = f"{entity.address[:6]}...{entity.address[-4:]}"

        # Add risk score if high
        if entity.risk_score and entity.risk_score > 0.7:
            label += f"<br/>Risk: {entity.risk_score:.2f}"

        # Determine class based on role
        style_class = "unknown"
        if entity.role == "victim":
            style_class = "victim"
            label += "<br/>(Victim)"
        elif entity.role == "perpetrator":
            style_class = "perpetrator"
            label += "<br/>(Perpetrator)"
        elif entity.role in ["bridge_service", "cex_deposit", "otc_service", "unidentified_service"]:
            style_class = "service"
            service_name = entity.labels[0] if entity.labels else entity.role
            # Escape service name to avoid special characters
            service_name = escape_label(str(service_name))
            label += f"<br/>({service_name})"

        label = escape_label(label)
        mermaid.append(f'    {node_id}("{label}"):::{style_class}')

    # Edges (Transactions)
    for path in trace_result.paths:
        for step in path.steps:
            src_id = get_node_id(step.from_address)
            dst_id = get_node_id(step.to_address)

            # Ensure nodes exist (if not in entities list for some reason)
            if step.from_address not in added_nodes:
                label = escape_label(f"{step.from_address[:6]}...")
                mermaid.append(f'    {src_id}("{label}"):::unknown')
            if step.to_address not in added_nodes:
                label = escape_label(f"{step.to_address[:6]}...")
                mermaid.append(f'    {dst_id}("{label}"):::unknown')

            amount_str = f"{step.amount_estimate:.2f} {step.asset}"
            # Escape amount string
            amount_str = escape_label(amount_str)
            mermaid.append(f'    {src_id} -- "{amount_str}" --> {dst_id}')

    return "\n".join(mermaid)

def build_ascii_tree(trace_result: TraceResult) -> str:
    """
    Generate an ASCII tree visualization of the trace paths.
    """
    if not trace_result.paths:
        return "No paths found."

    # Build entity lookup for role/labels
    entity_map = {e.address: e for e in trace_result.entities}

    def format_address(addr: str) -> str:
        """Format address with role info."""
        short = f"{addr[:8]}...{addr[-6:]}"
        entity = entity_map.get(addr)
        if entity:
            role_icons = {
                "victim": "🔴",
                "perpetrator": "💀",
                "intermediate": "🔵",
                "bridge_service": "🌉",
                "cex_deposit": "🏦",
                "otc_service": "💱",
                "unidentified_service": "❓",
                "cluster": "🔗"
            }
            icon = role_icons.get(entity.role, "⚪")
            label = entity.labels[0] if entity.labels else entity.role
            return f"{icon} {short} [{label}]"
        return f"⚪ {short}"

    def format_amount(amount: float, asset: str) -> str:
        """Format amount with commas."""
        if amount >= 1_000_000:
            return f"{amount/1_000_000:.2f}M {asset}"
        elif amount >= 1_000:
            return f"{amount/1_000:.2f}K {asset}"
        else:
            return f"{amount:.2f} {asset}"

    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                              TRACE FLOW DIAGRAM                              ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")

    meta = trace_result.case_meta
    lines.append(f"║ Case: {meta.case_id:<70} ║")
    lines.append(f"║ Asset: {meta.asset_symbol} on {meta.blockchain_name:<62} ║")
    lines.append(f"║ Initial Amount: {trace_result.trace_stats.initial_amount_estimate:<60} ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    for path_idx, path in enumerate(trace_result.paths):
        # Path header
        lines.append(f"┌─ Path {path.path_id} ─────────────────────────────────────────────────────────────")
        if path.description:
            # Word wrap path description
            desc_lines = _wrap_text(path.description, max_width=75)
            for d_line in desc_lines:
                lines.append(f"│  {d_line}")
        lines.append("│")

        for i, step in enumerate(path.steps):
            is_last = (i == len(path.steps) - 1)

            # Source node (only for first step or when source changes)
            if i == 0 or step.from_address != path.steps[i-1].to_address:
                lines.append(f"│  {format_address(step.from_address)}")

            # Arrow with amount
            amount_str = format_amount(step.amount_estimate, step.asset)
            step_type_str = step.step_type.replace("_", " ").title()

            if is_last:
                connector = "└"
                prefix = "   "
            else:
                connector = "├"
                prefix = "│  "

            lines.append(f"│  │")
            lines.append(f"│  │ ──[ {amount_str} ]── {step_type_str}")
            if step.tx_hash:
                lines.append(f"│  │    tx: {step.tx_hash[:20]}...{step.tx_hash[-8:]}")

            # Reasoning - show full text, word-wrapped
            if hasattr(step, 'reasoning') and step.reasoning:
                # Word wrap long reasoning to multiple lines
                reasoning_lines = _wrap_text(step.reasoning, max_width=70)
                for idx, r_line in enumerate(reasoning_lines):
                    prefix = "💭 " if idx == 0 else "   "
                    lines.append(f"│  │    {prefix}{r_line}")

            lines.append(f"│  ▼")

            # Destination node
            lines.append(f"│  {format_address(step.to_address)}")

            if not is_last:
                lines.append("│")

        # Path stop reason - show full text
        stop_reason = _get_path_stop_reason(path, trace_result.annotations)
        lines.append("│")
        stop_lines = _wrap_text(stop_reason, max_width=65)
        for idx, s_line in enumerate(stop_lines):
            prefix = "══► STOP: " if idx == 0 else "          "
            lines.append(f"│  {prefix}{s_line}")
        lines.append("└────────────────────────────────────────────────────────────────────────────")
        lines.append("")

    # Legend
    lines.append("┌─ LEGEND ───────────────────────────────────────────────────────────────────")
    lines.append("│  🔴 Victim    💀 Perpetrator    🔵 Intermediate    🌉 Bridge")
    lines.append("│  🏦 CEX       💱 OTC            ❓ Unknown Service 🔗 Cluster")
    lines.append("└────────────────────────────────────────────────────────────────────────────")

    return "\n".join(lines)


def build_report(trace_result: TraceResult) -> dict:
    return {
        "summary_text": build_summary_text(trace_result),
        "ascii_tree": build_ascii_tree(trace_result),
        "graph": build_graph(trace_result),
        "mermaid": build_mermaid_graph(trace_result)
    }
