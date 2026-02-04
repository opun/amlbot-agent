import logging
from typing import List, Set, Tuple
from datetime import datetime

from agent.models import TraceResult, Path, Step, Entity, Annotation

logger = logging.getLogger("trace_postprocess")

ALLOWED_STEP_TYPES = {
    "direct_transfer",
    "bridge_in",
    "bridge_out",
    "bridge_transfer",
    "bridge_arrival",
    "service_deposit",
    "internal_transfer",
}


def _coerce_float(value) -> float:
    try:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.replace(",", ""))
    except Exception:
        return 0.0
    return 0.0


def _coerce_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        # Try numeric strings
        try:
            return int(float(value))
        except Exception:
            pass
        # Try ISO date strings
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except Exception:
            return value
    return value


def postprocess_trace_result(trace_result: TraceResult) -> TraceResult:
    """
    Post-validate and auto-repair TraceResult:
    - Enforce linear steps per path (split siblings).
    - Normalize step types, amounts, time.
    - Detect cycles and annotate.
    - Ensure entities exist for all addresses.
    """
    new_paths: List[Path] = []
    annotations: List[Annotation] = list(trace_result.annotations or [])
    annotation_counter = len(annotations) + 1

    # Collect existing entities
    entity_map = {(e.address, e.chain): e for e in trace_result.entities or []}

    def ensure_entity(address: str, chain: str):
        key = (address, chain)
        if key not in entity_map:
            entity_map[key] = Entity(
                address=address,
                chain=chain,
                role="intermediate",
                risk_score=0.0,
                riskscore_signals={},
                labels=[],
                notes="Auto-added during postprocess",
            )

    for path in trace_result.paths or []:
        if not path.steps:
            continue

        current_steps: List[Step] = []
        prev_to = None
        split_index = 0

        for step in path.steps:
            # Normalize step fields
            if step.step_type not in ALLOWED_STEP_TYPES:
                step.step_type = "direct_transfer"
            step.amount_estimate = _coerce_float(step.amount_estimate)
            step.time = _coerce_time(step.time)
            if not step.chain:
                step.chain = trace_result.case_meta.blockchain_name
            if step.asset:
                step.asset = step.asset.upper()
            else:
                step.asset = trace_result.case_meta.asset_symbol
            if step.direction not in ["in", "out"]:
                step.direction = "out"

            # Ensure entities exist
            ensure_entity(step.from_address, step.chain)
            ensure_entity(step.to_address, step.chain)

            if prev_to is None or step.from_address == prev_to:
                step.step_index = len(current_steps)
                current_steps.append(step)
            else:
                # Split path due to sibling branch
                new_paths.append(Path(
                    path_id=path.path_id if split_index == 0 else f"{path.path_id}.{split_index+1}",
                    description=path.description,
                    steps=current_steps,
                    stop_reason=path.stop_reason,
                ))
                annotations.append(Annotation(
                    id=f"ann-{annotation_counter}",
                    label="Path Split",
                    related_addresses=[step.from_address],
                    related_steps=[f"{path.path_id}:{step.step_index}"],
                    text="Detected sibling branch in steps; split into separate path for linearity."
                ))
                annotation_counter += 1
                current_steps = [step]
                step.step_index = 0
                split_index += 1

            prev_to = step.to_address

        if current_steps:
            new_paths.append(Path(
                path_id=path.path_id if split_index == 0 else f"{path.path_id}.{split_index+1}",
                description=path.description,
                steps=current_steps,
                stop_reason=path.stop_reason,
            ))

    # Cycle detection
    for path in new_paths:
        seen: Set[Tuple[str, str]] = set()
        for step in path.steps:
            key = (step.to_address, step.chain)
            if key in seen:
                annotations.append(Annotation(
                    id=f"ann-{annotation_counter}",
                    label="Cycle Detected",
                    related_addresses=[step.to_address],
                    related_steps=[f"{path.path_id}:{step.step_index}"],
                    text="Cycle detected in path; results may include a loop."
                ))
                annotation_counter += 1
                if not path.stop_reason:
                    path.stop_reason = "Cycle detected - stopped"
                break
            seen.add(key)

    # Update trace_result
    trace_result.paths = new_paths
    trace_result.annotations = annotations
    trace_result.entities = list(entity_map.values())

    # Update stats
    if trace_result.trace_stats:
        trace_result.trace_stats.explored_paths = len(new_paths)

    # Update chains
    chains = {trace_result.case_meta.blockchain_name}
    for path in new_paths:
        for step in path.steps:
            chains.add(step.chain)
    trace_result.case_meta.chains = sorted(chains)

    return trace_result
