from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_BLOCKCHAINS = {"eth", "trx", "btc", "bsc", "matic", "sol", "arb", "op", "avax", "base", "bch", "ltc", "etc", "ada", "xrp"}

# Aliases accepted by users/frontend that map to SAILS currency codes
_BLOCKCHAIN_ALIASES = {
    "poly": "matic",
    "polygon": "matic",
    "bnb": "bsc",
    "binance": "bsc",
    "bep20": "bsc",
    "tron": "trx",
    "trc": "trx",
    "trc20": "trx",
    "ethereum": "eth",
}


class TracerConfig(BaseModel):
    description: str | None = None
    victim_address: str | None = Field(default=None, min_length=1)
    blockchain_name: str = "eth"
    asset_symbol: str | None = None
    approx_date: str | None = None
    known_tx_hashes: list[str] = Field(default_factory=list)
    tx_hash: str | None = None
    theft_asset: str | None = None
    stolen_amount: float | None = Field(default=None, ge=0)
    cex_single_cluster_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    traced_amount_tolerance: float = Field(default=0.03, ge=0.0, le=1.0)
    # Minimum FIFO-attributed share (as a fraction of stolen_amount) for a
    # hop to be worth pushing onto the scheduler. Branches whose attributed
    # share falls below this get their step recorded and stop_reason
    # "Below dust threshold"; we don't chase them further. 0.0 disables.
    min_path_attribution_ratio: float = Field(default=0.01, ge=0.0, le=1.0)

    @field_validator("blockchain_name")
    @classmethod
    def validate_blockchain(cls, v: str) -> str:
        v_lower = v.lower()
        v_lower = _BLOCKCHAIN_ALIASES.get(v_lower, v_lower)
        if v_lower not in SUPPORTED_BLOCKCHAINS:
            raise ValueError(f"Unsupported blockchain: {v}. Supported: {', '.join(sorted(SUPPORTED_BLOCKCHAINS))}")
        return v_lower

    @field_validator("victim_address")
    @classmethod
    def validate_victim_address(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

class CaseMeta(BaseModel):
    case_id: str
    trace_id: str | None = None
    description: str = ""
    victim_address: str
    blockchain_name: str
    chains: list[str]
    asset_symbol: str
    token_id: int | None = None
    approx_date: str | None = None

class DecisionRef(BaseModel):
    """Pointer to a single LLM decision captured during tracing.

    Attached to every ``Step`` created under an LLM call and to the
    top-level ``TraceResult.decision_log`` for global decisions (e.g.
    validator, seed-tx selection). The ``input_hash`` makes the decision
    replayable; ``usage`` + ``reasoning_tokens`` make it costable.
    """
    prompt_name: str
    prompt_version: str = "v1"
    model: str
    family: Literal["reasoning", "standard"] | str
    reasoning_effort: str | None = None
    input_hash: str
    output_summary: dict = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)
    latency_ms: int = 0
    decision_id: str
    from_replay: bool = False


class Step(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step_index: int
    from_address: str = Field(..., alias="from")
    to_address: str = Field(..., alias="to")
    tx_hash: str | None
    chain: str
    asset: str
    amount_estimate: float
    time: str | int | None = None  # Unix timestamp (int) or ISO 8601 string
    direction: str
    step_type: Literal["direct_transfer", "bridge_in", "bridge_out", "bridge_transfer", "bridge_arrival", "service_deposit", "internal_transfer"]
    service_label: str | None = None
    protocol: str | None = None
    reasoning: str | None = None  # Explanation for why this transaction was selected
    attributed_amount: float | None = None  # FIFO-attributed theft-origin share
    llm_decisions: list[DecisionRef] = Field(default_factory=list)

class Path(BaseModel):
    path_id: str
    description: str
    steps: list[Step]
    stop_reason: str | None = None  # Explanation for why tracing stopped on this path

class Entity(BaseModel):
    address: str
    chain: str
    role: Literal["victim", "perpetrator", "intermediate", "bridge_service", "cex_deposit", "dex_service", "otc_service", "unidentified_service", "cluster"]
    risk_score: float | None = None
    riskscore_signals: dict[str, float] = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    notes: str | None = None

class Annotation(BaseModel):
    id: str
    label: str
    related_addresses: list[str]
    related_steps: list[str]
    text: str

class TraceStats(BaseModel):
    initial_amount_estimate: float
    explored_paths: int
    terminated_reason: str | None = None
    total_traced_amount: float | None = None
    stolen_amount: float | None = None
    fifo_audit_log: list[dict] | None = None

class TraceResult(BaseModel):
    case_meta: CaseMeta
    paths: list[Path]
    entities: list[Entity]
    annotations: list[Annotation]
    trace_stats: TraceStats
    visualization_url: str | None = None
    decision_log: list[DecisionRef] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, by_alias=True)
