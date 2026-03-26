from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator

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
    description: Optional[str] = None
    victim_address: Optional[str] = Field(default=None, min_length=1)
    blockchain_name: str = "eth"
    asset_symbol: Optional[str] = None
    approx_date: Optional[str] = None
    known_tx_hashes: List[str] = Field(default_factory=list)
    tx_hash: Optional[str] = None
    theft_asset: Optional[str] = None
    stolen_amount: Optional[float] = Field(default=None, ge=0)
    cex_single_cluster_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    traced_amount_tolerance: float = Field(default=0.03, ge=0.0, le=1.0)

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
    def validate_victim_address(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

class CaseMeta(BaseModel):
    case_id: str
    trace_id: Optional[str] = None
    description: str = ""
    victim_address: str
    blockchain_name: str
    chains: List[str]
    asset_symbol: str
    token_id: Optional[int] = None
    approx_date: Optional[str] = None

class Step(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step_index: int
    from_address: str = Field(..., alias="from")
    to_address: str = Field(..., alias="to")
    tx_hash: Optional[str]
    chain: str
    asset: str
    amount_estimate: float
    time: Optional[Union[str, int]] = None  # Unix timestamp (int) or ISO 8601 string
    direction: str
    step_type: Literal["direct_transfer", "bridge_in", "bridge_out", "bridge_transfer", "bridge_arrival", "service_deposit", "internal_transfer"]
    service_label: Optional[str] = None
    protocol: Optional[str] = None
    reasoning: Optional[str] = None  # Explanation for why this transaction was selected
    attributed_amount: Optional[float] = None  # FIFO-attributed theft-origin share

class Path(BaseModel):
    path_id: str
    description: str
    steps: List[Step]
    stop_reason: Optional[str] = None  # Explanation for why tracing stopped on this path

class Entity(BaseModel):
    address: str
    chain: str
    role: Literal["victim", "perpetrator", "intermediate", "bridge_service", "cex_deposit", "dex_service", "otc_service", "unidentified_service", "cluster"]
    risk_score: Optional[float] = None
    riskscore_signals: Dict[str, float] = Field(default_factory=dict)
    labels: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class Annotation(BaseModel):
    id: str
    label: str
    related_addresses: List[str]
    related_steps: List[str]
    text: str

class TraceStats(BaseModel):
    initial_amount_estimate: float
    explored_paths: int
    terminated_reason: Optional[str] = None
    total_traced_amount: Optional[float] = None
    stolen_amount: Optional[float] = None
    fifo_audit_log: Optional[List[dict]] = None

class TraceResult(BaseModel):
    case_meta: CaseMeta
    paths: List[Path]
    entities: List[Entity]
    annotations: List[Annotation]
    trace_stats: TraceStats
    visualization_url: Optional[str] = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, by_alias=True)
