from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

class TracerConfig(BaseModel):
    description: Optional[str] = None
    victim_address: Optional[str] = None
    blockchain_name: str = "eth"
    asset_symbol: Optional[str] = None
    approx_date: Optional[str] = None
    known_tx_hashes: List[str] = Field(default_factory=list)
    tx_hash: Optional[str] = None
    theft_asset: Optional[str] = None

class CaseMeta(BaseModel):
    case_id: str
    trace_id: Optional[str] = None
    description: str = ""
    victim_address: str
    blockchain_name: str
    chains: List[str]
    asset_symbol: str
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
    time: Optional[Union[str, int]] # Allow int input
    direction: str
    step_type: Literal["direct_transfer", "bridge_in", "bridge_out", "bridge_transfer", "bridge_arrival", "service_deposit", "internal_transfer"]
    service_label: Optional[str] = None
    protocol: Optional[str] = None
    reasoning: Optional[str] = None  # Explanation for why this transaction was selected

class Path(BaseModel):
    path_id: str
    description: str
    steps: List[Step]
    stop_reason: Optional[str] = None  # Explanation for why tracing stopped on this path

class Entity(BaseModel):
    address: str
    chain: str
    role: Literal["victim", "perpetrator", "intermediate", "bridge_service", "cex_deposit", "otc_service", "unidentified_service", "cluster"]
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

class TraceResult(BaseModel):
    case_meta: CaseMeta
    paths: List[Path]
    entities: List[Entity]
    annotations: List[Annotation]
    trace_stats: TraceStats
    visualization_url: Optional[str] = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, by_alias=True)
