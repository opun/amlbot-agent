from typing import Dict, List, Any, Optional, Tuple
from agent.mcp_client import MCPClient
from agent.models import Entity

class AddressClassifier:
    def __init__(self, client: MCPClient):
        self.client = client

    async def classify(self, address: str, chain: str, asset_symbol: str, context: Dict[str, Any]) -> Entity:
        """
        Lightweight wrapper to fetch AML data for an address.
        The LLM orchestrator is expected to reason over this data.
        """
        aml_data = await self.client.get_address(chain, address)

        data = aml_data.get("data", {})
        risk_score_val = data.get("riskscore", {}).get("value", 0.0)
        signals = data.get("riskscore", {}).get("signals", {})

        owner_data = data.get("owner") or {}

        labels: List[str] = []
        reason = "LLM-driven classification; deterministic rules removed."

        if owner_data.get("name"):
            labels.append(owner_data["name"])
        if owner_data.get("subtype"):
            labels.append(owner_data["subtype"])

        return Entity(
            address=address,
            chain=chain,
            role="intermediate",
            risk_score=risk_score_val,
            riskscore_signals=signals,
            labels=labels,
            notes=reason
        )
