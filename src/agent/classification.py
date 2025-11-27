from typing import Dict, List, Any, Optional, Tuple
from agent.mcp_client import MCPClient
from agent.models import Entity

class AddressClassifier:
    def __init__(self, client: MCPClient):
        self.client = client

    async def classify(self, address: str, chain: str, asset_symbol: str, context: Dict[str, Any]) -> Entity:
        """
        Classify an address based on AML data.
        """
        # Fetch data
        aml_data = await self.client.get_address(chain, address)

        data = aml_data.get("data", {})
        risk_score_val = data.get("riskscore", {}).get("value", 0.0)
        signals = data.get("riskscore", {}).get("signals", {})

        # Use owner info from get_address instead of calling get_extra_address_info
        owner_data = data.get("owner") or {}

        role, labels, is_terminal, reason = self._determine_role(
            address, chain, risk_score_val, signals, owner_data, context
        )

        return Entity(
            address=address,
            chain=chain,
            role=role,
            risk_score=risk_score_val,
            riskscore_signals=signals,
            labels=labels,
            notes=reason
        )

    def _determine_role(
        self,
        address: str,
        chain: str,
        risk_score: float,
        signals: Dict[str, float],
        owner_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[str, List[str], bool, str]:

        labels = []
        is_terminal = False
        role = "intermediate"
        reason = f"Classified as {role} based on activity."

        # 1. Check Context (Victim)
        if context.get("is_victim"):
            return "victim", ["Victim"], False, "Address provided as victim by user."

        # 2. Check Owner Info - Primary Source for Identity
        owner_name = owner_data.get("name", "")
        owner_subtype = owner_data.get("subtype", "")
        # owner_type = owner_data.get("type", "") # e.g. 'other', 'exchange'

        # Collect identifiers for matching
        identifiers = []
        if owner_name: identifiers.append(owner_name)
        if owner_subtype: identifiers.append(owner_subtype)

        if identifiers:
            # Add owner name to labels for visibility
            labels.extend(identifiers)

            # Normalize for checking
            id_str = " ".join(identifiers).lower()

            # Bridge / Swap / DEX
            # Check subtype first, then name keywords
            bridge_keywords = ["bridge", "swap", "dex", "uniswap", "sushiswap", "pancakeswap", "curve", "layerzero", "stargate", "allbridge", "wormhole", "router"]

            is_bridge = "bridge" in owner_subtype.lower() or any(bk in id_str for bk in bridge_keywords)

            if is_bridge:
                return "bridge_service", labels, True, f"Identified as Bridge/Swap/DEX service: {owner_name} ({owner_subtype})"

            # CEX / Exchange
            cex_keywords = ["exchange", "binance", "coinbase", "kraken", "huobi", "okx", "kucoin", "bitfinex", "mxc", "gate.io", "poloniex", "bybit"]

            is_cex = "exchange" in owner_subtype.lower() or any(ck in id_str for ck in cex_keywords)

            if is_cex:
                 return "cex_deposit", labels + ["Exchange"], True, f"Identified as CEX/Exchange: {owner_name} ({owner_subtype})"

            # Mixer
            if "mixer" in id_str:
                return "unidentified_service", ["Mixer"] + labels, True, f"Identified as Mixer: {owner_name}"

            # OTC
            if "otc" in id_str:
                return "otc_service", ["OTC"] + labels, True, f"Identified as OTC: {owner_name}"

            # Generic Service (if name exists but didn't match above)
            # If it has a name, it's likely a service or known entity, not a private wallet
            # But we default to intermediate if we can't pinpoint the type,
            # UNLESS it's a known entity type we should stop at?
            # For now, treat as intermediate but labelled.
            pass

        # 3. Risk Signals - Secondary Context (Source of Funds)
        # We use these to flag high-risk wallets or infer roles if identity is missing.

        reason_parts = []
        if risk_score > 0.75:
            labels.append("High Risk")
            reason_parts.append(f"High Risk Score: {risk_score:.2f}")

        # If no specific identity found, check for overwhelming risk signals
        # but be careful not to mislabel a wallet holding stolen funds as the service itself unless sure.

        # 4. Heuristic for Perpetrator (usually first hop)
        if context.get("hop_index", 0) == 0 and role == "intermediate":
            role = "perpetrator"
            labels.append("Suspected Perpetrator")
            reason_parts.append("First hop from victim, suspected perpetrator wallet.")
        elif reason_parts:
            reason = "; ".join(reason_parts)

        return role, labels, is_terminal, reason
