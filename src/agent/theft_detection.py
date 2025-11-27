import re
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from agent.models import TracerConfig
from agent.mcp_client import MCPClient

def infer_approx_date_from_description(description: str) -> Optional[str]:
    """
    Extract approximate date from description using regex.
    Returns YYYY-MM-DD string or None.
    """
    # Patterns: YYYY-MM-DD, DD.MM.YYYY, on Month DD
    patterns = [
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{2})\.(\d{2})\.(\d{4})",
    ]

    for pat in patterns:
        match = re.search(pat, description)
        if match:
            if pat.startswith(r"(\d{4})"):
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            else:
                return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"

    # Month name handling (simple version)
    months = {
        "January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06",
        "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"
    }
    for month, num in months.items():
        # Pattern: on Month DD
        match = re.search(rf"on {month} (\d{{1,2}})", description, re.IGNORECASE)
        if match:
            day = match.group(1).zfill(2)
            year = datetime.now().year # Assumption
            return f"{year}-{num}-{day}"

        # NEW Pattern: DD Month YYYY (e.g. 24 September 2025)
        match_dmy = re.search(rf"(\d{{1,2}})\s+{month}\s+(\d{{4}})", description, re.IGNORECASE)
        if match_dmy:
            day = match_dmy.group(1).zfill(2)
            year = match_dmy.group(2)
            return f"{year}-{num}-{day}"

    return None

async def infer_asset_symbol(config: TracerConfig, client: MCPClient) -> Tuple[str, int]:
    """
    Detect asset symbol and token_id.
    If provided in config, verify it exists.
    If not, choose largest outgoing token.
    """
    if config.asset_symbol:
        # TODO: Validate it exists using token-stats?
        # For now, assume token_id 0 if generic, but we might need token_id lookup
        # This part is tricky without a token map. We'll rely on token-stats to find the ID.
        pass

    stats = await client.token_stats(config.blockchain_name, config.victim_address)
    tokens = stats.get("tokens", [])

    if not tokens:
        return config.asset_symbol or "ETH", 0

    # If asset_symbol provided, find its ID
    if config.asset_symbol:
        for t in tokens:
            if t.get("symbol", "").upper() == config.asset_symbol.upper():
                return config.asset_symbol, t.get("token_id", 0)
        # If not found, warn and return provided (maybe it's native?)
        return config.asset_symbol, 0

    # Heuristic: Max total_out
    best_token = max(tokens, key=lambda t: float(t.get("total_out", 0)))
    if float(best_token.get("total_out", 0)) == 0:
         # Fallback to native or first
         return "ETH", 0

    return best_token.get("symbol"), best_token.get("token_id")

async def fetch_candidate_theft_txs(config: TracerConfig, asset_symbol: str, token_id: int, client: MCPClient) -> List[Dict[str, Any]]:
    """
    Fetch outgoing transactions and filter by asset/amount.
    Optimized to avoid calling get-transaction for every candidate.
    """
    # Filter for outgoing
    filter_criteria = {
        "delta_coerced": [{"<=": -0.0001}],
        "amount_coerced": [{">": 0}]
    }

    # Add time window if date known
    if config.approx_date:
        try:
            dt = datetime.strptime(config.approx_date, "%Y-%m-%d")
            ts = int(dt.timestamp())
            window = 7 * 24 * 3600 # 7 days
            filter_criteria["block_time"] = {">=": ts - window, "<=": ts + window}
        except ValueError:
            pass

    # Fetch more transactions to increase chance of finding user-provided hash
    response = await client.all_txs(config.victim_address, config.blockchain_name, filter_criteria, limit=20)
    raw_txs = response.get("data", [])

    # 1. Check for User-Provided Hashes (Priority)
    if config.known_tx_hashes:
        known_set = set(h.lower() for h in config.known_tx_hashes)
        matching_txs = [tx for tx in raw_txs if tx.get("hash", "").lower() in known_set]

        if matching_txs:
            candidates = []
            for tx in matching_txs:
                # Use the transaction as-is, skipping heuristic filters
                amount = abs(float(tx.get("delta_coerced") or tx.get("amount_coerced") or 0))
                candidates.append({
                    "tx_hash": tx.get("hash"),
                    "time": tx.get("block_time"),
                    "amount": amount,
                    "to": None, # Resolved later
                    "path": tx.get("path", "0"),
                    "token_id": tx.get("token_id", 0),
                    "asset_symbol": asset_symbol # Might need to infer if different from requested?
                })
            return candidates

    # 2. Heuristic Filtering (Fallback)
    candidates = []
    for tx in raw_txs:
        tx_hash = tx.get("hash")
        amount = abs(float(tx.get("delta_coerced") or tx.get("amount_coerced") or 0))

        # Check token_id if possible (all-txs usually returns it)
        tx_token_id = tx.get("token_id", 0)
        if token_id != 0 and tx_token_id != token_id:
            continue

        # Optimization: Don't call get-transaction yet.
        # We will only resolve the 'to' address for the CHOSEN candidate(s) later.
        candidates.append({
            "tx_hash": tx_hash,
            "time": tx.get("block_time"),
            "amount": amount,
            "to": None, # Resolved later
            "path": tx.get("path", "0"),
            "token_id": tx.get("token_id", 0), # CHANGED: Use actual tx token_id
            "asset_symbol": asset_symbol
        })

    return candidates

def choose_theft_tx(candidates: List[Dict[str, Any]], approx_date: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Select the most likely theft transaction.
    """
    if not candidates:
        return None

    # Fallback: max amount
    return max(candidates, key=lambda x: x["amount"])
