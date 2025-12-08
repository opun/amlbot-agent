import re
import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from agent.models import TracerConfig
from agent.mcp_client import MCPClient
from agents import Agent, Runner

logger = logging.getLogger("theft_detection")

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

async def parse_case_description_with_llm(description: str) -> Dict[str, Any]:
    """
    Use OpenAI to parse case description and extract all required information.
    Returns a dictionary with extracted fields:
    - victim_address: Optional[str]
    - tx_hash: Optional[str]
    - blockchain_name: Optional[str] (default: "eth")
    - asset_symbol: Optional[str]
    - theft_asset: Optional[str]
    - approx_date: Optional[str] (YYYY-MM-DD format)
    - threshold: Optional[float] (as percentage, e.g., 10 for 10%)
    - known_tx_hashes: List[str]
    """
    if not description:
        return {}

    parser_agent = Agent(
        name="case_parser",
        instructions="""You are a case description parser for cryptocurrency theft investigations.
Your job is to extract structured information from case descriptions.

Extract the following information if mentioned:
1. **victim_address**: The wallet address of the victim (look for addresses starting with 0x, T, bc1, etc.)
2. **tx_hash**: Transaction hash (look for long hexadecimal strings, typically 64+ characters)
3. **blockchain_name**: Blockchain network (eth, trx, btc, etc.) - default to "eth" if not specified
4. **asset_symbol**: The cryptocurrency asset (USDT, ETH, BTC, etc.)
5. **theft_asset**: Same as asset_symbol, the asset that was stolen
6. **approx_date**: Approximate date of the theft (convert to YYYY-MM-DD format)
7. **threshold**: Threshold percentage for tracing (e.g., "10%" or "10 percent" -> 10)
8. **known_tx_hashes**: Any additional transaction hashes mentioned (comma-separated)

Return ONLY a valid JSON object with these fields. Use null for missing fields.
Do not include any explanation or markdown formatting, just the JSON.

Example output:
{
  "victim_address": "0x1234...",
  "tx_hash": "0xabcd...",
  "blockchain_name": "eth",
  "asset_symbol": "USDT",
  "theft_asset": "USDT",
  "approx_date": "2025-01-15",
  "threshold": 10,
  "known_tx_hashes": ["0xhash1", "0xhash2"]
}"""
    )

    prompt = f"""Parse the following case description and extract all relevant information:

{description}

Return a JSON object with the extracted fields."""

    try:
        result = await Runner.run(parser_agent, input=prompt)
        output = result.final_output.strip()

        # Remove markdown code fences if present
        if output.startswith("```json"):
            output = output[7:]
        elif output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        output = output.strip()

        # Parse JSON
        parsed = json.loads(output)

        # Convert threshold from percentage to ratio if provided
        if "threshold" in parsed and parsed["threshold"] is not None:
            parsed["threshold"] = float(parsed["threshold"]) / 100.0

        # Normalize blockchain_name
        if parsed.get("blockchain_name"):
            blockchain = parsed["blockchain_name"].lower()
            # Map common variations
            blockchain_map = {
                "ethereum": "eth",
                "eth": "eth",
                "tron": "trx",
                "trx": "trx",
                "bitcoin": "btc",
                "btc": "btc",
                "polygon": "poly",
                "poly": "poly",
                "bsc": "bsc",
                "binance": "bsc"
            }
            parsed["blockchain_name"] = blockchain_map.get(blockchain, blockchain)
        else:
            parsed["blockchain_name"] = "eth"  # default

        # Normalize asset symbols to uppercase
        if parsed.get("asset_symbol"):
            parsed["asset_symbol"] = parsed["asset_symbol"].upper()
        if parsed.get("theft_asset"):
            parsed["theft_asset"] = parsed["theft_asset"].upper()

        # Normalize trace_mode
        if parsed.get("trace_mode"):
            mode = parsed["trace_mode"].lower()
            if "transaction" in mode or "hash" in mode or "tx" in mode:
                parsed["trace_mode"] = "transaction"
            elif "address" in mode or "wallet" in mode:
                parsed["trace_mode"] = "address"
            else:
                parsed["trace_mode"] = None

        logger.debug(f"Parsed case description: {parsed}")
        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM output as JSON: {e}")
        logger.warning(f"Raw output: {output[:500]}")
        return {}
    except Exception as e:
        logger.warning(f"Error parsing case description with LLM: {e}")
        return {}

async def extract_victim_from_tx_hash(
    tx_hash: str,
    blockchain_name: str,
    client: MCPClient
) -> Tuple[str, int, Optional[str], Optional[int]]:
    """
    Extract victim address from a transaction hash using expert-search.
    Returns: (victim_address, token_id, asset_symbol, block_time)
    """
    logger.debug(f"Extracting victim address from tx_hash: {tx_hash}")

    # Step 1: Use expert-search to find the transaction
    search_result = await client.expert_search(tx_hash, filter="explorer")

    # Handle response structure
    if isinstance(search_result, dict) and "text" in search_result:
        try:
            import json
            search_result = json.loads(search_result["text"])
        except (json.JSONDecodeError, TypeError):
            pass

    data = search_result.get("data", [])
    if not data:
        raise ValueError(f"Transaction {tx_hash} not found via expert-search")

    # Step 2: Extract addresses from search results
    # Expert-search may return transaction or address results
    addresses_to_try = []
    for item in data:
        if item.get("type") == "address":
            addresses_to_try.append(item.get("label") or item.get("slug"))
        elif item.get("type") == "transaction":
            # If transaction type, might have addresses in it
            if "label" in item:
                addresses_to_try.append(item["label"])

    if not addresses_to_try:
        # Try to extract from any field that looks like an address
        for item in data:
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 20:  # Likely an address
                    addresses_to_try.append(value)

    if not addresses_to_try:
        raise ValueError(f"Could not extract addresses from expert-search results for {tx_hash}")

    # Step 3: Try each address to get transaction details
    # We need to find which address is involved in this transaction
    victim_address = None
    token_id = 0
    asset_symbol = None
    block_time = None

    for address in addresses_to_try[:5]:  # Limit to first 5 addresses
        try:
            # Try with token_id 0 first (native token)
            tx_detail = await client.get_transaction(address, tx_hash, blockchain_name, token_id=0, path="0")

            # Handle text-wrapped JSON response
            if isinstance(tx_detail, dict) and "text" in tx_detail:
                try:
                    import json
                    tx_detail = json.loads(tx_detail["text"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Handle response structure
            if tx_detail.get("success") is True and "data" in tx_detail:
                tx_data = tx_detail["data"]
            else:
                tx_data = tx_detail.get("data", tx_detail)

            # Check if this is the right transaction
            tx_hash_from_data = tx_data.get("hash", "")
            if tx_hash_from_data and tx_hash_from_data.lower() == tx_hash.lower():
                # Extract input address (victim) and output address
                input_addr = tx_data.get("input", {}).get("address")
                output_addr = tx_data.get("output", {}).get("address")

                if input_addr:
                    victim_address = input_addr
                    token_id = tx_data.get("token_id", 0)
                    block_time = tx_data.get("block_time")
                    # Try to infer asset from transaction type or token_id
                    tx_type = tx_data.get("type", "")
                    if tx_type == "send" and token_id == 0:
                        # Native token - infer from blockchain
                        asset_symbol = "ETH" if blockchain_name == "eth" else blockchain_name.upper()
                    logger.debug(f"Found victim address: {victim_address} from transaction")
                    break
                elif output_addr:
                    # If we only have output address, the address we used might be the victim
                    # This happens when the address we're querying is the sender
                    victim_address = address
                    token_id = tx_data.get("token_id", 0)
                    block_time = tx_data.get("block_time")
                    logger.debug(f"Using queried address as victim: {victim_address}")
                    break
        except Exception as e:
            logger.debug(f"Failed to get transaction with address {address}: {e}")
            continue

    if not victim_address:
        # Last resort: try to get transaction using all-txs on one of the addresses
        # and find the matching hash
        for address in addresses_to_try[:3]:
            try:
                # Get all transactions for this address
                all_txs_res = await client.all_txs(
                    address,
                    blockchain_name,
                    filter_criteria={"amount_coerced": [{">": 0}]},
                    limit=100
                )
                txs = all_txs_res.get("data", [])

                # Find the matching transaction
                for tx in txs:
                    if tx.get("hash", "").lower() == tx_hash.lower():
                        # This is an outgoing transaction (negative delta)
                        if tx.get("delta_coerced", 0) < 0:
                            victim_address = address
                            token_id = tx.get("token_id", 0)
                            block_time = tx.get("block_time")
                            logger.debug(f"Found victim address via all-txs: {victim_address}")
                            break

                if victim_address:
                    break
            except Exception as e:
                logger.debug(f"Failed to get all-txs for {address}: {e}")
                continue

    if not victim_address:
        raise ValueError(f"Could not extract victim address from transaction {tx_hash}")

    return victim_address, token_id, asset_symbol, block_time

async def infer_asset_symbol(config: TracerConfig, client: MCPClient) -> Tuple[str, int]:
    """
    Detect asset symbol and token_id.
    If provided in config, verify it exists.
    If not, choose largest outgoing token.
    """
    # If theft_asset is provided (Mode 2), use it directly
    if config.theft_asset:
        # Try to find token_id by checking token stats if victim_address is available
        if config.victim_address:
            try:
                stats = await client.token_stats(config.blockchain_name, config.victim_address)
                tokens = stats.get("tokens", [])
                for t in tokens:
                    if t.get("symbol", "").upper() == config.theft_asset.upper():
                        return config.theft_asset, t.get("token_id", 0)
            except Exception:
                pass
        # Fallback: return the provided asset with token_id 0 (will be updated from transaction if available)
        return config.theft_asset, 0

    # If asset_symbol is provided, use it
    if config.asset_symbol:
        # If victim_address is available, try to find token_id
        if config.victim_address:
            try:
                stats = await client.token_stats(config.blockchain_name, config.victim_address)
                tokens = stats.get("tokens", [])
                for t in tokens:
                    if t.get("symbol", "").upper() == config.asset_symbol.upper():
                        return config.asset_symbol, t.get("token_id", 0)
            except Exception:
                pass
        # Fallback: return provided asset with token_id 0
        return config.asset_symbol, 0

    # If no victim_address, can't infer from token stats
    if not config.victim_address:
        # Default to native token
        native_symbol = "ETH" if config.blockchain_name == "eth" else config.blockchain_name.upper()
        return native_symbol, 0

    stats = await client.token_stats(config.blockchain_name, config.victim_address)
    tokens = stats.get("tokens", [])

    if not tokens:
        return "ETH", 0

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
    # Mode 2: If tx_hash is provided and victim_address is None, we should have extracted it already
    # But if we're here and victim_address is still None, we need to extract it
    if not config.victim_address and config.tx_hash:
        victim_addr, extracted_token_id, extracted_asset, block_time = await extract_victim_from_tx_hash(
            config.tx_hash, config.blockchain_name, client
        )
        config.victim_address = victim_addr
        if extracted_token_id is not None and token_id == 0:
            token_id = extracted_token_id
        if extracted_asset and not asset_symbol:
            asset_symbol = extracted_asset

    # Ensure victim_address is set
    if not config.victim_address:
        raise ValueError("victim_address is required. Either provide it directly or provide tx_hash to extract it.")

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

    # If tx_hash is provided, prioritize finding it
    primary_tx_hash = config.tx_hash or (config.known_tx_hashes[0] if config.known_tx_hashes else None)

    if primary_tx_hash:
        logger.debug(f"Attempting to fetch primary tx directly: {primary_tx_hash}")
        try:
            # Try to fetch the specific transaction directly
            tx_detail = await client.get_transaction(
                config.victim_address,
                primary_tx_hash,
                config.blockchain_name,
                token_id=token_id,
                path="0"
            )

            # Handle text-wrapped JSON response
            if isinstance(tx_detail, dict) and "text" in tx_detail:
                try:
                    import json
                    tx_detail = json.loads(tx_detail["text"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Check success
            if tx_detail.get("success") is True:
                data = tx_detail.get("data", {})
                if data and data.get("hash", "").lower() == primary_tx_hash.lower():
                    logger.debug(f"Found primary tx directly via get-transaction")

                    # Use the amount from the transaction details
                    # Note: amount_coerced usually contains the value
                    amount = abs(float(data.get("amount_coerced") or 0))

                    return [{
                        "tx_hash": data.get("hash"),
                        "time": data.get("block_time"),
                        "amount": amount,
                        "to": data.get("output", {}).get("address"),
                        "path": data.get("path", "0"),
                        "token_id": data.get("token_id", token_id),
                        "asset_symbol": asset_symbol
                    }]
        except Exception as e:
            logger.debug(f"Failed to fetch primary tx directly: {e}")

    # Fetch more transactions to increase chance of finding user-provided hash
    response = await client.all_txs(config.victim_address, config.blockchain_name, filter_criteria, limit=20)
    raw_txs = response.get("data", [])

    # 1. Check for User-Provided Hashes (Priority)
    if config.known_tx_hashes or primary_tx_hash:
        known_set = set(h.lower() for h in (config.known_tx_hashes + ([primary_tx_hash] if primary_tx_hash else [])))
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
