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
    7. **known_tx_hashes**: Any additional transaction hashes mentioned (comma-separated)

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
    Extract victim address from a transaction hash using token-transfers.
    Returns: (victim_address, token_id, asset_symbol, block_time)
    """
    logger.debug(f"Extracting victim address from tx_hash: {tx_hash} using token-transfers")

    try:
        # Use token-transfers tool
        result = await client.token_transfers(tx_hash, blockchain_name)

        # Handle text-wrapped JSON response
        if isinstance(result, dict) and "text" in result:
             try:
                import json
                result = json.loads(result["text"])
             except (json.JSONDecodeError, TypeError):
                pass

        data = result.get("data", [])
        if not data:
            raise ValueError(f"No transfer data found for transaction {tx_hash}")

        # Use the first transfer
        transfer = data[0]

        # Extract details
        # The input address is the sender, which is typically the victim in a theft context
        input_data = transfer.get("input", {})
        victim_address = input_data.get("address") if isinstance(input_data, dict) else None

        if not victim_address:
             raise ValueError(f"Could not find input address in transfer data for {tx_hash}")

        token_id = transfer.get("token_id", 0)
        block_time = transfer.get("block_time")

        # Infer asset symbol
        asset_symbol = None
        if token_id == 0:
             asset_symbol = "ETH" if blockchain_name == "eth" else blockchain_name.upper()

        logger.debug(f"Found victim address: {victim_address}, token_id: {token_id}")
        return victim_address, token_id, asset_symbol, block_time

    except Exception as e:
        logger.error(f"Error extracting victim from tx_hash {tx_hash}: {e}")
        raise ValueError(f"Failed to extract victim from transaction {tx_hash}: {e}")

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
