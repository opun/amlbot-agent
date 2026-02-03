import json
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from agents import Agent, Runner, gen_trace_id

# Setup logger for tracer
logger = logging.getLogger("tracer")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[TRACE] %(message)s'))
    logger.addHandler(handler)

from agent.models import (
    TracerConfig,
    TraceResult,
    CaseMeta,
    TraceStats
)
from agent.mcp_client import MCPClient
from agent.theft_detection import (
    infer_asset_symbol,
    infer_approx_date_from_description,
    extract_victim_from_tx_hash
)
from agent.visualization import generate_visualization_payload
class CryptoTracer:
    def __init__(self, client: MCPClient):
        self.client = client
        self.prompt_path = Path(__file__).parent / "prompts" / "trace_orchestrator.md"

    def _load_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        return self.prompt_path.read_text(encoding="utf-8")

    def _strip_code_fences(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    async def _run_orchestrator(self, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the LLM orchestrator with MCP tools and return parsed JSON."""
        agent = Agent(
            name="trace_orchestrator",
            instructions=prompt,
            mcp_servers=[self.client.server],
        )
        logger.debug("Starting LLM orchestrator with MCP tools...")
        result = await Runner.run(agent, input=json.dumps(payload, indent=2), max_turns=60)
        raw_output = result.final_output or ""
        cleaned = self._strip_code_fences(raw_output)
        logger.debug("LLM orchestrator completed; parsing output...")
        return json.loads(cleaned)

    async def trace(self, config: TracerConfig) -> TraceResult:
        # 0. Setup
        case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        trace_id = gen_trace_id()

        # Extract victim from tx_hash when possible
        token_id_hint: Optional[int] = None
        if not config.victim_address and config.tx_hash:
            logger.debug(f"Extracting victim address from tx_hash: {config.tx_hash}")
            victim_addr, extracted_token_id, extracted_asset, block_time = await extract_victim_from_tx_hash(
                config.tx_hash, config.blockchain_name, self.client
            )
            config.victim_address = victim_addr
            token_id_hint = extracted_token_id
            if extracted_asset and not config.asset_symbol:
                config.asset_symbol = extracted_asset
            if block_time and not config.approx_date:
                try:
                    dt = datetime.fromtimestamp(block_time)
                    config.approx_date = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        if not config.victim_address:
            raise ValueError("victim_address is required. Provide tx_hash or wallet address.")

        if not config.approx_date and config.description:
            config.approx_date = infer_approx_date_from_description(config.description)

        asset_symbol, detected_token_id = await infer_asset_symbol(config, self.client)
        config.asset_symbol = asset_symbol
        if token_id_hint is None:
            token_id_hint = detected_token_id

        case_meta = CaseMeta(
            case_id=case_id,
            trace_id=trace_id,
            description=config.description or "",
            victim_address=config.victim_address,
            blockchain_name=config.blockchain_name,
            chains=[config.blockchain_name],
            asset_symbol=asset_symbol,
            approx_date=config.approx_date,
        )

        prompt_template = self._load_prompt()
        prompt_text = prompt_template.format(
            victim_address=config.victim_address or "",
            tx_hash=config.tx_hash or "",
            blockchain_name=config.blockchain_name,
            asset_symbol=config.asset_symbol or "",
            approx_date=config.approx_date or "",
            description=config.description or "",
        )

        payload = {
            "case_meta": case_meta.model_dump(),
            "token_id_hint": token_id_hint,
            "known_tx_hashes": config.known_tx_hashes,
            "inputs": {
                "victim_address": config.victim_address,
                "tx_hash": config.tx_hash,
                "blockchain_name": config.blockchain_name,
                "asset_symbol": config.asset_symbol,
                "approx_date": config.approx_date,
                "description": config.description,
            },
            "rules_version": "orchestrator-1",
        }

        logger.debug(f"Prompt loaded from {self.prompt_path}")
        logger.debug(
            "LLM input context: victim_address=%s tx_hash=%s blockchain=%s asset=%s approx_date=%s",
            config.victim_address,
            config.tx_hash,
            config.blockchain_name,
            config.asset_symbol,
            config.approx_date,
        )

        llm_output = await self._run_orchestrator(prompt_text, payload)

        try:
            trace_result = TraceResult.model_validate(llm_output)
        except Exception as exc:
            raise ValueError(f"LLM output could not be parsed into TraceResult: {exc}") from exc

        # Attach case meta if missing fields
        trace_result.case_meta = trace_result.case_meta or case_meta
        if not trace_result.case_meta.trace_id:
            trace_result.case_meta.trace_id = trace_id

        # --- Visualization Generation & Saving ---
        try:
            logger.info("Generating visualization...")
            viz_payload = generate_visualization_payload(trace_result)
            
            # Note: client.save_and_share_visualization has been added to MCPClient
            viz_result = await self.client.save_and_share_visualization(viz_payload)

            share_url = viz_result.get("share_url") or viz_result.get("data", {}).get("share_url") or viz_result.get("url")
            # Also check if it's nested in result or output
            if not share_url and "result" in viz_result:
                share_url = viz_result["result"].get("share_url")
            
            if share_url:
                logger.info(f"Visualization Link: {share_url}")
                trace_result.visualization_url = share_url
            else:
                 # Try to construct it if we have an ID but no URL? 
                 # Actually save_and_share tool should return it
                 logger.warning(f"Visualization saved but no share URL found in result keys: {viz_result.keys()}")

        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")

        return trace_result
