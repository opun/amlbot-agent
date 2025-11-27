import json
from typing import Any, Dict, Optional
from agents import Agent, Runner
from agents.mcp import MCPServer

class MCPClient:
    def __init__(self, mcp_server: MCPServer):
        self.server = mcp_server
        self.tool_exec = self._create_tool_exec_agent()

    def _create_tool_exec_agent(self) -> Agent:
        instructions = """You are a tool execution helper.
Your job is to execute blockchain tools when requested and return the raw JSON output.
Do not add any commentary or markdown formatting.
Return the tool output as-is."""
        return Agent(
            name="tool_exec_agent",
            instructions=instructions,
            mcp_servers=[self.server],
        )

    async def _execute(self, instruction: str) -> Dict[str, Any]:
        result = await Runner.run(self.tool_exec, input=instruction)
        text = result.final_output.strip()
        # Remove markdown fences if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Sometimes tools return text or empty
            return {"raw_output": text}

    async def all_txs(self, address: str, blockchain_name: str, filter_criteria: Optional[Dict] = None, limit: int = 100, offset: int = 0, direction: str = "asc", order: str = "time", transaction_type: str = "all") -> Dict[str, Any]:
        filter_json = json.dumps(filter_criteria) if filter_criteria else "null"
        instruction = f'Use all-txs with address="{address}", blockchain_name="{blockchain_name}", filter={filter_json}, limit={limit}, offset={offset}, direction="{direction}", order="{order}", transaction_type="{transaction_type}"'
        return await self._execute(instruction)

    async def get_transaction(self, address: str, tx_hash: str, blockchain_name: str, token_id: int = 0, path: str = "0") -> Dict[str, Any]:
        instruction = f'Use get-transaction with address="{address}", tx_hash="{tx_hash}", blockchain_name="{blockchain_name}", token_id={token_id}, path="{path}"'
        return await self._execute(instruction)

    async def get_address(self, blockchain_name: str, address: str) -> Dict[str, Any]:
        instruction = f'Use get-address with blockchain_name="{blockchain_name}", address="{address}"'
        return await self._execute(instruction)

    async def token_stats(self, blockchain_name: str, address: str) -> Dict[str, Any]:
        instruction = f'Use token-stats with blockchain_name="{blockchain_name}", address="{address}"'
        return await self._execute(instruction)

    async def get_extra_address_info(self, address: str, blockchain_name: str) -> Dict[str, Any]:
        # Note: The tool parameter is named 'asset', but the API expects the blockchain/currency code (e.g. 'ETH')
        instruction = f'Use get-extra-address-info with address="{address}", asset="{blockchain_name}"'
        return await self._execute(instruction)

    async def bridge_analyzer(self, chain: str, tx_hash: str) -> Dict[str, Any]:
        instruction = f'Use bridge-analyzer with chain="{chain}", tx_hash="{tx_hash}"'
        return await self._execute(instruction)
