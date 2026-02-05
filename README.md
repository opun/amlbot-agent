# amlbot-agent

OpenAI agent for tracing simple blockchain cases.

## Overview

**amlbot-agent** is an OpenAI-powered agent designed to trace simple blockchain cases, such as tracking the flow of funds from a victim wallet through various entities (bridges, mixers, CEXs, etc.). The agent automates the process of gathering transaction data, classifying entities, and visualizing the trace as a graph.

## Features

- Interactive CLI for entering case descriptions and wallet addresses
- Automated tracing of outgoing transactions up to 10 hops
- Entity classification (Victim, Scammer, Bridge, Mixer, CEX, P2P, Unknown)
- Risk scoring and color-coded graph output (Mermaid format)
- Extensible agent-based architecture

## Getting Started

To run the amlbot-agent system locally, you need to set up both the **MCP Server** (which provides access to the blockchain data) and the **Agent** (which performs the analysis).

For a complete step-by-step guide for beginners, see our [Local Setup Guide](file:///Users/opun/GitHub/AMLBot/amlbot-agent/docs/LOCAL_SETUP.md).

## Installation

### 1. Prerequisites
- Python 3.12 or higher (3.13 recommended)
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- [Docker](https://www.docker.com/) (optional, for running MCP server)

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/AMLBot.git
cd AMLBot
```

### 3. Setup MCP Server
Follow the instructions in the [mcp-server-amlbot README](file:///Users/opun/GitHub/AMLBot/mcp-server-amlbot/README.md) to build and run the data server.

### 4. Setup Agent
```bash
cd amlbot-agent
pip install .
# or if using uv:
uv pip install -e .
```

## Usage

### 1. Quick Run (All-in-One)
Launch both the API and the Frontend on port 3333:
```bash
npm install
npm run dev
```

### 2. Manual Configuration
#### Configure Environment
Create a `.env` file in the `amlbot-agent` directory:
```env
OPENAI_API_KEY=your_key_here
SAILS_URL=https://api.amlbot.com
USER_ID=your_user_id
```

#### Run the Agent CLI
```bash
python -m agent.cli
```

#### Run the API Backend
```bash
python -m agent.api
```

You will be prompted to enter:
- A simple case description
- The victim wallet address

The agent will then:
- Trace the flow of funds
- Classify entities
- Output a Mermaid graph code block for visualization

### Example

```
$ python -m agent.main
Give the simple case description: Stolen funds from wallet 0xABC... traced through Ethereum.
Give the victim wallet address: 0xABCDEF1234567890
...
Graph: ```mermaid
flowchart TD
...
```

## Project Structure

```
amlbot-agent/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── main.py
│       └── tracer.py
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

## Development

- Run tests with:
  ```bash
  pytest
  ```

- Lint and format code as needed.

## License

MIT License 
