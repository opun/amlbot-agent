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

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/amlbot-agent.git
   cd amlbot-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install .
   # For development (with testing tools):
   pip install .[dev]
   ```

   > **Note:** Requires Python 3.13 or higher.

3. **Set up environment variables:**
   - Create a `.env` file in the project root if you need to configure API keys or other secrets.

## Usage

Run the agent from the command line:

```bash
python -m agent.main
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
