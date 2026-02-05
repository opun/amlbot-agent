# Local Setup Guide

This guide provides step-by-step instructions to get the **AMLBot Agent** running on your local machine.

## 1. Prerequisites

- **Python**: Version 3.12 or higher.
- **Docker**: Needed to run the MCP Server conveniently.
- **API Keys**: You will need an OpenAI API key and access to the AMLBot API.

---

## 2. Clone the Repository

Download the project code to your machine:

```bash
git clone https://github.com/yourusername/AMLBot.git
cd AMLBot
```

---

## 3. Set Up the MCP Server

The MCP Server is the data layer that talks to blockchain explorers.

### Build the Docker Image
```bash
cd mcp-server-amlbot
docker build -t mcp-server-amlbot .
```

### Run the Server
The agent expects the server to be running in the background. You can run it via Docker:

```bash
docker run --rm -i -e USER_ID=your_user_id mcp-server-amlbot
```

---

## 4. Set Up the Agent

The agent is the intelligence that performs the tracing logic.

### Create a Virtual Environment (Optional but Recommended)
```bash
cd ../amlbot-agent
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -e .
```

### Configure Environment Variables
Create a file named `.env` inside the `amlbot-agent` directory:

```env
OPENAI_API_KEY=sk-....            # Your OpenAI Key
SAILS_URL=https://api.amlbot.com  # AMLBot API Base URL
USER_ID=your-unique-id           # Your AMLBot User ID
```

---

---

## 5. Running the System

You can run both the API and the Frontend with a single command.

### One-Command Setup (Recommended)
This uses `npm` and `concurrently` to launch everything.

1. **Install Root Dependencies**:
   ```bash
   npm install
   ```
2. **Run Everything**:
   ```bash
   npm run dev
   ```
   - **Frontend**: `http://localhost:3333` (Port 3333 is used to avoid common conflicts)
   - **API Backend**: `http://localhost:8000`

### Manual Execution

If you prefer to run components separately:

#### A. API Backend
```bash
python -m agent.api
```

#### B. Frontend (Port 3333)
```bash
cd frontend
npm install
npm run dev -- -p 3333
```

#### C. Terminal CLI (Interactive)
```bash
python -m agent.cli
```

---

## Troubleshooting

- **"Module not found"**: Ensure you have activated your virtual environment and ran `pip install -e .`.
- **MCP Connection Errors**: Check that your Docker container is running and that your `USER_ID` is correct.
- **API Key Invalid**: Double check your `.env` file for typos.
