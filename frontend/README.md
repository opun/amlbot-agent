# AMLBot Chat UI

A modern chat interface for the AMLBot Crypto Tracer agent, built with Next.js and inspired by OpenAI's ChatKit.

## Features

- 💬 Chat-style interface for natural language case descriptions
- 🔄 Real-time streaming updates during trace analysis
- 📊 Visual trace tree output
- 🔗 Direct links to OpenAI trace dashboard
- 🌙 Dark mode design

## Prerequisites

- Node.js 18+ or Bun
- The AMLBot backend API running on port 8000

## Quick Start

1. **Install dependencies:**

```bash
npm install
# or
bun install
```

2. **Start the backend API** (in the parent directory):

```bash
cd ..
uv run amlbot-api
```

3. **Start the frontend:**

```bash
npm run dev
# or
bun dev
```

4. **Open your browser:**

Navigate to [http://localhost:3000](http://localhost:3000)

## Usage

Simply describe your case in natural language:

> "Trace stolen funds from wallet 0x1234...abcd on Ethereum. The theft occurred around December 1st, 2024 and involved approximately 50,000 USDT."

The AI will:
1. Parse your description to extract key details
2. Ask for clarification if needed
3. Run the trace analysis
4. Display results with an interactive trace tree

## Configuration

The frontend proxies API requests to `http://localhost:8000`. To change this, edit `next.config.ts`:

```typescript
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://your-api-url/api/:path*",
      },
    ];
  },
};
```

## Tech Stack

- [Next.js 15](https://nextjs.org/) - React framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [React Markdown](https://github.com/remarkjs/react-markdown) - Markdown rendering
- [Lucide React](https://lucide.dev/) - Icons
