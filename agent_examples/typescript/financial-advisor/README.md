# Financial Advisor Agent — LangChain + Ollama

An AI agent that answers educational questions about personal finance. Built with [LangChain](https://js.langchain.com) for agent orchestration and [Ollama](https://ollama.com) for local LLM inference. Connects to the 101Agents LLP platform via WebSocket.

## What it does

- Answers questions about investments, budgeting, retirement, tax, debt, and savings
- Loads and reads PDF invoice attachments when provided
- Returns structured responses with category, risk level, recommendation, and key considerations
- Declines out-of-domain questions politely
- All responses are educational — not personalized financial advice

## Prerequisites

- [Node.js](https://nodejs.org) 18+
- [Ollama](https://ollama.com) running locally with a model pulled (default: `llama3.1`)
- An LLP platform instance and agent credentials

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Copy the example env file and fill in your values:
   ```bash
   cp .env.example .env
   ```

3. Pull the Ollama model if you haven't already:
   ```bash
   ollama pull llama3.1
   ```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_NAME` | `financial-advisor` | Agent name registered on the LLP platform |
| `AGENT_KEY` | — | API key for the LLP platform |
| `PLATFORM_ADDRESS` | `ws://localhost:4000/agent/websocket` | LLP platform WebSocket URL |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_API_KEY` | — | Optional — only required if your Ollama instance uses authentication |

## Running

```bash
npm start
```

The agent connects to the LLP platform and waits for incoming messages. Press `Ctrl+C` to disconnect.

## Framework

This example uses [LangChain](https://js.langchain.com) (`@langchain/core`, `@langchain/ollama`). For a comparable agent built with Mastra, see [`../loan-advisor`](../loan-advisor).
