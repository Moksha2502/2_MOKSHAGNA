# OpenRouter API Setup Guide

## What is OpenRouter?

OpenRouter is a unified API that provides access to multiple LLM providers (OpenAI, Anthropic, Google, etc.) through a single interface. This makes it easy to switch between different models.

## Getting Your API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up or log in
3. Navigate to your API keys section
4. Create a new API key
5. Copy your API key (starts with `sk-or-v1-...`)

## Setting Up the API Key

### Option 1: Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-api-key-here"
```

### Option 2: .env File

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
```

### Option 3: Streamlit Interface

Enter your API key directly in the Streamlit app sidebar.

## Available Models

The chatbot supports various models through OpenRouter:

- **OpenAI Models:**
  - `openai/gpt-3.5-turbo` (default, fast and cost-effective)
  - `openai/gpt-4` (more capable, higher cost)
  - `openai/gpt-4-turbo`

- **Anthropic Models:**
  - `anthropic/claude-3-haiku` (fast)
  - `anthropic/claude-3-sonnet` (balanced)
  - `anthropic/claude-3-opus` (most capable)

- **Google Models:**
  - `google/gemini-pro`
  - `google/gemini-pro-vision`

- **Other Models:**
  - Many more available on OpenRouter

## Using the Chatbot

1. Set your OpenRouter API key
2. Run: `streamlit run streamlit_app.py`
3. In the sidebar:
   - Check "Use OpenRouter API"
   - Select your preferred model
   - Enter your API key (or use environment variable)
   - Click "Initialize Chatbot"

## Benefits of OpenRouter

✅ **Single API Key**: Access multiple LLM providers
✅ **Easy Model Switching**: Change models without changing code
✅ **Cost Comparison**: See pricing for different models
✅ **Unified Interface**: Same API format for all providers
✅ **Rate Limiting**: Built-in rate limiting and management

