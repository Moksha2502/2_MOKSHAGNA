# Quick Start: OpenAI Chatbot with RAG Setup

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (for the web interface)
- langchain-openai (for OpenAI integration)
- openai (OpenAI Python SDK)
- All other dependencies including RAG components

## Step 2: Set Up OpenAI API Key

You have three options:

### Option A: Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Option B: Create .env File

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Option C: Use Streamlit Interface

You can also enter your API key directly in the Streamlit app sidebar (it's stored in session, not saved to disk).

## Step 3: Run the Chatbot

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Step 4: Use the Chatbot

1. Enter your OpenAI API key in the sidebar (if not using environment variable)
2. Click "Initialize Chatbot"
3. Start asking questions about drug interactions!

### Example Queries:

- "Check interactions between Aspirin and Warfarin"
- "What are the risks of taking Ibuprofen with blood thinners?"
- "Tell me about Abacavir and Naltrexone interactions"
- "Can I take Digoxin with Furosemide?"
- "Explain the mechanism of interaction between Dolutegravir and Aluminum hydroxide"

## Features

✅ **RAG-Powered** - Retrieval-Augmented Generation for accurate responses
✅ **OpenAI Integration** - Uses OpenAI embeddings and GPT models
✅ **Interactive Chat Interface** - Natural language queries
✅ **Backend Integration** - Uses real drug interaction database (56,367 interactions)
✅ **Real-time Checking** - Automatically detects and checks medications
✅ **Visual Display** - Shows detailed interaction information with risk levels
✅ **Enhanced Explanations** - LLM generates comprehensive explanations using retrieved context

## How RAG Works

1. **Retrieval**: When you ask about drug interactions, the system uses OpenAI embeddings to search the vector database
2. **Augmentation**: The retrieved relevant interactions are combined with your question
3. **Generation**: OpenAI LLM (GPT-3.5-turbo) generates a comprehensive response using the retrieved context

## Troubleshooting

**Issue: "langchain-openai not installed"**
- Solution: `pip install langchain-openai openai`

**Issue: "Streamlit not found"**
- Solution: `pip install streamlit`

**Issue: "API key not provided"**
- Solution: Make sure you've set the OPENAI_API_KEY environment variable or entered it in the sidebar

**Issue: Module import errors**
- Solution: Make sure you're in the project root directory and all dependencies are installed

**Issue: OpenAI API errors**
- Solution: Check your API key is valid and you have credits/quota available

