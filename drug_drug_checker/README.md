# Drug-Drug Interaction Checker (Graph + RAG)

A Python application that checks for drug-drug interactions using a graph-based approach (NetworkX) and provides explanations using RAG (Retrieval-Augmented Generation) with LangChain.

## Features

- **Graph-based Interaction Detection**: Uses NetworkX to model drug-drug interactions as a graph
- **RAG-powered Explanations**: Uses LangChain to generate explanations for interaction mechanisms
- **Risk Flagging**: Identifies and flags risky drug interaction pairs
- **DDInter Dataset Support**: Compatible with DDInter (Drug-Drug Interactions) dataset format

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional, for OpenAI API):
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY if using OpenAI embeddings/LLM
```

## Dataset

The application expects DDInter dataset format. Download from:
https://www.kaggle.com/datasets/montassarba/drug-drug-interactions-database-ddinter

Place the dataset files in the `data/` directory, or use the sample data provided.

## Usage

### Basic Usage

```python
from drug_interaction_checker import DrugInteractionChecker

# Initialize checker
checker = DrugInteractionChecker(data_path="data/ddinter_data.csv")

# Check for interactions
medications = ["Aspirin", "Warfarin", "Ibuprofen"]
results = checker.check_interactions(medications)

# Print results
for interaction in results:
    print(f"⚠️  {interaction['drug1']} + {interaction['drug2']}")
    print(f"   Risk Level: {interaction['risk_level']}")
    print(f"   Mechanism: {interaction['mechanism']}")
    print(f"   Explanation: {interaction['explanation']}\n")
```

### Command Line Usage

```bash
python main.py --meds "Aspirin,Warfarin,Ibuprofen"
```

## Project Structure

```
drug_drug_checker/
├── data/                          # Dataset directory
│   └── README.md                  # Data directory instructions
├── drug_interaction_checker.py    # Main checker class
├── graph_builder.py               # NetworkX graph construction
├── rag_system.py                  # RAG system for explanations
├── data_loader.py                 # Data loading utilities
├── main.py                        # CLI interface
├── example_usage.py               # Example usage scripts
├── test_basic.py                  # Basic functionality tests
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## How It Works

1. **Graph Construction**: Drug interactions are modeled as an undirected graph where nodes are drugs and edges represent interactions
2. **Interaction Detection**: Given a list of medications, the system checks all pairs against the graph
3. **RAG Explanation**: For each detected interaction, the RAG system retrieves relevant information and generates an explanation

## Chatbot Interface (Streamlit)

The project includes a Streamlit chatbot interface powered by Google Gemini AI:

### Setup

1. Set your Gemini API key:
```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your_api_key_here"

# Option 2: Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

3. Open your browser to the URL shown (usually http://localhost:8501)

### Chatbot Features

- **Interactive Chat Interface**: Natural language queries about drug interactions
- **Backend Integration**: Uses the drug interaction checker for accurate results
- **Gemini AI Powered**: Intelligent responses and explanations
- **Real-time Interaction Checking**: Automatically detects and checks medications mentioned
- **Visual Interaction Display**: Shows detailed interaction information with risk levels

### Example Queries

- "Check interactions between Aspirin and Warfarin"
- "What are the risks of taking Ibuprofen with blood thinners?"
- "Tell me about Abacavir and Naltrexone interactions"
- "Can I take Digoxin with Furosemide?"

## License

MIT

