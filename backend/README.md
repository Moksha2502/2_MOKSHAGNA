Drug–Drug Interaction Checker (Graph + RAG)

This small demo builds a NetworkX graph from a sample DDInter CSV, flags risky pairs from a user-provided medication list, and offers explanations. A Streamlit frontend provides a simple chatbot and visualization.

Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Run

```bash
cd /workspace/ddi_checker
streamlit run app.py
```

Notes

- The repo includes a tiny sample `data/ddinter_sample.csv`. For full dataset, download DDInter from Kaggle and replace the CSV path.
- Optional: set `OPENAI_API_KEY` in your environment to enable RAG responses via LangChain/OpenAI.
- Neo4j integration is intentionally left as an optional extension in `graph_builder.py`.
