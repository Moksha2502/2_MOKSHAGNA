from typing import List, Tuple, Dict
import networkx as nx
from graph_builder import load_graph_from_csv
import os

# Load graph once
DATA_CSV = os.environ.get("DDI_CSV", "data/ddinter_sample.csv")
G = load_graph_from_csv(DATA_CSV)


def flag_interactions(meds: List[str]) -> List[Dict]:
    """Given a list of medication names, return flagged interaction pairs with attributes.

    Returns a list of dicts: {drug_a, drug_b, mechanism, severity, description}
    """
    meds_clean = [m.strip() for m in meds if m and m.strip()]
    flagged = []
    for i in range(len(meds_clean)):
        for j in range(i + 1, len(meds_clean)):
            a = meds_clean[i]
            b = meds_clean[j]
            if G.has_edge(a, b):
                e = G.get_edge_data(a, b)
                if e and str(e.get("interaction", "Yes")).lower() in {"yes", "true", "1"}:
                    flagged.append({
                        "drug_a": a,
                        "drug_b": b,
                        "mechanism": e.get("mechanism", ""),
                        "severity": e.get("severity", ""),
                        "description": e.get("description", ""),
                    })
    return flagged


def explain_interaction(drug_a: str, drug_b: str) -> str:
    """Return a concise explanation for the interaction from graph data.

    If OpenAI key present, optionally expand with a short RAG response via LangChain.
    """
    if not G.has_edge(drug_a, drug_b):
        return "No known interaction in the local graph dataset."
    e = G.get_edge_data(drug_a, drug_b)
    base = f"Interaction between {drug_a} and {drug_b}: {e.get('description','')}. Mechanism: {e.get('mechanism','')} Severity: {e.get('severity','')}"

    # Optional RAG expansion using LangChain + OpenAI if key provided
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return base

    try:
        from langchain import OpenAI
        from langchain.prompts import PromptTemplate
        llm = OpenAI(temperature=0)
        prompt = PromptTemplate(
            input_variables=["base"],
            template=("You are a clinical pharmacist assistant. Take the following factual interaction summary and produce "
                      "a 2-3 sentence explanation suitable for clinicians, including actionable guidance if applicable:\n\n{base}")
        )
        resp = llm(prompt.format_prompt(base=base).to_string())
        return resp
    except Exception as ex:
        return base + f"\n(Note: RAG expansion failed: {ex})"


if __name__ == "__main__":
    # quick demo
    demo = ["Warfarin", "Aspirin", "Metformin"]
    flagged = flag_interactions(demo)
    for f in flagged:
        print(f)
        print(explain_interaction(f["drug_a"], f["drug_b"]))
