import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
from interaction_checker import flag_interactions, explain_interaction, G

st.set_page_config(page_title="DDI Checker", layout="wide")
st.title("Drug–Drug Interaction Checker (Graph + RAG)")

st.sidebar.header("Settings")
st.sidebar.markdown("Set `OPENAI_API_KEY` as env var to enable RAG explanations.")

meds_text = st.text_area("Enter medications (comma separated)", "Warfarin, Aspirin, Metformin")
if st.button("Check Interactions"):
    meds = [m.strip() for m in meds_text.split(",") if m.strip()]
    flagged = flag_interactions(meds)
    if not flagged:
        st.success("No flagged interactions found in the local graph.")
    else:
        st.subheader("Flagged Pairs")
        for f in flagged:
            with st.expander(f"{f['drug_a']} — {f['drug_b']} ({f['severity']})"):
                st.write("**Mechanism:**", f.get("mechanism"))
                st.write("**Description:**", f.get("description"))
                # Show RAG-augmented explanation
                expl = explain_interaction(f["drug_a"], f["drug_b"])
                st.write("**Explanation (expanded):**")
                st.write(expl)

        # visualize subgraph of meds
        subG = nx.Graph()
        for m in meds:
            if G.has_node(m):
                subG.add_node(m)
        for f in flagged:
            a = f["drug_a"]
            b = f["drug_b"]
            if G.has_edge(a, b):
                subG.add_edge(a, b, **G.get_edge_data(a, b))

        if subG.number_of_nodes() > 0 and subG.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            pos = nx.spring_layout(subG)
            nx.draw(subG, pos, with_labels=True, node_color="lightblue", ax=ax)
            edge_labels = {(u, v): d.get("severity", "") for u, v, d in subG.edges(data=True)}
            nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, ax=ax)
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.image(buf)

st.markdown("---")
st.subheader("Ask about an interaction")
q = st.text_input("Ask a question (e.g., 'Why is warfarin + aspirin risky?')")
if q:
    # naive retrieval: return any edge descriptions matching drug names in query
    query = q.lower()
    hits = []
    for u, v, d in G.edges(data=True):
        text = f"{u} {v} {d.get('mechanism','')} {d.get('description','')}``.lower()
        if any(token in query for token in [u.lower(), v.lower()]):
            hits.append(d.get('description','') + ' ' + d.get('mechanism',''))
    if not hits:
        st.info("No direct matches found in local graph; try different terms.")
    else:
        # join top hits and optionally expand with LLM if available
        context = "\n---\n".join(hits)
        st.write("**Retrieved context:**")
        st.write(context)
        # Attempt simple expansion via LangChain/OpenAI if available
        import os
        if os.environ.get('OPENAI_API_KEY'):
            try:
                from langchain import OpenAI
                llm = OpenAI(temperature=0)
                prompt = f"You are a clinical assistant. Answer the question concisely using the facts below:\n\nFacts:\n{context}\n\nQuestion: {q}\nAnswer:" 
                resp = llm(prompt)
                st.write(resp)
            except Exception as e:
                st.error(f"RAG expansion failed: {e}")
        else:
            st.write("(Set OPENAI_API_KEY to enable expanded answers.)")
