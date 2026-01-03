import pandas as pd
import networkx as nx


def load_graph_from_csv(csv_path: str) -> nx.Graph:
    """Load interactions CSV into a NetworkX graph.

    CSV expected columns: drug_1, drug_2, interaction (Yes/No), mechanism, severity, description
    """
    df = pd.read_csv(csv_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        d1 = str(row["drug_1"]).strip()
        d2 = str(row["drug_2"]).strip()
        attrs = {
            "interaction": row.get("interaction", "Yes"),
            "mechanism": row.get("mechanism", ""),
            "severity": row.get("severity", ""),
            "description": row.get("description", ""),
        }
        G.add_node(d1)
        G.add_node(d2)
        G.add_edge(d1, d2, **attrs)
    return G


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ddinter_sample.csv"
    G = load_graph_from_csv(path)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
