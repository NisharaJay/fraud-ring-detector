import pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def load_graph(path="data/processed/transaction_graph.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def basic_stats(G):
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

def degree_analysis(G, top_n=10):
    data = []
    for node in G.nodes():
        data.append({
            "account": node,
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node)
        })
    df = pd.DataFrame(data)
    print(df.sort_values("out_degree", ascending=False).head(top_n))

def detect_cycles_limited(G, max_cycles=5):
    cycles = []
    for cycle in nx.simple_cycles(G):
        cycles.append(cycle)
        if len(cycles) >= max_cycles:
            break
    print(f"Found {len(cycles)} cycles (limited)")
    for c in cycles:
        print(c)

def extract_fraud_subgraph(G):
    fraud_edges = [
        (u, v, k)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("is_fraud") == 1
    ]
    fraud_G = G.edge_subgraph(fraud_edges).copy()
    print("Fraud nodes:", fraud_G.number_of_nodes())
    print("Fraud edges:", fraud_G.number_of_edges())
    return fraud_G

def visualize_fraud_subgraph(fraud_G, max_nodes=20):
    nodes = list(fraud_G.nodes())[:max_nodes]
    small_subgraph = fraud_G.subgraph(nodes)

    plt.figure(figsize=(8, 6))
    nx.draw(
        small_subgraph,
        with_labels=False,
        node_size=50,
        alpha=0.7
    )
    plt.title("Fraud Ring Subgraph (Sample)")
    plt.show()


if __name__ == "__main__":
    G = load_graph()
    basic_stats(G)
    degree_analysis(G)
    detect_cycles_limited(G)
    fraud_G = extract_fraud_subgraph(G)
    visualize_fraud_subgraph(fraud_G)