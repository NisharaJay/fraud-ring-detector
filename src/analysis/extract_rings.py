import pickle
import pandas as pd
import networkx as nx

# Load graph
with open("data/processed/transaction_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load predicted fraud nodes
df = pd.read_csv("data/processed/predicted_nodes.csv")
fraud_nodes = df[df["pred_label"] == 1]["node"].tolist()

# Create fraud subgraph
fraud_subgraph = G.subgraph(fraud_nodes).to_undirected()

# Detect rings (connected components)
rings = list(nx.connected_components(fraud_subgraph))

ring_data = []
for i, ring in enumerate(rings):
    ring_data.append({
        "ring_id": i,
        "num_nodes": len(ring),
        "nodes": list(ring)
    })

df_rings = pd.DataFrame(ring_data)
df_rings.to_csv("data/processed/detected_rings.csv", index=False)

print(f"Detected {len(rings)} fraud rings")