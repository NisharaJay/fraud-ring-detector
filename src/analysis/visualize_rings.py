import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load graph
with open("data/processed/transaction_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load predictions
df = pd.read_csv("data/processed/predicted_nodes.csv")

fraud_nodes = df[df["pred_label"] == 1]["node"].tolist()
normal_nodes = df[df["pred_label"] == 0]["node"].tolist()

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=10, alpha=0.2)
nx.draw_networkx_nodes(G, pos, nodelist=fraud_nodes, node_color="red", node_size=40)
nx.draw_networkx_edges(G, pos, alpha=0.1)

plt.title("Detected Fraud Rings")
plt.show()