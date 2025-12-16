import pickle
import pandas as pd

with open('data/processed/transaction_graph.pkl', 'rb') as f:
    G = pickle.load(f)

# For MultiDiGraph, include keys=True
fraud_nodes = set()
for u, v, k, d in G.edges(keys=True, data=True):
    if d.get("is_fraud", 0) == 1:
        fraud_nodes.add(u)
        fraud_nodes.add(v)

# Create labels for all nodes
labels = {node: 1 if node in fraud_nodes else 0 for node in G.nodes()}

df_labels = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])
df_labels.to_csv('data/processed/node_labels.csv', index_label='node')

print(df_labels.head())
print("Total fraud nodes:", sum(labels.values()))
