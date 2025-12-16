import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from models import GraphSAGE

# Load graph
with open("data/processed/transaction_graph.pkl", "rb") as f:
    G = pickle.load(f)


# Load node features
df_features = pd.read_csv("data/processed/node_features.csv", index_col=0)
X = torch.tensor(df_features.values, dtype=torch.float)

# Map nodes to integers & edges
node_mapping = {node: i for i, node in enumerate(G.nodes())}
edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


# Load node labels
df_labels = pd.read_csv("data/processed/node_labels.csv", index_col=0)
y = torch.tensor(df_labels["label"].values, dtype=torch.float)


# Train split
nodes = torch.arange(len(y))
train_nodes, temp_nodes = train_test_split(
    nodes, test_size=0.3, random_state=42, stratify=y
)
val_nodes, test_nodes = train_test_split(
    temp_nodes, test_size=0.5, random_state=42, stratify=y[temp_nodes]
)

test_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask[test_nodes] = True


# PyG Data object
data = Data(x=X, edge_index=edge_index, y=y)
data.test_mask = test_mask


# Load trained model
model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=16,    
    dropout=0.6
)

model.load_state_dict(torch.load("models/graphsage_model.pth"))
model.eval()


# Load threshold
with open("models/threshold.txt", "r") as f:
    threshold = float(f.read().strip())


# Evaluate on test set
with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_out = out[data.test_mask]
    y_true = data.y[data.test_mask].int()
    y_pred = (torch.sigmoid(test_out) > threshold).int()


# Metrics
acc = accuracy_score(y_true.numpy(), y_pred.numpy())
precision = precision_score(y_true.numpy(), y_pred.numpy(), zero_division=0)
recall = recall_score(y_true.numpy(), y_pred.numpy(), zero_division=0)
f1 = f1_score(y_true.numpy(), y_pred.numpy(), zero_division=0)

print("==== Test Results ====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Threshold: {threshold:.2f}")