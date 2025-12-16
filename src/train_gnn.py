import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np


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


# Node labels
df_labels = pd.read_csv("data/processed/node_labels.csv", index_col=0)
y = torch.tensor(df_labels["label"].values, dtype=torch.float)


# Train/validation/test split
nodes = torch.arange(len(y))
train_nodes, temp_nodes = train_test_split(nodes, test_size=0.3, random_state=42, stratify=y)
val_nodes, test_nodes = train_test_split(temp_nodes, test_size=0.5, random_state=42, stratify=y[temp_nodes])

train_mask = torch.zeros(len(y), dtype=torch.bool)
train_mask[train_nodes] = True

val_mask = torch.zeros(len(y), dtype=torch.bool)
val_mask[val_nodes] = True

test_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask[test_nodes] = True


# PyG Data object
data = Data(x=X, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


# GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, dropout=0.6):
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze()

model = GraphSAGE(data.num_node_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# Focal loss for imbalance
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

criterion = FocalLoss()


# Training loop with early stopping
epochs = 100
patience = 10
no_improve = 0
best_f1 = 0
best_epoch = 0
best_threshold = 0.5

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = out[data.val_mask]
        y_true = data.y[data.val_mask].int()
        best_epoch_f1 = 0
        best_epoch_thresh = 0.5
        # Try thresholds 0.3 to 0.7
        for t in np.arange(0.3, 0.8, 0.05):
            y_pred = (torch.sigmoid(val_out) > t).int()
            f1 = f1_score(y_true.numpy(), y_pred.numpy(), zero_division=0)
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_thresh = t

        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = best_epoch_thresh
            no_improve = 0
            torch.save(model.state_dict(), "models/graphsage_model.pth")
            best_epoch = epoch
        else:
            no_improve += 1

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val F1: {best_epoch_f1:.4f}, Best Threshold: {best_epoch_thresh:.2f}")

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break


# Save threshold
with open("models/threshold.txt", "w") as f:
    f.write(str(best_threshold))

print(f"Training complete. Best F1: {best_f1:.4f} at epoch {best_epoch}, Threshold: {best_threshold:.2f}")