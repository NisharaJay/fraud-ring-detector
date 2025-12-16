import torch
import pandas as pd
from sklearn.model_selection import train_test_split

df_labels = pd.read_csv("data/processed/node_labels.csv", index_col=0)
y = torch.tensor(df_labels['label'].values, dtype=torch.float)

nodes = torch.arange(len(y))
train_nodes, test_nodes = train_test_split(nodes, test_size=0.2, random_state=42, stratify=y)

train_mask = torch.zeros(len(y), dtype=torch.bool)
train_mask[train_nodes] = True

test_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask[test_nodes] = True