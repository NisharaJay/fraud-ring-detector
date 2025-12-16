import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()  # For binary classification

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer
        x = self.conv2(x, edge_index)

        # Sigmoid for binary classification
        x = self.out_activation(x)
        return x.squeeze()


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=16, out_channels=1, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.activation = nn.ReLU()
        self.out_activation = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.out_activation(x)
        return x.squeeze()