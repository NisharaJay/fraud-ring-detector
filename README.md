# Graph-based Fraud Ring Detection in Transaction Networks

### Project Overview

A Graph-based ML pipeline for fraud detection in transaction networks that uses a Graph Neural Network (GNN) approach to detect fraud rings groups of accounts that collaborate to conduct fraudulent transactions. Accounts are represented as nodes and transactions as edges, forming a graph structure. A GraphSAGE model is trained to identify fraudulent nodes, enabling the detection of coordinated fraud activity across the network.

### Key Features
- Generates a synthetic transaction dataset with normal and fraudulent transactions
- Extracts node-level features (in-degree, out-degree, transaction amounts)
- Constructs a transaction graph and assigns fraud labels to nodes
- Implements a GraphSAGE model for node-level binary classification
- Handles class imbalance with Focal Loss
- Performs train/validation/test split and evaluates with dynamic thresholds
- Detects fraud rings from predicted fraudulent nodes
- Provides analysis:
    Number of detected fraud rings
    Ring sizes (min, max, average)
    False positives and false negatives
    Visualizes detected fraud rings

### Technologies Used
- Python
- PyTorch
- PyTorch Geometric (PyG) - graph neural network library
- Pandas & NumPy for data processing and numerical operations
- scikit-learn for metrics, train-test split
- Matplotlib / NetworkX for visualization of graphs and fraud rings
- Faker for synthetic dataset generation

### Metrics
- F1-score: 0.8462
- Detected fraud rings: 6
- Average ring size: 15.5 nodes