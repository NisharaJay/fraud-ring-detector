import networkx as nx
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from node2vec import Node2Vec

with open('data/processed/transaction_graph.pkl', 'rb') as f:
    G = pickle.load(f)

features = {}

#node level features
for node in tqdm(G.nodes()):
    in_deg = G.in_degree(node)
    out_deg = G.out_degree(node)
    
    total_sent = sum([d['amount'] for _, _, d in G.out_edges(node, data=True)])
    total_received = sum([d['amount'] for _, _, d in G.in_edges(node, data=True)])
    
    avg_sent = total_sent / out_deg if out_deg > 0 else 0
    avg_received = total_received / in_deg if in_deg > 0 else 0
    degree_ratio = out_deg / in_deg if in_deg > 0 else 0
    transaction_count = in_deg + out_deg
    
    max_sent = max([d['amount'] for _, _, d in G.out_edges(node, data=True)], default=0)
    min_sent = min([d['amount'] for _, _, d in G.out_edges(node, data=True)], default=0)
    std_sent = np.std([d['amount'] for _, _, d in G.out_edges(node, data=True)]) if out_deg > 0 else 0
    
    features[node] = {
        'in_degree': in_deg,
        'out_degree': out_deg,
        'total_sent': total_sent,
        'total_received': total_received,
        'avg_sent': avg_sent,
        'avg_received': avg_received,
        'degree_ratio': degree_ratio,
        'transaction_count': transaction_count,
        'max_sent': max_sent,
        'min_sent': min_sent,
        'std_sent': std_sent
    }


# Convert MultiDiGraph to DiGraph
H = nx.DiGraph()

for u, v, data in G.edges(data=True):
    if H.has_edge(u, v):
        # Optional: sum edge weights if multiple edges
        H[u][v]['amount'] += data.get('amount', 1)
    else:
        H.add_edge(u, v, **data)


#centrality features
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(H, max_iter=1000, tol=1e-06)

for node in G.nodes:
    features[node].update({
        'pagerank': pagerank[node],
        'betweenness': betweenness[node],
        'closeness': closeness[node],
        'eigenvector': eigenvector[node]
    })


# Convert to simple undirected graph for clustering/triangles
H_undirected = nx.Graph()
for u, v, data in G.to_undirected().edges(data=True):
    if H_undirected.has_edge(u, v):
        H_undirected[u][v]['amount'] += data.get('amount', 1)
    else:
        H_undirected.add_edge(u, v, **data)


#clustering features
clustering = nx.clustering(H_undirected)
triangles = nx.triangles(H_undirected)

for node in G.nodes():
    features[node].update({
        'clustering_coeff': clustering[node],
        'triangle_count': triangles[node]
    })


#Node2Vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=4, seed=42)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

for node in G.nodes:
    emb = model.wv[str(node)]
    for i, val in enumerate(emb):
        features[node][f'emb_{i}'] = val


#save features to a dataFrame
features_df = pd.DataFrame.from_dict(features, orient='index')
features_df.to_csv('data/processed/node_features.csv', index_label='node')