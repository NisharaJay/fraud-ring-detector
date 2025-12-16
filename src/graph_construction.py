import pandas as pd
import networkx as nx
import pickle

#load data
df = pd.read_csv('data/raw/transactions.csv')

#create directed graph from data
G = nx.from_pandas_edgelist(
    df, 
    'from_account', 
    'to_account', 
    edge_attr=['amount', 'is_fraud'],
    create_using=nx.MultiDiGraph()
)

with open('data/processed/transaction_graph.pkl', 'wb') as f:
    pickle.dump(G, f)