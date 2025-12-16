import pandas as pd

df = pd.read_csv("data/processed/detected_rings.csv")

#Fraud ring statistics
print("Total rings:", len(df))
print("Avg ring size:", df["num_nodes"].mean())
print("Max ring size:", df["num_nodes"].max())
print("Min ring size:", df["num_nodes"].min())