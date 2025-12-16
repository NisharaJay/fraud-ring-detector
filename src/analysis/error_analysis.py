import pandas as pd

df = pd.read_csv("data/processed/predicted_nodes.csv")

fp = ((df.true_label == 0) & (df.pred_label == 1)).sum()
fn = ((df.true_label == 1) & (df.pred_label == 0)).sum()

print("False Positives:", fp)
print("False Negatives:", fn)