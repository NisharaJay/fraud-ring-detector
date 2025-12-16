import pandas as pd
import random
from faker import Faker

fake = Faker()

num_accounts = 1000
num_transactions = 5000
num_fraud_rings = 10       
fraud_ring_size = 10      
fraud_tx_per_ring = 150     

# Create accounts
accounts = [fake.uuid4() for _ in range(num_accounts)]

transactions = []

# Normal transactions
for _ in range(num_transactions):
    src = random.choice(accounts)
    dst = random.choice(accounts)
    if src != dst:
        amount = random.uniform(10, 500)
        transactions.append([src, dst, amount, 0])  # 0 = non-fraud

# Create fraud rings
fraud_accounts = set()

for ring_id in range(num_fraud_rings):
    ring = random.sample(accounts, fraud_ring_size)
    fraud_accounts.update(ring)

    for _ in range(fraud_tx_per_ring):
        src = random.choice(ring)
        dst = random.choice(ring)
        if src != dst:
            amount = random.uniform(700, 2000) 
            transactions.append([src, dst, amount, 1])  # 1 = fraud

# Shuffle transactions
random.shuffle(transactions)

df = pd.DataFrame(
    transactions,
    columns=["from_account", "to_account", "amount", "is_fraud"]
)

df.to_csv("data/raw/transactions.csv", index=False)