import numpy as np
import pandas as pd
import random

# FAKE TRANSACTION GENERATOR
def generate_fake_transaction(fraud_ratio=0.4):

    if random.random() > fraud_ratio:
        time = random.uniform(0, 172800)
        amount = random.uniform(1, 500)

        # normal behavior (centered near 0)
        features = np.random.normal(0, 1, 28)
        label = 0

    # fraud transactions
    else:
        time = random.uniform(0, 172800)
        amount = random.uniform(500, 5000)
        features = np.random.normal(-5.0, 2.5, 28) 
        label = 1

    row = [time] + list(features) + [amount, label]
    return row

# GENERATE DATASET
def generate_fake_dataset(n_rows=100, fraud_ratio=0.4):

    data = []

    for _ in range(n_rows):
        data.append(generate_fake_transaction(fraud_ratio))

    columns = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount","Class"]

    return pd.DataFrame(data, columns=columns)


# ==========================
# CREATE CSV
# ==========================
fake_df = generate_fake_dataset(
    n_rows=100,
    fraud_ratio=0.4   # 🔥 change this
)

fake_df.to_csv("synthetic_fraud_data.csv", index=False)

print("✅ CSV created successfully!")
print(fake_df["Class"].value_counts())