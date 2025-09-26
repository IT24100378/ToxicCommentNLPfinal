# 01_balance_dataset.py
# -----------------------
# This script loads the Kaggle toxic comments dataset,
# balances it with 8000 toxic and 8000 clean comments,
# saves the balanced dataset as balanced.csv,
# and plots a bar chart to show class distribution.

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the raw Kaggle dataset
# Make sure the file name matches your dataset (e.g., train.csv from Kaggle)
data = pd.read_csv("data/train.csv")

# 2. We only need the 'comment_text' and 'toxic' column
# Kaggle dataset has multiple toxic categories (toxic, severe_toxic, obscene, etc.)
# We'll use "toxic" (0 = clean, 1 = toxic)
data = data[["comment_text", "toxic"]]

# 3. Select 8000 toxic and 8000 clean comments
toxic_comments = data[data["toxic"] == 1].sample(n=8000, random_state=42)
clean_comments = data[data["toxic"] == 0].sample(n=8000, random_state=42)

# 4. Combine them into one balanced dataset
balanced_data = pd.concat([toxic_comments, clean_comments])

# 5. Shuffle the dataset (important to mix toxic and clean comments)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Save to CSV
balanced_data.to_csv("data/balanced.csv", index=False)

print(" Balanced dataset saved as balanced.csv with shape:", balanced_data.shape)

# 7. Plot class distribution
counts = balanced_data["toxic"].value_counts()

plt.bar(["Clean (0)", "Toxic (1)"], counts)
plt.title("Balanced Dataset Distribution")
plt.xlabel("Comment Type")
plt.ylabel("Count")
plt.savefig("visuals/balance_chart.png")
plt.show()