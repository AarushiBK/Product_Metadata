# evaluate_buckets.py

import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv('./data/flipkart_with_actual_buckets.csv')
df_eval = df[df['actual_bucket'].notnull()]

# Ensure all buckets match the same label set
valid_labels = {"Clothing", "Jewelry", "Home & Decor", "Budget", "Tech", "Uncategorized"}
df_eval = df_eval[df_eval['final_bucket'].isin(valid_labels)]
df_eval = df_eval[df_eval['predicted_bucket'].isin(valid_labels)]

# 1. Evaluate rule-based bucketer
print("📊 Rule-based vs Actual:")
print(classification_report(df_eval['actual_bucket'], df_eval['final_bucket'], zero_division=0))

# 2. Evaluate model predictions
print("\n🤖 Model vs Actual:")
print(classification_report(df_eval['actual_bucket'], df_eval['predicted_bucket'], zero_division=0))
