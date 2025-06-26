# validate_predictions.py

import pandas as pd

# === Step 1: Load predictions ===
df = pd.read_csv('./data/flipkart_with_predictions.csv')

# === Step 2: Merge in final_bucket ===
df_labels = pd.read_csv('./data/flipkart_buckets_single_label.csv')
df['final_bucket'] = df_labels['final_bucket']  # Merge by index/position

# === Step 3: Define manual labels ===
manual_labels = {
    0: "Clothing", 3: "Clothing", 6: "Clothing", 9: "Clothing", 13: "Clothing", 15: "Clothing", 21: "Clothing",
    2: "Clothing", 8: "Clothing", 10: "Clothing", 17: "Clothing", 23: "Clothing", 40: "Clothing",
    1: "Home & Decor", 7: "Home & Decor", 16: "Home & Decor", 19: "Home & Decor",
    4: "Home & Decor", 12: "Home & Decor", 20: "Home & Decor",
    25: "Clothing", 31: "Clothing", 35: "Clothing", 37: "Clothing",
    26: "Uncategorized",  # Ambiguous → set to 'Uncategorized'
    30: "Home & Decor",
    32: "Home & Decor", 33: "Home & Decor", 34: "Home & Decor",
    22: "Clothing", 39: "Clothing",

    # Footwear and Watches → Clothing
    **{i: "Clothing" for i in range(200, 231)},

    # Jewelry vs Clothing in 408–449
    408: "Clothing", 409: "Clothing",
    410: "Jewelry", 411: "Jewelry", 412: "Jewelry", 415: "Jewelry", 417: "Jewelry",
    423: "Jewelry", 424: "Jewelry", 425: "Jewelry", 427: "Jewelry", 428: "Jewelry",
    429: "Jewelry", 430: "Jewelry", 435: "Jewelry", 439: "Jewelry", 440: "Jewelry",
    413: "Clothing", 414: "Clothing", 416: "Clothing", 418: "Clothing", 419: "Clothing",
    420: "Clothing", 421: "Clothing", 422: "Clothing", 426: "Clothing", 431: "Clothing",
    432: "Clothing", 433: "Clothing", 434: "Clothing", 436: "Clothing", 437: "Home & Decor",
    438: "Clothing", 441: "Clothing", 442: "Clothing", 443: "Clothing", 444: "Clothing",
    445: "Clothing", 446: "Clothing", 447: "Clothing", 448: "Clothing", 449: "Clothing"
}

# === Step 4: Add actual_bucket column ===
df['actual_bucket'] = None

for idx, label in manual_labels.items():
    if idx in df.index:
        df.at[idx, 'actual_bucket'] = label

# === Step 5: Save merged version ===
output_path = './data/flipkart_with_actual_buckets.csv'
df.to_csv(output_path, index=False)
print(f"✅ Saved {len(manual_labels)} manual labels to {output_path}")
