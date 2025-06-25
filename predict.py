# run_prediction.py
import pandas as pd
import joblib

# === Load model and vectorizer ===
print("📦 Loading trained model and vectorizer...")
model = joblib.load('./models/bucket_classifier.pkl')
vectorizer = joblib.load('./models/vectorizer.pkl')

# === Load new product data (can be cleaned or raw if you're testing) ===
print("📄 Loading new product data...")
df = pd.read_csv('./data/flipkart_cleaned.csv')  # Change if you want a different file

# === Prepare text features ===
print("📝 Combining text fields...")
df['text'] = (df['product_name'].fillna('') + ' ' +
              df['description'].fillna('') + ' ' +
              df['product_category_tree'].fillna(''))

# === Vectorize using the saved TF-IDF model ===
print("🔍 Vectorizing input...")
X = vectorizer.transform(df['text'])

# === Make predictions ===
print("🤖 Predicting product buckets...")
df['predicted_bucket'] = model.predict(X)

# === Save results ===
output_path = './data/flipkart_with_predictions.csv'
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to: {output_path}")

# === Show sample output ===
print("\n🔎 Sample predictions:")
print(df[['product_name', 'predicted_bucket']].head(10))
