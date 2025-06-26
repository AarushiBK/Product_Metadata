# run_prediction.py
import pandas as pd
import joblib

model = joblib.load('./models/bucket_classifier.pkl')
vectorizer = joblib.load('./models/vectorizer.pkl')

df = pd.read_csv('./data/flipkart_cleaned.csv')
df['text'] = (df['product_name'].fillna('') + ' ' +
              df['description'].fillna('') + ' ' +
              df['product_category_tree'].fillna(''))

X = vectorizer.transform(df['text'])
df['predicted_bucket'] = model.predict(X)
output_path = './data/flipkart_with_predictions.csv'
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to: {output_path}")
