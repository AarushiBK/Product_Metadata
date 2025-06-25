import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === Load strictly labeled data ===
df = pd.read_csv('./data/flipkart_labeled_seed.csv')

# === Combine text fields ===
df['text'] = (df['product_name'].fillna('') + ' ' +
              df['description'].fillna('') + ' ' +
              df['product_category_tree'].fillna(''))

# === Extract features ===
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['text'])

# === Labels ===
y = df['confident_bucket']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === Save model & vectorizer (optional, if using later) ===
import joblib
joblib.dump(clf, './models/bucket_classifier.pkl')
joblib.dump(vectorizer, './models/vectorizer.pkl')
