import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import re

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '<URL>', text)    # replace URLs
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)  # replace emails
    text = re.sub(r'[^a-z\s<>]', '', text)      # remove punctuation except <URL>/<EMAIL>
    text = re.sub(r'\s+', ' ', text)            # remove extra spaces
    return text.strip()

# --- Load dataset ---
df = pd.read_csv(r"D:\BVB\PYTHON\archive\combined_dataset.csv")
df['text'] = df['text'].apply(clean_text)

# Convert labels to 0/1
X = df['text']
y = df['target'].map({'ham': 0, 'spam': 1})

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TF-IDF + Logistic Regression pipeline ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True,
                              stop_words='english',
                              ngram_range=(1,3),   # word and character n-grams
                              analyzer='char_wb')),
    ('lr', LogisticRegression(max_iter=1000))
])

# --- Train ---
pipeline.fit(X_train, y_train)

# --- Save pipeline ---
joblib.dump(pipeline, r"D:\BVB\PYTHON\spam_pipeline.pkl")

# --- Evaluate ---
print("Training accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))
