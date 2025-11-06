import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already available
try:
    STOP = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP = set(stopwords.words('english'))

ps = PorterStemmer()

MODEL_DIR = "models"
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.joblib")

# Create folder if missing
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def clean_text(s: str) -> str:
    """Clean text for ML processing"""
    s = s.lower()
    s = re.sub(r"http\\S+", "", s)
    s = re.sub(r"[^a-z0-9\\s]", " ", s)
    tokens = [ps.stem(word) for word in s.split() if word not in STOP]
    return " ".join(tokens)


def train_default_model(dataset_path: str = None):
    """Train a default TF-IDF + Naive Bayes model if no saved model exists."""
    # Try to locate dataset file
    if dataset_path is None:
        possible = ["SPAM.csv", "spam.csv", "SMSSpamCollection", "dataset.csv"]
        for f in possible:
            if os.path.exists(f):
                dataset_path = f
                break

    if dataset_path is None:
        # Download sample dataset if none exists
        import requests
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        data = requests.get(url, timeout=10).content
        open("sms.tsv", "wb").write(data)
        dataset_path = "sms.tsv"

    # Load dataset
    if dataset_path.endswith(".tsv"):
        df = pd.read_csv(dataset_path, sep="\t", names=["label", "text"])
    else:
        df = pd.read_csv(dataset_path, encoding="latin-1")

    df["text"] = df["text"].astype(str).apply(clean_text)
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)

    # Save trained components
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_FILE)
    joblib.dump(pipeline.named_steps["nb"], MODEL_FILE)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained with accuracy: {acc:.4f}")
    return acc


def load_model_and_vectorizer():
    """Load or train model + vectorizer"""
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        vec = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        return vec, model
    else:
        train_default_model()
        vec = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        return vec, model


def predict_sms(text: str):
    """Predict spam/ham from input text"""
    vec, model = load_model_and_vectorizer()
    cleaned = clean_text(text)
    X = vec.transform([cleaned])

    try:
        proba = model.predict_proba(X)[0]
        spam_index = list(model.classes_).index("spam")
        spam_prob = float(proba[spam_index])
    except Exception:
        pred = model.predict(X)[0]
        spam_prob = 1.0 if str(pred).lower() == "spam" else 0.0

    label = "SPAM" if spam_prob >= 0.5 else "NOT SPAM"
    return {"label": label, "probability": spam_prob}
