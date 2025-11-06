import os
import joblib
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Ensure stopwords are available
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

# Create a models directory
if not os.path.exists("models"):
    os.makedirs("models")

MODEL_FILE = "models/nb_model.joblib"
VECTORIZER_FILE = "models/vectorizer.joblib"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def train_and_save_model():
    """Train a simple model from SPAM.csv"""
    df = pd.read_csv("SPAM.csv")
    df.columns = [col.lower() for col in df.columns]

    # Find the text and label columns automatically
    text_col = [c for c in df.columns if "text" in c or "message" in c or "sms" in c][0]
    label_col = [c for c in df.columns if "label" in c or "spam" in c or "class" in c][0]

    df[text_col] = df[text_col].astype(str).apply(clean_text)
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_FILE)
    joblib.dump(pipeline.named_steps["nb"], MODEL_FILE)
    print("✅ Model trained successfully and saved.")
    return pipeline


def load_model_and_vectorizer():
    """Load the saved model; if not found, train it."""
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        vec = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
    else:
        pipe = train_and_save_model()
        vec = pipe.named_steps["tfidf"]
        model = pipe.named_steps["nb"]
    return vec, model


def predict_sms(text):
    """Predict whether message is SPAM or NOT SPAM"""
    vec, model = load_model_and_vectorizer()
    cleaned = clean_text(text)
    X = vec.transform([cleaned])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][list(model.classes_).index(pred)]
    return {"label": str(pred).upper(), "probability": float(prob)}
