import os
import joblib
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Ensure NLTK stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "nb_model.joblib")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def train_and_save_model():
    """Train a model from SPAM.csv in a flexible way."""
    try:
        # Try multiple encodings
        encodings_to_try = ["utf-8", "latin-1", "ISO-8859-1"]
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv("SPAM.csv", encoding=enc, errors="ignore")
                break
            except Exception as e:
                print(f"⚠️ Could not read with encoding {enc}: {e}")

        if df is None:
            raise Exception("SPAM.csv could not be loaded with any encoding")

        # Drop empty columns and rows
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")

        # Try to find label and text columns automatically
        df.columns = [c.lower().strip() for c in df.columns]
        text_col = None
        label_col = None

        for c in df.columns:
            if any(k in c.lower() for k in ["message", "text", "sms", "email"]):
                text_col = c
            if any(k in c.lower() for k in ["label", "spam", "target", "class"]):
                label_col = c

        if text_col is None or label_col is None:
            raise Exception(f"Could not identify text/label columns. Found: {df.columns}")

        print(f"✅ Using columns -> text: {text_col}, label: {label_col}")

        # Clean text and train
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

    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise


def load_model_and_vectorizer():
    """Load saved model; train if missing."""
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            vec = joblib.load(VECTORIZER_FILE)
            model = joblib.load(MODEL_FILE)
        else:
            pipe = train_and_save_model()
            vec = pipe.named_steps["tfidf"]
            model = pipe.named_steps["nb"]
        return vec, model
    except Exception as e:
        print(f"❌ Failed to load model/vectorizer: {e}")
        raise


def predict_sms(text):
    """Predict SPAM or NOT SPAM"""
    vec, model = load_model_and_vectorizer()
    cleaned = clean_text(text)
    X = vec.transform([cleaned])
    pred = model.predict(X)[0]

    try:
        prob = model.predict_proba(X)[0][list(model.classes_).index(pred)]
    except Exception:
        prob = 0.0

    return {"label": str(pred).upper(), "probability": float(prob)}

