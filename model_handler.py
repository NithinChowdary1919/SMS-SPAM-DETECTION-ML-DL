# model_handler.py
except Exception:
df = pd.read_csv(dataset_path, sep='\t', header=None, names=['label','text'], encoding='latin-1')


df = df[['label','text']]
df['text_clean'] = df['text'].astype(str).apply(clean_text)


X = df['text_clean']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


pipeline = Pipeline([
("tfidf", TfidfVectorizer()),
("nb", MultinomialNB())
])
pipeline.fit(X_train, y_train)


# save components
joblib.dump(pipeline.named_steps['tfidf'], VECTORIZER_FILE)
joblib.dump(pipeline.named_steps['nb'], MODEL_FILE)


preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
return acc




# Load model and vectorizer
def load_model_and_vectorizer():
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
vec = joblib.load(VECTORIZER_FILE)
m = joblib.load(MODEL_FILE)
return vec, m
else:
acc = train_default_model()
vec = joblib.load(VECTORIZER_FILE)
m = joblib.load(MODEL_FILE)
return vec, m




# Prediction function used by API and Streamlit
def predict_sms(text: str):
vec, m = load_model_and_vectorizer()
cleaned = clean_text(text)
X = vec.transform([cleaned])
prob_spam = None
try:
prob = m.predict_proba(X)[0]
# assume classes contain 'spam' or 'ham'
if m.classes_[1].lower() == 'spam':
prob_spam = float(prob[1])
else:
# find index of 'spam'
idx = 0
for i,c in enumerate(m.classes_):
if str(c).lower() == 'spam':
idx = i
prob_spam = float(prob[idx])
except Exception:
# fallback to predict()
pred = m.predict(X)[0]
prob_spam = 1.0 if str(pred).lower()== 'spam' else 0.0


label = 'SPAM' if prob_spam >= 0.5 else 'NOT SPAM'
return {"label": label, "probability": prob_spam}