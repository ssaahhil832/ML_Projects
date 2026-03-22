import re
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = Path("data/spam.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# 1) Load data with flexible column detection
df = pd.read_csv(DATA_PATH, encoding="latin1")
df.columns = [c.strip().lower() for c in df.columns]

# Try common label/text names
label_col_candidates = ["label", "v1", "category", "class"]
text_col_candidates  = ["text", "v2", "message", "sms"]

def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not find any of {candidates} in {cols}")

label_col = pick(df.columns, label_col_candidates)
text_col  = pick(df.columns, text_col_candidates)

# Normalize labels to 0/1
df = df[[label_col, text_col]].dropna()
df[label_col] = df[label_col].astype(str).str.lower().str.strip()
df[label_col] = df[label_col].map({"ham":0, "spam":1}).fillna(df[label_col])

# If labels aren’t "ham"/"spam", try to coerce (optional)
if df[label_col].dtype == object:
    # fallback: treat anything that contains 'spam' as spam
    df[label_col] = df[label_col].apply(lambda x: 1 if "spam" in str(x) else 0)

X = df[text_col].astype(str)
y = df[label_col].astype(int)

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Build pipeline (vectorizer + NB)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=2)),
    ("nb", MultinomialNB(alpha=0.5))
])

# 4) Train
pipe.fit(X_train, y_train)

# 5) Evaluate
pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["ham","spam"]))

# 6) Save
joblib.dump(pipe, ART_DIR / "spam_nb_tfidf.joblib")
print("Saved to artifacts/spam_nb_tfidf.joblib")
