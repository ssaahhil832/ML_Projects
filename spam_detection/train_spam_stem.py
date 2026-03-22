import re
import joblib
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure NLTK resources
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_and_stem(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    tokens = [ps.stem(t) for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

DATA_PATH = Path("data/spam.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8", errors="ignore")
df.columns = [c.strip().lower() for c in df.columns]
label_col = next(c for c in df.columns if c in ["label","v1","category","class"])
text_col  = next(c for c in df.columns if c in ["text","v2","message","sms"])
df = df[[label_col, text_col]].dropna()
df[label_col] = df[label_col].astype(str).str.lower().str.strip().map({"ham":0,"spam":1}).fillna(0).astype(int)

X = df[text_col].astype(str).apply(clean_and_stem)
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)),
    ("nb", MultinomialNB(alpha=0.5))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["ham","spam"]))

joblib.dump(pipe, ART_DIR / "spam_nb_tfidf_stem.joblib")
print("Saved to artifacts/spam_nb_tfidf_stem.joblib")
