import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------- LOAD DATA ----------
df = pd.read_csv("final_dataset.csv")

# ---------- FEATURE EXTRACTOR ----------
def extract_features(url):
    parsed = urlparse(url)

    features = [
        len(url),                                # length
        url.count('.'),                         # dots
        url.count('-'),                         # hyphens
        1 if parsed.scheme=="https" else 0,     # https
        1 if any(c.isdigit() for c in url) else 0,
        1 if any(w in url.lower() for w in
                 ['login','signin','verify','update','secure','account']) else 0,
        len(parsed.netloc),                     # domain length
        len(parsed.path)                        # path length
    ]
    return features

# ---------- BUILD FEATURES ----------
X = np.array([extract_features(u) for u in df["url"]])
y = df["label"]

# ---------- TRAIN ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ---------- EVALUATE ----------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# ---------- SAVE MODEL ----------
joblib.dump(model, "final_phishing_model.pkl")

print("Model saved!")