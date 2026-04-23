import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Utils
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- LOAD DATA ----------
df = pd.read_csv("dataset_phishing.csv")

# ---------- LABEL MAPPING ----------
# 1 = legitimate, 0 = phishing
df['status'] = df['status'].map({
    'legitimate': 1,
    'phishing': 0
})
def extract_features(url):
    parsed = urlparse(url)
    host = parsed.netloc if parsed.netloc else ""

    return [
        len(url),                                # length_url
        len(host),                               # length_hostname
        1 if re.match(r'\d+\.\d+\.\d+\.\d+', host) else 0,  # ip
        url.count('.'),                          # nb_dots
        url.count('-'),                          # nb_hyphens
        url.count('@'),                          # nb_at
        url.count('?'),                          # nb_qm
        url.count('&'),                          # nb_and
        url.count('='),                          # nb_eq
        url.count('_'),                          # nb_underscore
        url.count('%'),                          # nb_percent
        url.count('/'),                          # nb_slash
        url.count('www'),                        # nb_www
        url.count('.com'),                       # nb_com
        1 if 'https' in host else 0,             # https_token
        sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,  # ratio_digits_url
        max(host.count('.') - 1, 0)              # nb_subdomains
    ]
# ---------- BUILD X AND y ----------
X =  np.array([extract_features(u) for u in df["url"]]) # features (86 columns)
y = df["status"]                        # target

# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- DEFINE MODELS ----------
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.1, max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ('model', DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ]),

    "XGBoost": Pipeline([
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
}

# ---------- TRAIN & COMPARE ----------
results = {}

for name, model in models.items():
    print(f"\n===== {name} =====")
    
    # fit (training)
    model.fit(X_train, y_train)
    
    # predict
    pred = model.predict(X_test)
    
    # evaluate
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test, pred))
    
    results[name] =acc
    

# ---------- FINAL COMPARISON ----------
print("\n===== FINAL COMPARISON =====")
for model, score in results.items():
    print(f"{model}: {score}")