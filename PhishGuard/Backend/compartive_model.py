import pandas as pd
import numpy as np

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

# ---------- LEXICAL-ONLY FEATURES ----------
# Only URL-structure features reliably computable at runtime.
# External/page-content features (google_index, page_rank, web_traffic, etc.)
# were excluded due to train/inference mismatch — they had real scraped values
# in the dataset but default to 0 at prediction time.
LEXICAL_COLS = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at',
    'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent',
    'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar',
    'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path', 'https_token',
    'ratio_digits_url', 'ratio_digits_host', 'punycode', 'port', 'tld_in_path',
    'tld_in_subdomain', 'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix',
    'random_domain', 'shortening_service', 'path_extension', 'nb_redirection',
    'nb_external_redirection', 'length_words_raw', 'char_repeat', 'shortest_words_raw',
    'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 'longest_word_host',
    'longest_word_path', 'avg_words_raw', 'avg_word_host', 'avg_word_path',
    'phish_hints', 'domain_in_brand', 'brand_in_subdomain', 'brand_in_path',
    'suspecious_tld', 'statistical_report'
]

# ---------- BUILD X AND y ----------
X = df[LEXICAL_COLS]
y = df["status"]

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
        ('model', XGBClassifier(eval_metric='logloss'))
    ])
}

# ---------- TRAIN & COMPARE ----------
results = {}

for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    results[name] = acc

# ---------- FINAL COMPARISON ----------
print("\n===== FINAL COMPARISON =====")
for model, score in results.items():
    print(f"{model}: {score:.4f}")