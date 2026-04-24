import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# ---------- LOAD DATA ----------
df = pd.read_csv("dataset_phishing.csv")

# ---------- LABEL MAPPING ----------
df['status'] = df['status'].map({
    'legitimate': 1,
    'phishing': 0
})

# ---------- LEXICAL-ONLY FEATURES ----------
# Only URL-structure features that can be reliably computed at runtime
# without any network/page-fetch calls.
# External/page-content features (nb_hyperlinks, google_index, page_rank, etc.)
# were excluded because the training dataset had real scraped values while the
# extractor defaults them all to 0 at prediction time — causing severe mismatch.
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

X = df[LEXICAL_COLS]
y = df['status']

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- MODEL ----------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric='logloss'
)

# ---------- TRAIN ----------
model.fit(X_train, y_train)

# ---------- PREDICT ----------
pred = model.predict(X_test)

# ---------- EVALUATION ----------
print("\n===== TEST PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# ---------- ROC & AUC ----------
probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

print("\nAUC Score:", auc)

# ---------- ROC PLOT ----------
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------- SAVE MODEL + FEATURES ----------
model_data = {
    "model": model,
    "features": LEXICAL_COLS
}

joblib.dump(model_data, "xgboost_phishing_model.pkl")

print(f"\nModel + {len(LEXICAL_COLS)} lexical feature names saved to xgboost_phishing_model.pkl")
print("Deployment only needs the .pkl file — no CSV required.")