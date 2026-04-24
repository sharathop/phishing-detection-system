import pandas as pd

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---------- LOAD DATA ----------
df = pd.read_csv("dataset_phishing.csv")

# ---------- LABEL MAPPING ----------
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

# ---------- FEATURES ----------
X = df[LEXICAL_COLS]
y = df["status"]

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- PIPELINE ----------
pipe = Pipeline([
    ('model', RandomForestClassifier())  # placeholder
])

# ---------- PARAM GRID ----------
param_grid = [

    # Random Forest
    {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    },

    # XGBoost
    {
        'model': [XGBClassifier(eval_metric='logloss')],
        'model__n_estimators': [100, 200],
        'model__max_depth': [4, 6, 8],
        'model__learning_rate': [0.01, 0.1],
        'model__subsample': [0.8, 1.0]
    }
]

# ---------- GRID SEARCH ----------
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='f1_macro',   # balanced metric
    n_jobs=-1,
    verbose=2
)

# ---------- TRAIN ----------
grid.fit(X_train, y_train)

# ---------- BEST MODEL ----------
print("\n===== BEST RESULT =====")
print("Best Model & Params:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_model = grid.best_estimator_

# ---------- TEST EVALUATION ----------
pred = best_model.predict(X_test)

print("\n===== TEST PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))