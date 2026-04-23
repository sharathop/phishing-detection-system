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

# ---------- FEATURES ----------
X = df.drop(columns=["url", "status"])
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

    #  Random Forest
    {
        'model': [RandomForestClassifier(random_state=42)],
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    },

    #  XGBoost
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