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

# ---------- FEATURES ----------
X = df.drop(columns=["url", "status"])
y = df["status"]

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- MODEL (use your tuned params) ----------
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

# ---------- SAVE MODEL ----------
joblib.dump(model, "xgboost_phishing_model.pkl")
print("\nModel saved successfully!")