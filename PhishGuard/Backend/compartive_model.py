import pandas as pd

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

# ---------- BUILD X AND y ----------
X = df.drop(columns=["url", "status"])  # features (86 columns)
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