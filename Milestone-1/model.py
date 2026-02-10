# ==============================================================================
# 🏆 FINAL PHISHING DETECTION MODEL (Production Ready)
# ==============================================================================
import pandas as pd
import numpy as np
import tldextract
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ==========================================
# 1. THE "PROVEN" FEATURE EXTRACTOR
# ==========================================
def extract_features(url):
    """
    Extracts ONLY the features that we proved work:
    1. Length
    2. Dot Count
    3. HTTPS Check (The most important!)
    4. Has Digits
    5. Suspicious Words (Bonus protection)
    """
    features = []
    try:
        parsed = urlparse(url)
        
        # Feature 1: Total Length
        features.append(len(url))
        
        # Feature 2: Dot Count (High in phishing subdomains)
        features.append(url.count('.'))
        
        # Feature 3: Hyphen Count
        features.append(url.count('-'))
        
        # Feature 4: HTTPS Check (1 = Secure, 0 = Not Secure)
        # This is the "Golden Feature" for your dataset
        features.append(1 if parsed.scheme == 'https' else 0)
        
        # Feature 5: Has Numbers? (Google has none, Phishing has many)
        features.append(1 if any(c.isdigit() for c in url) else 0)
        
        # Feature 6: Suspicious Keywords (Bonus)
        sus_words = ['login', 'signin', 'bank', 'verify', 'update']
        features.append(1 if any(w in url.lower() for w in sus_words) else 0)

    except:
        return [0, 0, 0, 0, 0, 0] # Return zeros if error
    return features

# ==========================================
# 2. LOAD DATA
# ==========================================
print("📂 Loading Data...")
# Use 'r' to fix path issues
file_path = r"C:\Infosys\Dev-of-a-Machine-Learning-Based-Phishing-Website-Detection-and-Classif-System_Feb_Batch-8_2026\phishing_dataset_full.csv"

try:
    df = pd.read_csv(file_path)
    
    # Rename 'text' to 'url' automatically
    if 'text' in df.columns:
        df.rename(columns={'text': 'url'}, inplace=True)
        
    print(f"✅ File Found! Loaded {len(df)} rows.")

except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Ensure Labels are Integers (0/1)
if df['label'].dtype == 'object':
    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['phishing', 'yes', '1'] else 0)

# ==========================================
# 3. TRAIN
# ==========================================
print("🚀 Extracting Features... (Fast)")
X = [extract_features(u) for u in df['url']]
y = df['label']

print("🧠 Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"🎉 FINAL ACCURACY: {acc * 100:.2f}%")

# Save
joblib.dump(model, 'final_phishing_model.pkl')
print("💾 Model Saved: 'final_phishing_model.pkl'")

# ==========================================
# 4. FINAL VERIFICATION
# ==========================================
print("\n🔎 REAL-TIME TEST:")
test_cases = [
    "https://www.google.com",            # Should be SAFE
    "http://paypal-login-verify.xyz",    # Should be PHISHING
    "https://www.amazon.com",            # Should be SAFE
    "http://192.168.1.1/bank"            # Should be PHISHING
]

for url in test_cases:
    feats = extract_features(url)
    pred = model.predict([feats])[0]
    status = "🔴 PHISHING" if pred == 1 else "🟢 SAFE"
    print(f"{url} -> {status}")