# ==========================================
# 🔮 PREDICT ANY URL (Interactive Mode)
# ==========================================
import joblib
from urllib.parse import urlparse

# 1. LOAD THE SAVED MODEL
print("⏳ Loading Model...")
model = joblib.load('phishing_model.pkl')
print("✅ Model Loaded! Type a URL to test (or 'q' to quit)")

# 2. DEFINE THE SAME FEATURES (Must match training!)
def extract_features(url):
    try:
        parsed = urlparse(url)
        return [
            len(url),
            url.count('.'),
            url.count('-'),
            1 if parsed.scheme == 'https' else 0, # The "Golden Feature"
            1 if any(c.isdigit() for c in url) else 0,
            1 if any(w in url.lower() for w in ['login', 'signin', 'bank', 'verify', 'update']) else 0
        ]
    except:
        return [0, 0, 0, 0, 0, 0]

# 3. INTERACTIVE LOOP
while True:
    print("-" * 30)
    user_url = input("🔗 Enter URL: ")
    
    if user_url.lower() == 'q':
        break
        
    # Predict
    features = extract_features(user_url)
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1] # Confidence score
    
    result = "PHISHING" if prediction == 1 else "SAFE"
    print(f"   Result: {result}")
    print(f"   Confidence: {probability * 100:.1f}%")