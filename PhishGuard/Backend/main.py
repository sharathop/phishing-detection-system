"""
PhishGuard — FastAPI Backend
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import joblib, numpy as np, os
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

load_dotenv()

import tldextract
from urllib.parse import urlparse

from db_setup import get_db, engine
from db_models import Base, User, ScanHistory
from feature_extraction import extract_features

# ── Known-safe domain allowlist ──────────────────────────────
# Pure lexical URL features can't reliably distinguish a clean, short,
# well-known root domain (github.com) from a clean, short phishing domain.
# Real detectors combine ML with an allowlist for exactly this reason.
# Bypasses the model entirely for domains on this list — no network call.
KNOWN_SAFE_DOMAINS = {
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
    'twitter.com', 'x.com', 'instagram.com', 'github.com', 'microsoft.com',
    'apple.com', 'chatgpt.com', 'openai.com', 'netflix.com', 'linkedin.com',
    'reddit.com', 'yahoo.com', 'bing.com', 'office.com', 'dropbox.com',
    'adobe.com', 'ebay.com', 'paypal.com', 'whatsapp.com', 'stackoverflow.com',
    'wordpress.com', 'blogspot.com', 'tumblr.com', 'medium.com', 'quora.com',
    'pinterest.com', 'twitch.tv', 'discord.com', 'slack.com', 'zoom.us',
    'salesforce.com', 'shopify.com', 'stripe.com', 'notion.so', 'figma.com',
    'gitlab.com', 'bitbucket.org', 'atlassian.com', 'jira.com', 'trello.com',
    'spotify.com', 'soundcloud.com', 'vimeo.com', 'nytimes.com', 'bbc.com',
    'cnn.com', 'forbes.com', 'bloomberg.com', 'reuters.com', 'wsj.com',
    'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com', 'ibm.com',
    'oracle.com', 'sap.com', 'intel.com', 'nvidia.com', 'amd.com',
    'samsung.com', 'sony.com', 'lg.com', 'huawei.com', 'xiaomi.com',
    'uber.com', 'lyft.com', 'airbnb.com', 'booking.com', 'expedia.com',
    'tripadvisor.com', 'yelp.com', 'walmart.com', 'target.com', 'bestbuy.com',
    'homedepot.com', 'costco.com', 'ikea.com', 'nike.com', 'adidas.com',
    'coursera.org', 'udemy.com', 'edx.org', 'khanacademy.org', 'duolingo.com',
    'mit.edu', 'stanford.edu', 'harvard.edu', 'w3.org', 'mozilla.org',
    'python.org', 'npmjs.com', 'docker.com', 'kubernetes.io', 'terraform.io',
    'anthropic.com', 'claude.ai', 'meta.com', 'tiktok.com', 'snapchat.com',
    'telegram.org', 'signal.org', 'protonmail.com', 'icloud.com', 'gmail.com',
    'outlook.com', 'live.com', 'hotmail.com', 'yandex.com', 'baidu.com',
    'alibaba.com', 'tencent.com', 'jd.com', 'ea.com', 'steampowered.com',
    'epicgames.com', 'roblox.com', 'minecraft.net', 'blizzard.com',
}


def get_registered_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else ext.domain.lower()


def uses_https(url: str) -> bool:
    """
    Human-readable HTTPS check for the UI.
    NOTE: This is deliberately separate from the model's `https_token`
    feature. That feature's raw value is inverted to match how the
    training dataset defines it (1 = plain http, 0 = https) — correct
    for the model, but wrong to show a user as "does this URL use HTTPS".
    Always use this function for display, never the raw feature value.
    """
    return urlparse(url).scheme == "https"

app = FastAPI(title="PhishGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    for path in ["index.html", "../index.html", "../Frontend/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="index.html not found.")

# ── ML Model ──────────────────────────────────────────────────
MODEL = None
FEATURE_NAMES = []
THRESHOLD = 0.4  

for model_path in ["xgboost_phishing_model.pkl"]:
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            MODEL = model_data["model"]
            FEATURE_NAMES = model_data["features"]
            THRESHOLD = model_data.get("threshold", 0.4)
        else:
            MODEL = model_data  # fallback for old pkl format
        print(f" ML Model loaded: {model_path}")
        print(f" Feature names loaded: {len(FEATURE_NAMES)} features")
        print(f" Threshold: {THRESHOLD}")
        break
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client  = None
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(" Gemini AI connected")
except Exception as e:
    print(f"  Gemini not connected: {e}")

GEMINI_SYSTEM = """
You are PhishGuard AI, a cybersecurity assistant integrated into a phishing detection system.

PERSONALIZATION:
- The user's name will always be provided.
- Always greet the user using their name in the first response.
- Keep the tone professional, helpful, and concise.

SCOPE (STRICT):
You ONLY answer questions related to:
- phishing detection
- URL safety
- scan results
- cybersecurity threats
- phishing project/model-related explanations

If the user asks anything unrelated (coding outside this project, sports, movies, general knowledge, entertainment, etc.), respond with:
"[User Name], I can only assist with phishing detection, this project, scan results, and cybersecurity-related queries. Please ask about URL safety, phishing detection, or cyber threats."

URL ANALYSIS:
If the user provides a URL:
- Explain the analysis clearly using key phishing indicators such as URL length, dots, HTTPS usage, suspicious keywords, subdomains, special characters, redirects, IP usage, path structure, and query behavior.
- Keep the explanation simple and practical.
- End with a clear verdict: Safe or Phishing, with reasoning.

MODEL EXPLANATION:
When asked about the project or model:
- Multiple models were tested: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- XGBoost was selected because it achieved the highest accuracy.
- The model uses 56 lexical URL features extracted purely from the URL string itself.
- Feature selection was a critical step: the original dataset had 87 features, but 31 external/page-content features (like google_index, page_rank, web_traffic) were removed because they had real scraped values during training but defaulted to 0 at runtime — causing wrong predictions. Only the 56 features reliably computable at inference time were kept.
- The final model achieves 92.3% accuracy and 0.974 AUC on the test set.

STYLE RULES:
- Be direct and structured.
- Avoid long paragraphs.
- Avoid unnecessary technical jargon unless asked.
- Do NOT explain unrelated topics.
- Always prioritize clarity over complexity.

GOAL:
Provide accurate, fast, and understandable phishing detection insights to the user.
"""

# ── Schemas ───────────────────────────────────────────────────
class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ScanRequest(BaseModel):
    url: str
    email: str

class ChatRequest(BaseModel):
    message: str
    email: str

# ── Routes ────────────────────────────────────────────────────

@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    db.add(User(name=user.name, email=user.email, password=user.password))
    db.commit()
    return {"status": "success", "message": "User created successfully"}


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        User.email == user.email,
        User.password == user.password
    ).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {
        "status": "success",
        "user": {"name": db_user.name or db_user.email, "email": db_user.email}
    }


@app.post("/scan")
async def scan_url(request: ScanRequest, db: Session = Depends(get_db)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="ML model not loaded.")

    # 0. Allowlist check — bypasses the model for well-known domains,
    # since pure lexical features can't reliably clear a clean, short,
    # famous root domain (see /docs or README for why).
    if get_registered_domain(request.url) in KNOWN_SAFE_DOMAINS:
        features = extract_features(request.url)
        return {
            "result": "Safe",
            "confidence": 99.0,
            "source": "allowlist",
            "uses_https": uses_https(request.url),
            "features_array": features,
            "features": dict(zip(FEATURE_NAMES, features)) if FEATURE_NAMES else {}
        }

    # 1. Check DB cache
    existing = db.query(ScanHistory).filter(ScanHistory.url == request.url).first()
    if existing:
        # Generate the features for the UI even if the result is cached
        features = extract_features(request.url)
        return {
            "result": existing.output,
            "confidence": 95.0,
            "source": "database",
            "uses_https": uses_https(request.url),
            "features_array": features,
            "features": dict(zip(FEATURE_NAMES, features)) if FEATURE_NAMES else {}
        }

    # 2. ML prediction
    try:
        features     = extract_features(request.url)
        features_arr = np.array(features).reshape(1, -1)

        # predict_proba gives [P(legitimate), P(phishing)]
        # label mapping: legitimate=0, phishing=1
        # use saved threshold (0.4) instead of default 0.5
        proba        = MODEL.predict_proba(features_arr)[0]
        phish_prob   = float(proba[1])                          # P(phishing)
        prediction   = 1 if phish_prob >= THRESHOLD else 0
        result_label = "Phishing" if prediction == 1 else "Safe"
        confidence   = round(phish_prob * 100 if prediction == 1 else (1 - phish_prob) * 100, 2)

        db.add(ScanHistory(url=request.url, output=result_label, user_email=request.email))
        db.commit()

        return {
            "result": result_label,
            "confidence": confidence,
            "source": "ml_model",
            "uses_https": uses_https(request.url),
            "features_array": features,
            "features": dict(zip(FEATURE_NAMES, features)) if FEATURE_NAMES else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/extract")
def extract_url_features(url: str):
    """Allows the frontend History page to grab the features on demand"""
    try:
        features = extract_features(url)
        return {"features_array": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Gemini not configured.")
    try:
        # Get user's name from DB for personalization
        db_user = db.query(User).filter(User.email == request.email).first()
        user_name = db_user.name if db_user and db_user.name else request.email.split("@")[0]

        prompt = (
            f"{GEMINI_SYSTEM}\n\n"
            f"The user's name is: {user_name}\n"
            f"User message: {request.message}"
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return {"reply": response.text}
    except Exception as e:
        print(f" Gemini error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/all")
def get_all_history(db: Session = Depends(get_db)):
    rows = db.query(ScanHistory).order_by(ScanHistory.time.desc()).all()
    return [
        {"url": h.url, "result": h.output,
         "time": h.time.strftime("%Y-%m-%d %H:%M") if h.time else "—",
         "source": "database", "user": h.user_email}
        for h in rows
    ]


@app.get("/history/{email}")
def get_user_history(email: str, db: Session = Depends(get_db)):
    rows = db.query(ScanHistory).filter(
        ScanHistory.user_email == email
    ).order_by(ScanHistory.time.desc()).all()
    return [
        {"url": h.url, "result": h.output,
         "time": h.time.strftime("%Y-%m-%d %H:%M") if h.time else "—",
         "source": "database"}
        for h in rows
    ]


@app.get("/dashboard-stats")
def dashboard_stats(db: Session = Depends(get_db)):
    total    = db.query(ScanHistory).count()
    phishing = db.query(ScanHistory).filter(ScanHistory.output.ilike("phishing")).count()
    safe     = db.query(ScanHistory).filter(ScanHistory.output.ilike("safe")).count()
    threat_level = round((phishing / total) * 100, 2) if total else 0

    today          = datetime.utcnow().date()
    daily          = {(today - timedelta(days=i)).isoformat(): 0 for i in range(6, -1, -1)}
    daily_phishing = {(today - timedelta(days=i)).isoformat(): 0 for i in range(6, -1, -1)}

    for s in db.query(ScanHistory).all():
        if not s.time:
            continue
        day = s.time.date().isoformat()
        if day in daily:
            daily[day] += 1
            if s.output.lower().strip() == "phishing":
                daily_phishing[day] += 1

    return {
        "total_scanned": total, "phishing_detected": phishing,
        "safe_urls": safe, "threat_level": threat_level,
        "daily": daily, "daily_phishing": daily_phishing
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)