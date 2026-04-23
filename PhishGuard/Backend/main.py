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
import os
load_dotenv()

from db_setup import get_db, engine
from db_models import Base, User, ScanHistory
from feature_extraction import extract_features

import pandas as pd

df = pd.read_csv("dataset_phishing.csv")
FEATURE_NAMES = list(df.drop(columns=["url", "status"]).columns)

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
for model_path in ["xgboost_phishing_model.pkl"]:
    try:
        MODEL = joblib.load(model_path)
        print(f" ML Model loaded: {model_path}")
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
- The model uses 80+ extracted URL features.
- Feature selection was tested, but using all features gave better results.
- The final system is optimized for high accuracy and real-time phishing detection.

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

    # 1. Check DB cache
    existing = db.query(ScanHistory).filter(ScanHistory.url == request.url).first()
    if existing:
        # Generate the 86 features for the UI even if the result is cached
        features = extract_features(request.url)
        return {
            "result": existing.output, 
            "confidence": 95.0, 
            "source": "database",
            "features_array": features,
            "features": dict(zip(FEATURE_NAMES, features))
        }

    # 3. ML prediction
    try:
        features     = extract_features(request.url)
        features_arr = np.array(features).reshape(1, -1)
        prediction   = int(MODEL.predict(features_arr)[0])
        result_label = "Safe" if prediction == 1 else "Phishing"
        confidence   = 95.0
        try:
            proba      = MODEL.predict_proba(features_arr)[0]
            confidence = round(float(proba[prediction]) * 100, 2)
        except Exception:
            pass
            
        db.add(ScanHistory(url=request.url, output=result_label, user_email=request.email))
        db.commit()

        return {
         "result": result_label,
         "confidence": confidence,
         "source": "ml_model",
         "features_array": features,
         "features": dict(zip(FEATURE_NAMES, features))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/extract")
def extract_url_features(url: str):
    """Allows the frontend History page to grab the 86 features on demand"""
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