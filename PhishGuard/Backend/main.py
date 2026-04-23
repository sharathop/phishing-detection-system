"""
PhishGuard — FastAPI Backend (Production Ready)
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

from db_setup import get_db
from db_models import User, ScanHistory
from feature_extraction import extract_features

# -------------------- APP --------------------
app = FastAPI(title="PhishGuard API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- SERVE FRONTEND --------------------
@app.get("/")
def serve_frontend():
    for path in ["index.html", "../index.html", "../Frontend/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="index.html not found.")

# -------------------- LOAD MODEL + FEATURES --------------------
MODEL = None
FEATURE_NAMES = None

try:
    data = joblib.load("xgboost_phishing_model.pkl")
    MODEL = data["model"]
    FEATURE_NAMES = data["features"]
    print(" Model and features loaded")
except Exception as e:
    print(f" Model load failed: {e}")

# -------------------- GEMINI --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = None

try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(" Gemini connected")
except Exception as e:
    print(f" Gemini error: {e}")

GEMINI_SYSTEM = """
You are PhishGuard AI, a cybersecurity assistant.

- Greet user by name
- Only answer phishing / URL / cybersecurity queries
- Refuse unrelated questions
- Keep answers short, clear, structured
"""

# -------------------- SCHEMAS --------------------
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

# -------------------- AUTH --------------------
@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email exists")
    db.add(User(name=user.name, email=user.email, password=user.password))
    db.commit()
    return {"status": "success"}

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        User.email == user.email,
        User.password == user.password
    ).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "status": "success",
        "user": {"name": db_user.name, "email": db_user.email}
    }

# -------------------- SCAN --------------------
@app.post("/scan")
async def scan_url(request: ScanRequest, db: Session = Depends(get_db)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = extract_features(request.url)
    features_arr = np.array(features).reshape(1, -1)

    prediction = int(MODEL.predict(features_arr)[0])
    result_label = "Safe" if prediction == 1 else "Phishing"

    confidence = 95.0
    try:
        proba = MODEL.predict_proba(features_arr)[0]
        confidence = round(float(proba[prediction]) * 100, 2)
    except:
        pass

    db.add(ScanHistory(url=request.url, output=result_label, user_email=request.email))
    db.commit()

    return {
        "result": result_label,
        "confidence": confidence,
        "features": dict(zip(FEATURE_NAMES, features))
    }

# -------------------- CHAT --------------------
@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Gemini not configured")

    db_user = db.query(User).filter(User.email == request.email).first()
    name = db_user.name if db_user else request.email

    prompt = f"{GEMINI_SYSTEM}\nUser: {name}\nMessage: {request.message}"

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return {"reply": response.text}

# -------------------- HISTORY --------------------
@app.get("/history/{email}")
def get_history(email: str, db: Session = Depends(get_db)):
    rows = db.query(ScanHistory).filter(
        ScanHistory.user_email == email
    ).order_by(ScanHistory.time.desc()).all()

    return [
        {"url": r.url, "result": r.output}
        for r in rows
    ]

# -------------------- DASHBOARD --------------------
@app.get("/dashboard-stats")
def dashboard(db: Session = Depends(get_db)):
    total = db.query(ScanHistory).count()
    phishing = db.query(ScanHistory).filter(
        ScanHistory.output == "Phishing"
    ).count()

    safe = total - phishing

    return {
        "total": total,
        "phishing": phishing,
        "safe": safe
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)