"""
PhishGuard — FastAPI Backend
Handles Auth, Database Caching, and ML Prediction.
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import joblib, numpy as np
from pydantic import BaseModel, EmailStr

from db_setup import get_db, engine
from db_models import Base, User, ScanHistory
from model import extract_features

Base.metadata.create_all(bind=engine)

app = FastAPI(title="PhishGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load ML Model ─────────────────────────────────────────────
MODEL = None
for model_path in ["final_phishing_model.pkl", "phishing_model.pkl"]:
    try:
        MODEL = joblib.load(model_path)
        print(f"✅ ML Model loaded: {model_path}")
        break
    except Exception as e:
        print(f"⚠  Could not load {model_path}: {e}")

if MODEL is None:
    print("❌ No model file found. /scan will return 500 until a model is loaded.")


# ── Pydantic Schemas ──────────────────────────────────────────
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ScanRequest(BaseModel):
    url: str
    email: EmailStr


# ── ROUTES ────────────────────────────────────────────────────

@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(User.gmail == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    db.add(User(name=user.name, gmail=user.email, password=user.password))
    db.commit()
    return {"status": "success", "message": "User created successfully"}


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        User.gmail == user.email,
        User.password == user.password
    ).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {
        "status": "success",
        "user": {
            "name" : db_user.name or db_user.gmail,
            "email": db_user.gmail
        }
    }


@app.post("/scan")
async def scan_url(request: ScanRequest, db: Session = Depends(get_db)):
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="ML model not loaded. Make sure final_phishing_model.pkl exists."
        )

    # 1. Check DB cache first
    existing = db.query(ScanHistory).filter(ScanHistory.url == request.url).first()
    if existing:
        return {
            "result"    : existing.output,
            "confidence": 95.0,
            "source"    : "database"
        }

    # 2. Run ML prediction
    try:
        features     = extract_features(request.url)
        features_arr = np.array(features).reshape(1, -1)
        prediction   = int(MODEL.predict(features_arr)[0])
        result_label = "Phishing" if prediction == 1 else "Safe"

        confidence = 95.0
        try:
            proba      = MODEL.predict_proba(features_arr)[0]
            confidence = round(float(np.max(proba)) * 100, 2)
        except Exception:
            pass

        # 3. Save to DB
        db.add(ScanHistory(
            url        = request.url,
            output     = result_label,
            user_gmail = request.email
        ))
        db.commit()

        return {
            "result"    : result_label,
            "confidence": confidence,
            "source"    : "ml_model"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/history/all")
def get_all_history(db: Session = Depends(get_db)):
    """Global history — all users combined, used for dashboard table."""
    rows = db.query(ScanHistory).order_by(ScanHistory.time.desc()).all()
    return [
        {
            "url"   : h.url,
            "result": h.output,
            "time"  : h.time.strftime("%Y-%m-%d %H:%M") if h.time else "—",
            "source": "database",
            "user"  : h.user_gmail
        }
        for h in rows
    ]

@app.get("/history/{email}")
def get_user_history(email: str, db: Session = Depends(get_db)):
    rows = db.query(ScanHistory).filter(
        ScanHistory.user_gmail == email
    ).order_by(ScanHistory.time.desc()).all()

    return [
        {
            "url"   : h.url,
            "result": h.output,
            "time"  : h.time.strftime("%Y-%m-%d %H:%M") if h.time else "—",
            "source": "database"
        }
        for h in rows
    ]

from datetime import datetime, timedelta

@app.get("/dashboard-stats")
def dashboard_stats(db: Session = Depends(get_db)):

    total = db.query(ScanHistory).count()

    phishing = db.query(ScanHistory).filter(
        ScanHistory.output.ilike("phishing")
    ).count()

    safe = db.query(ScanHistory).filter(
        ScanHistory.output.ilike("safe")
    ).count()

    threat_level = round((phishing / total) * 100, 2) if total else 0

    # ── last 7 days setup ──
    today = datetime.utcnow().date()

    daily = {
        (today - timedelta(days=i)).isoformat(): 0
        for i in range(6, -1, -1)
    }

    daily_phishing = {
        (today - timedelta(days=i)).isoformat(): 0
        for i in range(6, -1, -1)
    }

    scans = db.query(ScanHistory).all()

    for s in scans:
        if not s.time:
            continue

        day = s.time.date().isoformat()

        if day in daily:
            daily[day] += 1

            if s.output.lower().strip() == "phishing":
                daily_phishing[day] += 1

    return {
        "total_scanned": total,
        "phishing_detected": phishing,
        "safe_urls": safe,
        "threat_level": threat_level,
        "daily": daily,
        "daily_phishing": daily_phishing
    }

@app.get("/dashboard-charts")
def dashboard_charts(db: Session = Depends(get_db)):

    rows = db.query(ScanHistory).all()

    daily = {}
    phishing = 0
    safe = 0

    for r in rows:
        day = r.time.strftime("%Y-%m-%d") if r.time else "unknown"

        daily[day] = daily.get(day, 0) + 1

        if r.output == "Phishing":
            phishing += 1
        else:
            safe += 1

    return {
        "daily": daily,
        "split": {
            "phishing": phishing,
            "safe": safe
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)