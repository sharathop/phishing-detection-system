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
from model import extract_features



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
for model_path in ["final_phishing_model.pkl", "phishing_model.pkl"]:
    try:
        MODEL = joblib.load(model_path)
        print(f"✅ ML Model loaded: {model_path}")
        break
    except Exception as e:
        print(f"⚠  Could not load {model_path}: {e}")

# ── Gemini ────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client  = None
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI connected")
except Exception as e:
    print(f"⚠  Gemini not connected: {e}")

GEMINI_SYSTEM = (
    "You are PhishGuard AI, a cybersecurity assistant built into the PhishGuard "
    "phishing detection platform. "
    "PERSONALIZATION: The user's name will be provided at the start of each message. "
    "Always greet the user by their name in your first response. "
    "Use their name naturally in conversation to make it feel personal. "

    "URL ANALYSIS: If the user provides a URL, analyze it and explain each of these "
    "15 features one by one in simple terms: "
    "1. URL Length - longer URLs are more suspicious, "
    "2. Dot Count - too many dots suggest subdomains used to trick users, "
    "3. Hyphen Count - hyphens are common in fake domains, "
    "4. HTTPS - legitimate sites usually use HTTPS, "
    "5. Digits in Domain - numbers in domain names are suspicious, "
    "6. Suspicious Keywords - words like login, verify, secure, update, confirm, account, password, billing, suspended, recover, alert, unlock, validate, authenticate, authorize, reactivate, restore, renew, "
    "7. IP Address as Host - using raw IP instead of domain name is a red flag, "
    "8. @ Symbol - presence of @ in URL is a phishing trick, "
    "9. Domain Length - very long domains are suspicious, "
    "10. Subdomain Depth - too many subdomains are a red flag, "
    "11. Path Length - very long paths can indicate phishing, "
    "12. Query Parameters - excessive = signs suggest data harvesting, "
    "13. Free Hosting - hosted on weebly/wix/wordpress/blogspot/github.io/netlify/vercel/glitch, "
    "14. Special Characters in Domain - special chars in domain label are suspicious, "
    "15. Double Slash in Path - double slashes in path are a redirect trick. "
    "After explaining the features, give an overall verdict of Safe or Phishing with a summary. "

    "MODEL EXPLANATION: If the user asks how the model works, explain: "
    "The dataset was collected from PhishTank for phishing URLs and Cisco Umbrella Top Sites "
    "for legitimate URLs. After collecting the URLs, several lexical features such as URL length, "
    "number of dots, presence of suspicious keywords, and HTTPS usage were extracted. "
    "These features were then used to train a Random Forest machine learning model to classify "
    "URLs as phishing or legitimate. The model achieved 99.6% accuracy on the test dataset. "
    "A Random Forest works by building multiple decision trees and combining their results "
    "for a more accurate and robust prediction. "

    "RESTRICTIONS: You ONLY answer questions related to phishing, URL safety, cybersecurity, "
    "scan results, and online threats. "
    "If the user asks anything unrelated (e.g. sports, cooking, coding, general knowledge), "
    "politely refuse and say their name followed by: 'I can only help with phishing detection "
    "and cybersecurity questions. Please ask me about URL safety or online threats.' "
    "Be concise, clear, and professional."
)

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


# ── Trusted domains whitelist ─────────────────────────────────
TRUSTED_DOMAINS = [
    'google.com', 'email.com', 'youtube.com', 'googleads.com',
    'amazon.com', 'aws.amazon.com', 'amazonaws.com',
    'microsoft.com', 'office.com', 'live.com', 'outlook.com', 'azure.com',
    'apple.com', 'icloud.com',
    'facebook.com', 'instagram.com', 'whatsapp.com', 'messenger.com',
    'twitter.com', 'x.com',
    'linkedin.com',
    'github.com', 'githubusercontent.com',
    'stackoverflow.com',
    'wikipedia.org',
    'netflix.com',
    'zoom.us',
    'dropbox.com',
    'paypal.com',
    'adobe.com',
    'gemini.google.com',
    'accounts.google.com',
    'play.google.com',
    'maps.google.com',
    'vercel.app'
]

def is_trusted(url: str) -> bool:
    from urllib.parse import urlparse   
    try:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        return any(domain == td or domain.endswith('.' + td) for td in TRUSTED_DOMAINS)
    except:
        return False

@app.post("/scan")
async def scan_url(request: ScanRequest, db: Session = Depends(get_db)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="ML model not loaded.")

    # 1. Check DB cache
    existing = db.query(ScanHistory).filter(ScanHistory.url == request.url).first()
    if existing:
        return {"result": existing.output, "confidence": 95.0, "source": "database"}

    # 2. Check trusted domain whitelist
    if is_trusted(request.url):
        db.add(ScanHistory(url=request.url, output="Safe", user_email=request.email))
        db.commit()
        return {"result": "Safe", "confidence": 99.0, "source": "trusted_domain"}

    # 3. ML prediction
    try:
        features     = extract_features(request.url)
        features_arr = np.array(features).reshape(1, -1)
        prediction   = int(MODEL.predict(features_arr)[0])
        result_label = "Phishing" if prediction == 1 else "Safe"
        confidence   = 95.0
        try:
            proba      = MODEL.predict_proba(features_arr)[0]
            confidence = round(float(np.max(proba)) * 100, 2)
        except Exception:
            pass
        db.add(ScanHistory(url=request.url, output=result_label, user_email=request.email))
        db.commit()
        return {"result": result_label, "confidence": confidence, "source": "ml_model"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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
        print(f"❌ Gemini error: {e}")
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


    ###