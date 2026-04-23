# 🛡️ PhishGuard — ML-Based Phishing Website Detection System

## 📌 Project Overview

PhishGuard is a full-stack web application developed as part of the Infosys Springboard Internship (Batch 8, 2026).

The system detects whether a given URL is **Safe or Phishing** using a Machine Learning model and provides real-time insights, scan history, analytics, and AI-powered explanations.

---

## 🎯 Objective

* Detect phishing URLs using Machine Learning
* Provide real-time feedback with confidence score
* Store scan history and analytics
* Improve cybersecurity awareness for users

---

## 🚀 Live Demo

* 🌐 Frontend (Vercel):https://phishing-detection-system-tau.vercel.app
* ⚙️ Backend (Render): https://phishguard-api-emtj.onrender.com

---

## 🧠 Key Features

* 🔐 User Authentication (Register/Login)
* 🔍 URL Phishing Detection
* 📊 Confidence Score
* 📈 Feature-Level Analysis (84 Features)
* 🧾 Scan History Tracking
* 📊 Dashboard Analytics
* 🤖 AI Chatbot (Google Gemini Integration)
* ✅ Trusted Domain Whitelisting

---

## 🏗️ Tech Stack

### Frontend

* HTML, CSS, JavaScript
* Deployed on Vercel

### Backend

* FastAPI (Python)
* SQLAlchemy ORM
* Alembic (Migrations)
* Deployed on Render

### Machine Learning

* XGBoost Classifier
* 80+ Engineered URL Features
* Feature Importance Analysis

### Database

* PostgreSQL (Neon Cloud)

---

## 📂 Project Structure

```
PhishGuard/
│
├── Frontend/
│   └── index.html
│
├── Backend/
│   ├── main.py
│   ├── model.py
│   ├── feature_extraction.py
│   ├── db_models.py
│   ├── db_setup.py
│   ├── migrations/
│   └── requirements.txt
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```
git clone https://github.com/sharathop/phishing-detection-system.git
cd phishing-detection-system/PhishGuard
```

---

### 2️⃣ Backend Setup

```
cd Backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

### 3️⃣ Environment Variables

Create `.env` inside Backend:

```
DATABASE_URL=your_neon_database_url
GEMINI_API_KEY=your_gemini_api_key
```

---

### 4️⃣ Run Backend

```
uvicorn main:app --reload
```

👉 Runs at: http://127.0.0.1:8000

---

### 5️⃣ Frontend Setup

Open:

```
Frontend/index.html
```

Set API:

```
const API = "http://127.0.0.1:8000";
```

---

## 🌐 Deployment Architecture

```
Frontend (Vercel)
        ↓
Backend (Render - FastAPI)
        ↓
Database (Neon PostgreSQL)
```

---

## 📊 Machine Learning Details

### Dataset Sources

https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset

### Model Comparison

Multiple models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

### Final Model

* XGBoost selected based on highest accuracy (~96–97%)

### Features

* 80+ engineered features including:

  * URL structure
  * Domain properties
  * lexical patterns
  * statistical indicators

### Feature Selection

* Random Forest feature importance was used
* Experiments showed **all features performed better**, so full feature set retained

---

## 🔐 Security Best Practices

* Environment variables for sensitive data
* `.env` excluded via `.gitignore`
* Secure database connection
* Input validation for URLs

---

## 📌 Acknowledgment

Developed as part of **Infosys Springboard Internship — Batch 8 (2026)**
Focused on real-world application of Machine Learning and Full-Stack Development in cybersecurity.

---

## ⭐ Future Improvements

* Real-time browser extension
* Deep learning-based detection
* API rate limiting & security enhancements
* Advanced threat intelligence integration
