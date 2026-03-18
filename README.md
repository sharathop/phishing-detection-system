# Dev-of-a-Machine-Learning-Based-Phishing-Website-Detection-and-Classif-System_Feb_Batch-8_2026

# 🛡️ PhishGuard — ML-Based Phishing Website Detection System

## 📌 Project Overview

**PhishGuard** is a full-stack web application developed as part of the
**Infosys Springboard Internship (Batch 8, 2026)**.

The system detects whether a given URL is **Safe or Phishing** using a Machine Learning model and provides detailed insights, history tracking, and AI-powered explanations.

---

## 🎯 Objective

To design and deploy an end-to-end phishing detection system that:

* Identifies malicious URLs using ML techniques
* Provides real-time feedback to users
* Stores scan history and analytics
* Enhances user awareness of cybersecurity threats

---

## 🚀 Live Demo

* 🌐 Frontend (Vercel): https://your-vercel-link
* ⚙️ Backend (Render): https://your-render-link

---

## 🧠 Key Features

* 🔐 User Authentication (Register/Login)
* 🔍 URL Phishing Detection
* 📊 Confidence Score
* 🧾 Scan History Tracking
* 📈 Dashboard Analytics
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
* Alembic (Database Migrations)
* Deployed on Render

### Machine Learning

* Random Forest Classifier
* URL Feature Engineering (15 features)

### Database

* PostgreSQL (Neon Cloud Database)

---

## 📂 Project Structure

```bash
PhishGuard/
│
├── Frontend/
│   └── index.html
│
├── Backend/
│   ├── main.py
│   ├── db_models.py
│   ├── db_setup.py
│   ├── model.py
│   ├── migrations/
│   └── requirements.txt
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sharathop/phishing-detection-system.git
cd phishing-detection-system/PhishGuard
```

---

### 2️⃣ Backend Setup

```bash
cd Backend
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

### 3️⃣ Environment Variables

Create a `.env` file inside `Backend/`:

```env
DATABASE_URL=your_neon_database_url
GEMINI_API_KEY=your_gemini_api_key
```

---

### 4️⃣ Run Backend

```bash
uvicorn main:app --reload
```

👉 Backend runs on:
http://127.0.0.1:8000

---

### 5️⃣ Frontend Setup

Open:

```bash
Frontend/index.html
```

Ensure API URL is set correctly:

```javascript
const API = "http://127.0.0.1:8000";
```

---

## 🌐 Deployment Architecture

```text
Frontend (Vercel)
        ↓
Backend (Render - FastAPI)
        ↓
Database (Neon PostgreSQL)
```

---

## 📊 Machine Learning Details

* Dataset Sources:

  * PhishTank (phishing URLs)
  * Cisco Umbrella Top Sites (legitimate URLs)
* Model: Random Forest Classifier
* Features: 15 URL-based features


---

## 🔐 Security Best Practices

* Environment variables used for sensitive data
* `.env` file excluded via `.gitignore`
* Database credentials not exposed in code

---


## 📌 Acknowledgment

This project was developed as part of the
**Infosys Springboard Internship Program — Batch 8 (2026)**,
focusing on real-world application of Machine Learning and Full-Stack Development in cybersecurity.

---
