# 🚀 SMDTRIO - Spam SMS Detector

SMDTRIO is a **machine learning powered web application** that detects whether an SMS message is **Spam or Legitimate**.

The system combines **dual-engine machine learning models with AI analysis** to perform advanced SMS scanning and deliver instant spam detection with up to **99% accuracy**.

---

## ✨ Features

• 📩 Detects spam SMS messages
• ⚙️ Multiple detection engines:

* Logistic Regression
* Support Vector Machine (SVC)
* AI analysis using Groq API
  • 📊 Spam probability estimation
  • 🌐 Clean and modern web interface (Django)
  • ☁️ Easily deployable on cloud platforms

---

## 🛠️ Tech Stack

• Backend: Django
• Machine Learning: Scikit-Learn
• Vectorization: TF-IDF
• AI API: Groq
• Language: Python

---

## 📥 Download from GitHub

Download the project ZIP from GitHub and extract it to your system.

---

## ⚙️ Installation

Install all required dependencies:

pip install -r requirements.txt

---

## 🔐 Environment Setup

Create a `.env` file in the project root and add your **Groq API key**:

GROQ_API_KEY=your_groq_api_key_here

---

## 💻 Run Locally

Start the Django development server:

python manage.py runserver

Open in browser:

http://127.0.0.1:8000/

---

## ☁️ Deployment (Render / Production)

Run using Gunicorn:

gunicorn SMD.wsgi:application --bind 0.0.0.0:$PORT

---

## 🌐 Live Demo

https://smdtrio.onrender.com

---

## 📌 Future Improvements

• 📱 Mobile responsive enhancements
• 📊 Dashboard for analytics
• 🔐 User authentication system
• 🤖 More advanced AI models

---

## 👨‍💻 Author

Developed by **Jit Singha Mahapatra**

---
