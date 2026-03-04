from django.shortcuts import render
import os
import joblib
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML models
model1 = joblib.load(os.path.join(BASE_DIR, "mySVCModel.pkl"))
model2 = joblib.load(os.path.join(BASE_DIR, "myModel.pkl"))

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def index(request):
    return render(request, "index.html")


def checkspam(request):

    if request.method == "POST":

        algo = request.POST.get("algo")
        rawtext = request.POST.get("rawtext", "").strip()

        if rawtext == "":
            return render(request, "index.html", {"error": "Please enter a message"})

        ans = ""
        probability = ""
        model_name = ""

        # -------------------------
        # SVC
        # -------------------------
        if algo == "1":

            prob = model1.predict_proba([rawtext])[0]

            spam_prob = prob[1] * 100
            ham_prob = prob[0] * 100

            ans = "spam" if spam_prob > ham_prob else "ham"
            probability = f"{spam_prob:.2f}%"

            model_name = "SVC"
        # -------------------------
        # Logistic Regression
        # -------------------------
        elif algo == "2":

            prob = model2.predict_proba([rawtext])[0]

            spam_prob = prob[1] * 100
            ham_prob = prob[0] * 100

            ans = "spam" if spam_prob > ham_prob else "ham"
            probability = f"{spam_prob:.2f}%"

            model_name = "Logistic Regression"


        # -------------------------
        # AI Model
        # -------------------------
        elif algo == "3":

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an SMS spam detection AI."
                    },
                    {
                        "role": "user",
                        "content": f"""
Classify this SMS as spam or ham.

Rules:
Spam = promotions, loans, prizes, suspicious links.
Ham = normal personal messages.

Reply EXACTLY in this format:
spam|confidence
or
ham|confidence

Example:
spam|92
ham|10

SMS: {rawtext}
"""
                    }
                ],
                temperature=0
            )

            try:
                result = completion.choices[0].message.content.strip().lower()
                parts = result.split("|")

                ans = parts[0]
                probability = parts[1] + "%"

            except:
                ans = "ham"
                probability = "AI"

            model_name = "AI Model"

        param = {
            "prediction": ans,
            "original_msg": rawtext,
            "model_type": model_name,
            "probability": probability
        }

        return render(request, "output.html", param)

    return render(request, "index.html")