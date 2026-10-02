import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# ==============================
# 1. Training Dataset
# ==============================

data = {
    "message": [
        "Congratulations! You won a lottery!",
        "You have won 1 crore rupees. Claim now!",
        "Congratulations! You won a free iPhone!",
        "Click this link to claim your prize",
        "You have been selected for a cash reward",
        "Win a free vacation today",
        "You won a lucky draw. Send your bank details",
        "Urgent! Claim your lottery prize now",
        "Get free money by clicking this link",
        "Congratulations! You are the lucky winner",

        "Your class starts at 10 AM",
        "Please submit your assignment tomorrow",
        "Your exam will start from Monday",
        "Can you send me today's notes?",
        "Let's meet in the college tomorrow",
        "The Python lecture is at 9 AM",
        "Please bring your practical file",
        "Your attendance is 80 percent",
        "The teacher uploaded the assignment",
        "College will remain closed tomorrow"
    ],

    "label": [
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",

        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam",
        "not_spam"
    ]
}


# ==============================
# 2. Create DataFrame
# ==============================

df = pd.DataFrame(data)


# ==============================
# 3. Convert Text into Numbers
# ==============================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["message"])
y = df["label"]


# ==============================
# 4. Train Machine Learning Model
# ==============================

model = MultinomialNB()

model.fit(X, y)

folder = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(folder, "spam_model.pkl")
vectorizer_path = os.path.join(folder, "vectorizer.pkl")

with open(model_path, "wb") as file:
    pickle.dump(model, file)

with open(vectorizer_path, "wb") as file:
    pickle.dump(vectorizer, file)

print("Model trained successfully!")
print("spam_model.pkl created!")
print("vectorizer.pkl created!")