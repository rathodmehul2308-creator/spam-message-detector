from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


# Load trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load text vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    # Convert message into numbers
    message_vector = vectorizer.transform([message])

    # Predict spam or not spam
    prediction = model.predict(message_vector)[0]

    # Get confidence
    probabilities = model.predict_proba(message_vector)[0]
    confidence = max(probabilities) * 100

    if prediction == "spam":
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)