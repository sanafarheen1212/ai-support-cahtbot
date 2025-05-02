import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/intent_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)
def classify_intent(text):
    X = vectorizer.transform([text])
    prediction = classifier.predict(X)[0]
    print(f"Intent predicted: {prediction}")  # Debug line
    return prediction

def fetch_response(intent):
    responses = {
       "greeting": "Hello! How can I assist you today?",
    "track_order": "You can track your order via the dashboard.",
    "dashboard_options": "Dashboard options include order history, profile, and support.",
    "refund_policy": "Our refund policy allows returns within 30 days of purchase.",
    "thanks": "You're welcome!",
    "goodbye": "Thank you for contacting us. Goodbye!"
    }
    return responses.get(intent, responses["fallback"])

