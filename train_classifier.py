import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "hi", "hello", "hey there", "good morning",           # greeting
    "bye", "goodbye", "see you later", "talk to you soon", # goodbye
    "thanks", "thank you", "appreciate it",               # thanks
    "what services do you offer", "can i know something", "how can i track my order",
    "what is your refund policy", "where is my package" 
    "how can I track my order", "where is my package", "track order",
    "what are the dashboard options", "show me dashboard options",
    "what is your refund policy", "do you offer refunds?"
]
labels = [
    "track_order", "track_order", "track_order",
    "dashboard_options", "dashboard_options",
    "refund_policy", "refund_policy","track_order", "track_order", "track_order",
    "dashboard_options", "dashboard_options",
    "refund_policy", "refund_policy"
]
  # question


labels = [
    "greeting", "greeting", "greeting", "greeting",
    "goodbye", "goodbye", "goodbye", "goodbye",
    "thanks", "thanks", "thanks",
    "question", "question", "question", "question", "question"
]

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

# Save model
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/intent_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Better training complete. Model updated.")

