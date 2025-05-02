from transformers import pipeline
from app.utils import classify_intent, fetch_response

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def get_bot_response(user_input):
    intent = classify_intent(user_input)
    if intent == "question":
        context = "We offer 24/7 customer support. You can track your order via the dashboard."
        return qa_pipeline(question=user_input, context=context)['answer']
    else:
        return fetch_response(intent)
 
