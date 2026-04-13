import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import requests
import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from intent_handler import detect_conversational_intent
from dotenv import load_dotenv

load_dotenv()

# ── Load dataset & embeddings ──────────────────────────────────────────────────
df1 = pd.read_json('train_expanded.json', lines=True)
df2 = pd.read_json('generated_faqs.json', lines=True)
df_combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = joblib.load("question_embeddings.pkl")
answer_embeddings   = joblib.load("answer_embeddings.pkl")

# ── Groq API Fallback ──────────────────────────────────────────────────────────
#  key is loaded from .env
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def ask_groq_fallback(user_question):
    if not GROQ_API_KEY:
        return "I'm not sure about that. Please contact our support team for help! 📧"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",  #   GROQ FREE MODEL
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly ecommerce customer support assistant. "
                    "Answer the user's question politely and concisely. "
                    "If the question is about store-specific information you don't know "
                    "(like a specific order ID, tracking number, or account detail), "
                    "politely say you don't have access to that and suggest they contact support. "
                    "Keep responses short — 2 to 3 sentences max. Use emojis occasionally to stay friendly."
                )
            },
            {
                "role": "user",
                "content": user_question
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return "I'm having trouble connecting right now. Please try again or contact support! 🙏"

#  FAQ Matching
def get_faq_answer(user_question):
    user_embedding = model.encode([user_question])
    similarities   = cosine_similarity(user_embedding, question_embeddings) # type: ignore
    best_match     = np.argmax(similarities)
    best_score     = similarities[0][best_match]

    if best_score < 0.5:
        return None, best_score

    return df_combined['answer'].iloc[best_match], best_score

# Main Chatbot Logic 
def chatbot(user_input, history):
    user_input = user_input.strip()

    if not user_input:
        return "Please type a question! 😊"

    # Layer 1: Conversational Intent Detection
    conversational_reply = detect_conversational_intent(user_input)
    if conversational_reply:
        return conversational_reply

    # Layer 2: FAQ Dataset Matching
    faq_answer, score = get_faq_answer(user_input)
    if faq_answer:
        return faq_answer

    # Layer 3: Groq API Fallback
    return ask_groq_fallback(user_input)

#  Gradio UI 
demo = gr.ChatInterface(
    fn=chatbot,
    title="🛒 E-Commerce FAQ Chatbot",
    description="Ask me anything about orders, payments, returns and more! You can also just say hello 👋",
    examples=[
        "Hello!",
        "How can I cancel my order?",
        "What payment methods do you accept?",
        "How do I return a product?",
        "How can I track my order?",
        "Thank you!"
    ]
)

demo.launch(share=True)