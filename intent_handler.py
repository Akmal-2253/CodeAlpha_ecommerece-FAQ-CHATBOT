
import re


INTENT_RESPONSES = {
    "greeting": {
        "patterns": [
            r"\b(hello|hi|hey|howdy|greetings|good morning|good afternoon|good evening|salam|assalam)\b"
        ],
        "responses": [
            "Hello! 👋 Welcome to our store! How can I help you today?",
            "Hi there! 😊 I'm your shopping assistant. What can I do for you?",
            "Hey! Great to see you! Ask me anything about orders, returns, or payments! 🛒"
        ]
    },
    "how_are_you": {
        "patterns": [
            r"\b(how are you|how r u|how do you do|hows it going|how's it going|you good|u good)\b"
        ],
        "responses": [
            "I'm doing great, thanks for asking! 😄 Ready to help you with anything you need!",
            "All good here! 🤖 More importantly, how can I assist YOU today?"
        ]
    },
    "thanks": {
        "patterns": [
            r"\b(thank|thanks|thank you|thankyou|thx|ty|appreciate|great help|helpful)\b"
        ],
        "responses": [
            "You're welcome! 😊 Is there anything else I can help you with?",
            "Happy to help! 🙌 Feel free to ask if you have more questions!",
            "Anytime! Let me know if there's anything else! ✨"
        ]
    },
    "farewell": {
        "patterns": [
            r"\b(bye|goodbye|see you|see ya|later|take care|cya|good night|goodnight)\b"
        ],
        "responses": [
            "Goodbye! 👋 Have a wonderful day and happy shopping!",
            "See you later! 😊 Come back anytime you need help!",
            "Take care! 🛍️ Hope we could help you today!"
        ]
    },
    "bot_identity": {
        "patterns": [
            r"\b(who are you|what are you|are you a bot|are you human|are you ai|what is your name|your name)\b"
        ],
        "responses": [
            "I'm your E-Commerce Assistant Bot! 🤖 I'm here to help with orders, returns, payments, and more. What do you need?",
            "I'm an AI chatbot built to answer your shopping questions! Ask me about orders, tracking, refunds — I've got you covered! 🛒"
        ]
    },
    "affirmation": {
        "patterns": [
            r"^(ok|okay|alright|got it|sure|fine|noted|understood|yes|yeah|yep|k|kk)$"
        ],
        "responses": [
            "Great! 😊 Let me know if you need anything else!",
            "Awesome! Feel free to ask more questions anytime! 🙌"
        ]
    }
}

import random

def detect_conversational_intent(user_input):
    
    text = user_input.lower().strip()
    
    for intent, data in INTENT_RESPONSES.items():
        for pattern in data["patterns"]:
            if re.search(pattern, text):
                return random.choice(data["responses"])
    
    return None