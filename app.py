import streamlit as st
import joblib
import os

# Load model and vectorizer
try:
    model = joblib.load("reddit_sentiment_model.pkl")
    vectorizer = joblib.load("reddit_vectorizer.pkl")
except FileNotFoundError:
    st.error("Error path")
    st.stop()

def clean_text(text):
    return text.lower()

st.title("Sentiment Predictor")
st.write("Enter your feeling so that i can analyze.")

user_input = st.text_area("Enter here")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a text before analyzing.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment_map = {
            1: "Positive 😊",
            0: "Neutral 😐",
            -1: "Negative 😠"
        }
        sentiment = sentiment_map.get(pred, "Unknown 🤔")
        st.success(f"Sentiment: **{sentiment}**")
