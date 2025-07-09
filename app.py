import streamlit as st
import joblib
import os

# Load model and vectorizer
try:
    model = joblib.load("reddit_sentiment_model.pkl")
    vectorizer = joblib.load("reddit_vectorizer.pkl")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Make sure the .pkl files are in the correct path.")
    st.stop()

def clean_text(text):
    # Add more preprocessing as needed (e.g., removing punctuation, stopwords)
    return text.lower()

st.title("Sentiment Predictor")
st.write("Enter your feeling so that i can analyze XD")

user_input = st.text_area("Enter  here")

if st.button("Analyze yoooo"):
    if not user_input.strip():
        st.warning("Please enter a comment before analyzing.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        sentiment_map = {
            1: "Thikchaaa 😊",
            0: "Thikei Matra 😐",
            -1: "Galat bhawanaaaa lmao 😠"
        }
        sentiment = sentiment_map.get(pred, "Unknown 🤔")
        st.success(f"Sentiment: **{sentiment}**")
