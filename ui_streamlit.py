import streamlit as st
import pickle

# Load the Model, Vectorizer, and Label Encoder
try:
    model = pickle.load(open("sentiment_model.sav", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()  # Stop execution if files are missing

# Prediction Function
def predict_sentiment(tweet):
    tweet_vector = vectorizer.transform([tweet])  # Ensure vectorizer is loaded
    prediction_encoded = model.predict(tweet_vector)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    return prediction

# Streamlit UI
st.title("Tweet Sentiment Analysis")

tweet = st.text_area("Enter your tweet here:")
if st.button("Analyze Sentiment"):
    if tweet.strip():  # Check for non-empty input
        sentiment = predict_sentiment(tweet)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("⚠️ Please enter a tweet before analyzing.")
