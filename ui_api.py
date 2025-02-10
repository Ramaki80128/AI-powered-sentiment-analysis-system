from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle

# Load the Model, Vectorizer, and Label Encoder
try:
    model = pickle.load(open("sentiment_model.sav", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise SystemExit


# Initialize FastAPI App
app = FastAPI()

# Prediction Function
def predict_sentiment(tweet):
    tweet_vector = vectorizer.transform([tweet])
    prediction_encoded = model.predict(tweet_vector)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    return prediction

# HTML UI (Root Route)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Tweet Sentiment Analysis</title>
        <style>
            body { font-family: 'Arial', sans-serif; background: #f4f4f9; margin: 0; padding: 20px; }
            .container { max-width: 500px; margin: auto; background: white; padding: 20px; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 10px; }
            textarea, button { padding: 10px; width: 100%; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; }
            button { background: linear-gradient(90deg, #007BFF, #00C6FF); color: white; border: none; cursor: pointer; transition: 0.3s; }
            button:hover { background: linear-gradient(90deg, #0056b3, #0096c7); }
            h1 { color: #333; text-align: center; }
            #result { text-align: center; font-weight: bold; margin-top: 20px; padding: 10px; border-radius: 5px; }
            .positive { background-color: #d4edda; color: #155724; }
            .negative { background-color: #f8d7da; color: #721c24; }
            .neutral { background-color: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tweet Sentiment Analysis</h1>
            <textarea id="tweet" rows="4" placeholder="Enter your tweet here..."></textarea>
            <button onclick="getPrediction()">Analyze Sentiment</button>
            <div id="result"></div>
        </div>

        <script>
            async function getPrediction() {
                const tweet = document.getElementById("tweet").value;
                const resultDiv = document.getElementById("result");

                if (tweet.trim() === "") {
                    resultDiv.innerText = "⚠️ Please enter a tweet.";
                    resultDiv.className = "neutral";
                    return;
                }

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: tweet })
                });

                const result = await response.json();
                resultDiv.innerText = "Predicted Sentiment: " + result.sentiment;
                resultDiv.className = result.sentiment.toLowerCase();
            }
        </script>
    </body>
    </html>
    """

# API Route for Sentiment Prediction
class TweetInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(data: TweetInput):
    sentiment = predict_sentiment(data.text)
    return {"sentiment": sentiment}
