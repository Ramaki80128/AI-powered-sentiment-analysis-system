# AI-powered-sentiment-analysis-system
AI system that analyzes customer reviews
## Overview
This project focuses on sentiment analysis of tweets using machine learning techniques. It includes data preprocessing, model training, evaluation, and deployment with both API and UI interfaces.

## Features
- Data preprocessing and cleaning
- Feature extraction using TF-IDF Vectorization
- Model training with machine learning algorithms
- Sentiment prediction (Positive, Negative, Neutral)
- Deployment using FastAPI (API) and Streamlit (UI)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd sentiment-analysis
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Model Training
Train the sentiment analysis model using the provided Jupyter Notebook. Save the trained model along with the vectorizer and label encoder:
```bash
jupyter notebook train.ipynb
```

### Streamlit UI
Run the Streamlit app to analyze tweet sentiments:
```bash
streamlit run ui_streamlit.py
```
- Enter your tweet in the text area.
- Click on "Analyze Sentiment" to get the prediction.

### FastAPI UI and API
Run the FastAPI app for both API and a simple HTML UI:
```bash
uvicorn ui_api:app --reload
```
- Open `http://127.0.0.1:8000/` in your browser for the UI.
- Send POST requests to `http://127.0.0.1:8000/predict` with JSON:
  ```json
  {
    "text": "I love this!"
  }
  ```

## Project Structure
```
.
├── data
├── models
│   ├── sentiment_model.sav
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
├── ui_streamlit.py
├── ui_api.py
├── train_model.ipynb
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.x
- Libraries: scikit-learn, pandas, numpy, streamlit, fastapi, uvicorn, pickle


