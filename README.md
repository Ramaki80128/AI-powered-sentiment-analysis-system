# AI-powered-sentiment-analysis-system
AI system that analyzes customer reviews
# Sentiment Analysis Project

## Overview
This project focuses on sentiment analysis of tweets using machine learning techniques. It includes data preprocessing, model training, evaluation, and deployment with both API and UI interfaces.

## Features
- Data preprocessing and cleaning
- Feature extraction using TF-IDF Vectorization
- Model training with machine learning algorithms
- Sentiment prediction (Positive, Negative, Neutral)
- Deployment using FastAPI (API) and Streamlit (UI)

## Dataset
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd AI-powered-sentiment-analysis-system
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r Requirements.txt
   ```

## Usage

### Model Training
Train the sentiment analysis model using the provided scripts. Save the trained model along with the vectorizer and label encoder:
```bash
jupyter notebook train.ipynb
```
### Model Selection
Multiple machine learning algorithms were tested for sentiment analysis, including Logistic Regression, Random Forest, XGBoost, and LSTM. The best model was selected based on the following evaluation metrics:
- **Accuracy**: Measures the percentage of correct predictions.
- **Precision, Recall, and F1-Score**: Evaluate the model's performance in classifying positive, negative, and neutral sentiments.
- **Confusion Matrix**: Provides insight into misclassifications.
- **Cross-Validation Scores**: Ensures model robustness and generalization.

The model with the highest F1-Score and balanced precision-recall scores across all sentiment classes was chosen as the final model.
### Streamlit UI
Run the Streamlit app to analyze tweet sentiments:
```bash
streamlit run ui_streamlit.py
```
- Enter your tweet in the text area.
- Click on "Analyze Sentiment" to get the prediction.

- ![image](https://github.com/user-attachments/assets/6b564c47-8b7b-48ce-9ad1-919b7c141d32)


- ### FastAPI UI and API
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

- ![image](https://github.com/user-attachments/assets/3afce606-f2b9-4cf3-820b-faa999abc882)

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
├── train_model.py
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.x
- Libraries: scikit-learn, pandas, numpy, streamlit, fastapi, uvicorn, pickle


