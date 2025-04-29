from flask import Flask, request, jsonify
import joblib
import string
import re
from textblob import TextBlob
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Clean and preprocess incoming review
def preprocess_review(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    length = len(text)
    sentiment = TextBlob(text).sentiment.polarity
    tfidf_vector = vectorizer.transform([text])
    combined_features = np.hstack((tfidf_vector.toarray(), [[length, sentiment]]))
    return combined_features

@app.route('/')
def home():
    return "Fake Review Detection API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({"error": "No review provided"}), 400
    
    review = data['review']
    processed = preprocess_review(review)
    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed)
    
    # Get probability of predicted class
    label_index = list(model.classes_).index(prediction)
    confidence = round(proba[0][label_index], 2)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
