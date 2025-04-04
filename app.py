from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load trained model and vectorizer
with open("models/fake_review_nb.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
@app.route('/')
def home():
    return "Fake Product Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON input
    review = data.get("review", "")

    if not review:
        return jsonify({"error": "Review text is required"}), 400
    
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)

    result = "🛑 Fake Review" if prediction[0] == 1 else "✅ Genuine Review"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
