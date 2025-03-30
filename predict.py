import pickle
import sys

# Load saved models
with open("models/fake_review_nb.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Get user input from command line
review = sys.argv[1]
review_vectorized = vectorizer.transform([review])
prediction = model.predict(review_vectorized)

print("Prediction:", "🛑 Fake Review" if prediction[0] == 1 else "✅ Genuine Review")
