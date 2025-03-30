import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (only needed once)
nltk.download("stopwords")

# Initialize stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# 🔹 Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Remove stopwords and apply stemming
    return text

print("🔹 Loading dataset...")
df = pd.read_csv("data/reviews.csv")  # Make sure reviews.csv exists

print("🔹 Preprocessing text...")
df["cleaned_text"] = df["reviews.text"].apply(preprocess_text)


# 🔹 Convert labels (Assuming 1 = Fake, 0 = Genuine)
df["label"] = df["reviews.doRecommend"].fillna(0).astype(int)

# 🔹 Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["label"], test_size=0.2, random_state=42)

print("🔹 Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("🔹 Training Naïve Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

print("🔹 Training k-NN model...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_vec, y_train)

print("🔹 Evaluating models...")

# Evaluate Naïve Bayes
nb_predictions = nb_model.predict(X_test_vec)
print("\n📊 Naïve Bayes Performance:")
print(classification_report(y_test, nb_predictions))

# Evaluate k-NN
knn_predictions = knn_model.predict(X_test_vec)
print("\n📊 k-NN Performance:")
print(classification_report(y_test, knn_predictions))

# 🔹 Save models and vectorizer
print("🔹 Saving models...")
with open("models/fake_review_nb.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("models/fake_review_knn.pkl", "wb") as f:
    pickle.dump(knn_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model training complete! Models saved in 'models/' directory.")
