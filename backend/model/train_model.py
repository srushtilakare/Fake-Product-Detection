# train_model.py
import pandas as pd
import numpy as np
import string
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from textblob import TextBlob

# 1. Load Sample Dataset (You can replace this with another CSV)
df = pd.read_csv('https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv')
df = df[['label', 'tweet']]
df.columns = ['label', 'review']
df['label'] = df['label'].map({0: 'genuine', 1: 'fake'})

# 2. Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

df['clean_review'] = df['review'].apply(clean_text)

# 3. Add Text Features
df['length'] = df['clean_review'].apply(len)
df['sentiment'] = df['clean_review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 4. Vectorize Text (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['clean_review'])

# Combine Text + Numerical Features
X = np.hstack((X_text.toarray(), df[['length', 'sentiment']].values))
y = df['label']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Base Models
nb = MultinomialNB()
knn = KNeighborsClassifier(n_neighbors=5)

# 7. Ensemble Voting Classifier
model = VotingClassifier(estimators=[
    ('nb', nb),
    ('knn', knn)
], voting='hard')

model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Save Model and Vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and Vectorizer saved.")
