import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")  # or your own dataset
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['review']
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF + Ensemble
tfidf = TfidfVectorizer(stop_words="english")
nb = MultinomialNB()
lr = LogisticRegression(max_iter=200)

ensemble = VotingClassifier(
    estimators=[('nb', nb), ('lr', lr)],
    voting='soft'
)

model = Pipeline([('tfidf', tfidf), ('ensemble', ensemble)])
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
