import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import joblib

def train_model(dataset_path="data/IMDB Dataset.csv"):
    """
    Train ensemble model and save it as model.pkl
    """
    df = pd.read_csv(dataset_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(stop_words="english")
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=200)
    ensemble = VotingClassifier([('nb', nb), ('lr', lr)], voting='soft')
    model = Pipeline([('tfidf', tfidf), ('ensemble', ensemble)])
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    return model, X_test, y_test

def load_model():
    return joblib.load("model.pkl")

def predict_with_neutral(text, threshold=0.3):
    """
    Predict sentiment including neutral class
    """
    model = load_model()
    probs = model.predict_proba([text])[0]
    neg_prob, pos_prob = probs[0], probs[1]

    if abs(pos_prob - neg_prob) < threshold:
        return "neutral", {"positive": round(pos_prob,3), "negative": round(neg_prob,3), "neutral": round(1 - abs(pos_prob - neg_prob),3)}
    else:
        sentiment = "positive" if pos_prob > neg_prob else "negative"
        return sentiment, {"positive": round(pos_prob,3), "negative": round(neg_prob,3)}
