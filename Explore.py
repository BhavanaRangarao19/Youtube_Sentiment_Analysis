
#pip install vaderSentiment scikit-learn transformers torch
# ==============================
# 1. VADER Sentiment Analysis
# ==============================
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader_analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)
    return score['compound']  # between -1 (neg) and +1 (pos)


# ==============================
# 2. Naive Bayes + Logistic Regression
# ==============================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Example training data (you should replace with a real labeled dataset, e.g., YouTube comments)
X_train = [
    "I love this movie, it was awesome!",
    "Terrible plot and boring characters.",
    "The video was okay, not great but not bad.",
    "Absolutely fantastic experience!",
    "Worst thing I’ve ever watched."
]
y_train = [1, 0, 1, 1, 0]  # 1 = positive, 0 = negative

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Models
nb = MultinomialNB()
lr = LogisticRegression()

# Ensemble (Voting Classifier)
voting_clf = VotingClassifier(estimators=[('nb', nb), ('lr', lr)], voting='soft')
voting_clf.fit(X_train_vec, y_train)

def ml_sentiment(text):
    X_vec = vectorizer.transform([text])
    proba = voting_clf.predict_proba(X_vec)[0][1]  # probability of positive
    return proba*2 - 1  # scale to [-1,1] like VADER


# ==============================
# 3. BERT Sentiment Analysis
# ==============================
from transformers import pipeline

bert_pipeline = pipeline("sentiment-analysis")

def bert_sentiment(text):
    result = bert_pipeline(text)[0]
    label = result['label']
    score = result['score']
    if label.lower().startswith("neg"):
        return -score
    else:
        return score


# ==============================
# 4. Test All Approaches
# ==============================
texts = [
    "I didn’t like this video at all.",
    "The movie was sick!",
    "Yeah, that was just amazing 🙄",
    "Not bad, actually pretty good."
]

for t in texts:
    print(f"\nText: {t}")
    print("VADER:", vader_sentiment(t))
    print("Naive Bayes + LR:", ml_sentiment(t))
    print("BERT:", bert_sentiment(t))
