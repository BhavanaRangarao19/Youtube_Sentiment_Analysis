# 🎬💬 Sentiment Analysis Web App

An interactive **Streamlit web app** for analyzing sentiment from **YouTube comments** and **WhatsApp chats**.  
Supports multiple sentiment analysis approaches (VADER, Naive Bayes, Logistic Regression, Voting Classifier, BERT).  
Also integrates **Gemini API** for summarization of comments/chats.

---

## ✨ Features
- 🔹 Fetch **YouTube video comments** using YouTube Data API
- 🔹 Upload **WhatsApp chat export (.txt)** for analysis
- 🔹 Sentiment analysis with:
  - VADER (lexicon-based)
  - Naive Bayes + Logistic Regression (ML models)
  - Voting Classifier (ensemble)
  - BERT (contextual deep learning)
- 🔹 Visualizations: Bar charts, scatter plots, word clouds
- 🔹 **Summarization of comments/chats** with Gemini API
- 🔹 Unified pipeline: same models used for YouTube + WhatsApp data

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Create & activate virtual environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   - Get a **YouTube Data API key** → Google Developers Console
   - Get a **Gemini API key** → Google AI Studio
   - Add them in a `.env` file:
     ```ini
     YOUTUBE_API_KEY=your_api_key_here
     GEMINI_API_KEY=your_api_key_here
     ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🎯 Usage

- **YouTube Mode**:
  - Paste a video URL
  - Fetch comments, analyze sentiment, generate visualizations, and summarize with Gemini

- **WhatsApp Mode**:
  - Export chat from WhatsApp → upload `.txt` file
  - App analyzes sentiment of messages and provides chat summary

---

## 📊 Tech Stack
- **Frontend/Framework:** Streamlit
- **APIs:** YouTube Data API, Gemini API
- **NLP:** VADER, scikit-learn (Naive Bayes, Logistic Regression), Voting Classifier, HuggingFace BERT
- **Visualization:** Matplotlib, WordCloud

---

## 🚀 Future Enhancements
- Add toxicity & emotion classification
- Multi-language support
- Real-time dashboard with time-based sentiment trends
