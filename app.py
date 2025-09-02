import streamlit as st
import pandas as pd
from preprocessing import clean_text
from parsers.whatsapp_parser import parse_whatsapp
from parsers.youtube_parser import fetch_comments
from model import predict_with_neutral, train_model
from utils import plot_sentiment_distribution, generate_wordcloud, plot_sentiment_trend, save_csv, save_visualizations_as_pdf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Sentiment Analysis App")

# --------------------- Model retraining ---------------------
st.sidebar.subheader("Retrain Model")
dataset_file = st.sidebar.file_uploader("Upload dataset CSV", type="csv")
if st.sidebar.button("Retrain Model"):
    if dataset_file:
        train_model(dataset_file)
        st.success("Model retrained successfully!")
    else:
        st.warning("Upload a dataset to retrain.")

# --------------------- Single Review Prediction ---------------------
st.subheader("Single Review Prediction")
single_text = st.text_input("Enter text for sentiment prediction")
if st.button("Predict Sentiment"):
    if single_text:
        clean_single = clean_text(single_text)
        sentiment, scores = predict_with_neutral(clean_single)
        st.write(f"Predicted Sentiment: **{sentiment.upper()}**")
        st.write("Scores:", scores)

# --------------------- WhatsApp Chat Upload ---------------------
st.subheader("Upload WhatsApp Chat (.txt)")
uploaded_file = st.file_uploader("Choose WhatsApp chat file", type="txt")
if uploaded_file:
    df_whatsapp = parse_whatsapp(uploaded_file)
    df_whatsapp['clean_message'] = df_whatsapp['message'].apply(clean_text)
    df_whatsapp['sentiment'] = df_whatsapp['clean_message'].apply(lambda x: predict_with_neutral(x)[0])
    st.dataframe(df_whatsapp)

    # Visualizations
    fig_bar, fig_pie = plot_sentiment_distribution(df_whatsapp, 'sentiment')
    fig_wordcloud = generate_wordcloud(df_whatsapp['clean_message'])
    fig_trend = plot_sentiment_trend(df_whatsapp, 'datetime', 'sentiment')

    st.pyplot(fig_bar)
    st.pyplot(fig_pie)
    st.pyplot(fig_wordcloud)
    st.pyplot(fig_trend)

    if st.button("Download WhatsApp Predictions CSV"):
        save_csv(df_whatsapp)
        st.success("CSV downloaded!")

    if st.button("Export WhatsApp Visualizations as PDF"):
        save_visualizations_as_pdf([fig_bar, fig_pie, fig_wordcloud, fig_trend])
        st.success("PDF exported!")

# --------------------- YouTube Video Comments ---------------------
st.subheader("Analyze YouTube Comments")
video_url = st.text_input("Enter YouTube video URL")
if st.button("Fetch & Analyze Comments"):
    if video_url:
        df_youtube = fetch_comments(video_url)
        if not df_youtube.empty:
            df_youtube['clean_comment'] = df_youtube['comment'].apply(clean_text)
            df_youtube['sentiment'] = df_youtube['clean_comment'].apply(lambda x: predict_with_neutral(x)[0])
            st.dataframe(df_youtube)

            fig_bar, fig_pie = plot_sentiment_distribution(df_youtube, 'sentiment')
            fig_wordcloud = generate_wordcloud(df_youtube['clean_comment'])
            fig_trend = plot_sentiment_trend(df_youtube, 'datetime', 'sentiment')

            st.pyplot(fig_bar)
            st.pyplot(fig_pie)
            st.pyplot(fig_wordcloud)
            st.pyplot(fig_trend)

            if st.button("Download YouTube Predictions CSV"):
                save_csv(df_youtube)
                st.success("CSV downloaded!")

            if st.button("Export YouTube Visualizations as PDF"):
                save_visualizations_as_pdf([fig_bar, fig_pie, fig_wordcloud, fig_trend])
                st.success("PDF exported!")
        else:
            st.warning("No comments fetched. Check video URL or backend API key.")
