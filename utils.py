import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud
import pandas as pd

def plot_sentiment_distribution(df, column='sentiment'):
    """
    Bar chart and pie chart of sentiment counts
    Returns bar and pie figures
    """
    counts = df[column].value_counts()

        # Handle empty DataFrame
    if counts.empty:
        fig_bar, fig_pie = plt.subplots(), plt.subplots()
        return fig_bar, fig_pie

    # Map colors safely
    color_map = {
        'positive': 'green',
        'negative': 'red',
        'neutral': 'blue'
    }

    # Clean labels (strip spaces, lowercase)
    labels = [str(label).strip().lower() for label in counts.index]
    colors = [color_map.get(label, 'gray') for label in labels]

    # Bar chart
    fig_bar, ax_bar = plt.subplots()
    counts.plot(kind='bar', color=colors, ax=ax_bar)
    ax_bar.set_title("Bar Chart of Sentiment Distribution")
    ax_bar.set_xlabel("Sentiment")
    ax_bar.set_ylabel("Count")
    plt.close(fig_bar)

    # Pie chart
    fig_pie, ax_pie = plt.subplots()
    counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax_pie)
    ax_pie.set_title("Pie Chart of Sentiment Distribution")
    plt.close(fig_pie)
    
    return fig_bar, fig_pie



def generate_wordcloud(text_list):
    """
    Generate word cloud from list of texts
    Returns figure for PDF and Streamlit
    """
    text = " ".join(text_list)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.close(fig)
    return fig

def plot_sentiment_trend(df, datetime_col='datetime', sentiment_col='sentiment'):
    """
    Plots sentiment trend over time and returns figure
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    trend = df.groupby([pd.Grouper(key=datetime_col, freq='D'), sentiment_col]).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12,6))
    trend.plot(ax=ax, marker='o')
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Messages")
    ax.grid(True)
    plt.close(fig)
    return fig

def save_csv(df, filename="outputs/predictions.csv"):
    df.to_csv(filename, index=False)

def save_visualizations_as_pdf(figures, filename="outputs/visualizations.pdf"):
    """
    Save a list of matplotlib figures as a single PDF
    """
    with PdfPages(filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
        pdf.close()
