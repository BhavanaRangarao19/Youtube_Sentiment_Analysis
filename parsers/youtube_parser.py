import pandas as pd
import re
from googleapiclient.discovery import build
from config import YOUTUBE_API_KEY  # Backend-stored API key

def get_video_id(url):
    """Extract YouTube video ID from URL"""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_comments(video_url, max_results=100):
    """
    Fetch comments from YouTube video using backend API key
    Returns DataFrame: datetime, author, comment
    """
    video_id = get_video_id(video_url)
    if not video_id:
        return pd.DataFrame(columns=['datetime','author','comment'])
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=max_results
    )
    response = request.execute()
    for item in response['items']:
        snippet = item['snippet']['topLevelComment']['snippet']
        comments.append([snippet['publishedAt'], snippet['authorDisplayName'], snippet['textDisplay']])
    return pd.DataFrame(comments, columns=['datetime', 'author', 'comment'])
