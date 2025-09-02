import pandas as pd
import re
from googleapiclient.discovery import build

# ---------------- CONFIG ----------------
# Replace with your actual backend API key
YOUTUBE_API_KEY = 'AIzaSyBZq0vuacwksdqDQwxLzKT_MdaeM1cm0zs'

# ---------------- FUNCTIONS ----------------
def get_video_id(url):
    """Extract YouTube video ID from URL"""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_comments(video_url, max_results=50):
    """
    Fetch comments from YouTube video using API key
    Returns DataFrame: datetime, author, comment
    """
    video_id = get_video_id(video_url)
    if not video_id:
        print("Invalid video URL")
        return pd.DataFrame(columns=['datetime','author','comment'])
    
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=max_results
        )
        response = request.execute()
        
        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append([
                snippet['publishedAt'],
                snippet['authorDisplayName'],
                snippet['textDisplay']
            ])
    except Exception as e:
        print("Error fetching comments:", e)
        return pd.DataFrame(columns=['datetime','author','comment'])
    
    df = pd.DataFrame(comments, columns=['datetime','author','comment'])
    return df

# ---------------- TESTING ----------------
if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ").strip()
    df_youtube = fetch_comments(video_url)
    
    if df_youtube.empty:
        print("No comments fetched.")
    else:
        print("First 5 comments:")
        print(df_youtube.head())
        print(f"\nTotal comments fetched: {len(df_youtube)}")
