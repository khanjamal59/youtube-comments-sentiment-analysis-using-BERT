from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_comments(video_id, max_comments=100):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

