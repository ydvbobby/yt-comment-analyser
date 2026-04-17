
from fastapi import FastAPI, HTTPException
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mlflow
from io import BytesIO
from typing import List
import pandas as pd
import os
import re
from src.utils.text_utils import lemmatize_text

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for servers
import matplotlib.pyplot as plt


import os
from dotenv import load_dotenv

from tensorflow import keras


import boto3

s3 = boto3.client("s3")


#======================================================================================================================================================

load_dotenv()

class InputData(BaseModel):
    text: List[str]
    
class SentimentCounts(BaseModel):
    positive: int  
    neutral: int  
    negative: int

class YouTubeFetchRequest(BaseModel):
    video_id: str

class YouTubeCommentsResponse(BaseModel):
    comments: List[str]
    total_comments: int
    positive: int  
    neutral: int  
    negative: int

app = FastAPI()

# Allow your extension's origin:
origins = [
    "chrome-extension://bidebcjfcpllndgdmfgifiknmmlaehno"  # your extension ID
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow.set_tracking_uri(os.getenv("mlflow_tracking_uri"))

print("_____________________________________________________________________________")
print(mlflow.get_tracking_uri())

print("______________________________________________________________________________")

# Load model from Unity Catalog
model = mlflow.keras.load_model(model_uri = "models:/yt-comment-analyzer/production")

def preprocess(clean_text:pd.Series):
    clean_text = clean_text.apply(lambda x: x.lower())
    clean_text = clean_text.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    clean_text = clean_text.apply(lambda x: x.strip())
    clean_text = clean_text.apply(lambda x: lemmatize_text(x))
    return clean_text



#=============================================================================================================================================
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post('/predict')
def predict(data: InputData):
    
    
    comments = preprocess(pd.Series(data.text)).astype(str).to_numpy()
    


    preds = model.predict(comments)
    predictions = preds.argmax(axis=1)
    print(predictions)

    reverse_map = {0:-1,1:0,2:1}

    sentiments = [reverse_map[int(p)] for p in predictions]

    return {"predictions": sentiments}











@app.post("/pie-chart")
async def generate_pie_chart(counts: SentimentCounts):
    # Map to readable labels
    labels = ["Positive", "Neutral", "Negative"]
    sizes = [counts.positive, counts.neutral, counts.negative]

    if sum(sizes) == 0:
        raise HTTPException(status_code=400, detail="All counts are zero, cannot make pie chart.")

    # Optional: filter out zero classes so the pie looks better
    filtered_labels = []
    filtered_sizes = []
    for label, size in zip(labels, sizes):
        if size > 0:
            filtered_labels.append(label)
            filtered_sizes.append(size)

    # Create pie chart
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(
        filtered_sizes,
        labels=filtered_labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")  # Equal aspect ratio ensures the pie is circular.

    # Make background transparent (looks nicer over dark/light UIs)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")



@app.post("/fetch-youtube-comments")
async def fetch_youtube_comments(request: YouTubeFetchRequest):
    """
    Fetch comments from a YouTube video server-side.
    Replaces the client-side fetching logic in popup.js.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Dictionary with total_comments count and list of comment texts
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY not configured")
    
    all_comments = []
    page_token = ""
    max_comments = 500
    base_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    
    async with httpx.AsyncClient() as client:
        while len(all_comments) < max_comments:
            params = {
                "part": "snippet",
                "videoId": request.video_id,
                "maxResults": 100,
                "key": api_key
            }
            if page_token:
                params["pageToken"] = page_token
            
            try:
                response = await client.get(base_url, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=response.status_code if hasattr(e, 'response') and e.response else 500,
                    detail=f"YouTube API error: {str(e)}"
                )
            
            items = data.get("items", [])
            if not items:
                break
            
            for item in items:
                comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                all_comments.append(comment_text)
                
                if len(all_comments) >= max_comments:
                    break
            
            next_page_token = data.get("nextPageToken")
            if not next_page_token or len(all_comments) >= max_comments:
                break
            
            page_token = next_page_token
    
    return {
        "total_comments": len(all_comments),
        "comments": all_comments
    }

