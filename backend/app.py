
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from io import BytesIO
import pickle
from typing import List
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for servers
import matplotlib.pyplot as plt

from backend.utilities import lemmatize_text

import numpy as np
import os
from dotenv import load_dotenv

#======================================================================================================================================================

load_dotenv()

class InputData(BaseModel):
    text: List[str]
    
class SentimentCounts(BaseModel):
    positive: int  
    neutral: int  
    negative: int

app = FastAPI()

# Allow your extension's origin:
origins = [
    "chrome-extension://ddonnkogojeammnmjkhonjookeelpjok"  # your extension ID
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
model = mlflow.sklearn.load_model(model_uri = "models:/yt-comment-analyzer/Production")

#=============================================================================================================================================
def prerocess(comment:str):
    
   
    #lowercase every row data
    comment = comment.lower()
    

    #remove stopwords
    import nltk
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    to_remove_stopWords = stop_words - {'not','but','however','no','yet'}
    comment = " ".join([word for word in comment.split(" ") if word.lower() not in to_remove_stopWords])

    # Lemitization 
    import nltk
    from nltk.stem import WordNetLemmatizer
    lemitizer  = WordNetLemmatizer()
    comment = " ".join([lemitizer.lemmatize(word) for word in comment.split()])
    
    return comment
#=============================================================================================================================================


@app.post('/predict')
def predict(data: InputData):
    
    processed_comments = [prerocess(comment) for comment in data.text]
    


    predictions = model.predict(processed_comments)
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

