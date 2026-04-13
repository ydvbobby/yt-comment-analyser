import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix
from scipy import sparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
import sys
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()




model = load_model("artifacts/models/comment_classifier.keras")



test_data = pd.read_csv('data/processed/processed_test.csv')
test_data.dropna(inplace=True)
x_test = test_data['clean_comment'].astype(str).to_numpy()
y_test = test_data['category'].map({-1:0,0:1,1:2}).astype("int32").to_numpy()



mlflow.set_tracking_uri(os.getenv("mlflow_tracking_uri"))

mlflow.set_experiment("april_Experiment")

with mlflow.start_run():

    predictions = model.predict(x_test)
    predicted_targets = predictions.argmax(axis=1)

    report = classification_report(predicted_targets, y_test, output_dict=True)
    
    
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
        else:
            mlflow.log_metric(label, metrics)
    
    model_signature = infer_signature(x_test[:50], predictions[:50])
            
    mlflow.sklearn.log_model(model,artifact_path="model", signature=model_signature,registered_model_name="yt-comment-analyzer")
    


