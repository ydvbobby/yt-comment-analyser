# YouTube Comment Sentiment Analyser

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![DVC Pipeline](https://img.shields.io/badge/DVC-tracked-blueviolet)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning project that analyzes the sentiment of YouTube comments. The repository covers dataset preparation, model training, experiment tracking, backend inference APIs, and a Chrome extension UI for running sentiment analysis on YouTube videos.

## Overview

This project is built around three connected parts:

- `src/`: data preparation, preprocessing, training, evaluation, and visualization code.
- `backend/`: a FastAPI service that fetches comments, runs sentiment inference, and generates chart images.
- `frontend/`: a Chrome extension popup that works on YouTube video pages and displays sentiment results.

The current application flow is:

1. Open a YouTube video.
2. Launch the Chrome extension popup.
3. The extension sends the video ID to the backend.
4. The backend fetches comments with the YouTube Data API.
5. The backend predicts sentiment for each comment using the trained model.
6. The extension displays counts, percentages, grouped comments, and a pie chart.

## Features

- DVC-tracked ML pipeline for dataset preparation and training
- Text preprocessing for YouTube comments
- Sentiment classification into negative, neutral, and positive classes
- FastAPI backend for prediction and YouTube comment retrieval
- Pie chart generation for sentiment distribution
- Chrome extension frontend for analyzing the active YouTube video
- MLflow integration for experiment tracking and model registry

## Repository Structure

```text
yt-comment-analyser/
|-- backend/               FastAPI app and API dependencies
|-- frontend/              Chrome extension popup UI
|-- src/
|   |-- data/              Dataset creation and preprocessing scripts
|   |-- models/            Model training and prediction scripts
|   `-- visualization/     Visualization helpers
|-- artifacts/             Serialized model artifacts
|-- data/                  DVC-managed datasets
|-- docs/                  Project documentation
|-- notebooks/             EDA and experiments
|-- scripts/tests/         Basic test coverage
|-- dvc.yaml               DVC pipeline definition
|-- params.yaml            Training parameters
|-- metrics.json           Evaluation metrics snapshot
|-- requirements.txt       Python dependencies
`-- README.md
```

## Machine Learning Pipeline

The DVC pipeline is defined in `dvc.yaml` and currently contains these stages:

1. `make_datasets`
   Creates the raw train/test-ready files from the external dataset source.
2. `preprocess_data`
   Cleans and normalizes YouTube comment text into processed datasets.
3. `train_model`
   Trains a `CountVectorizer + XGBClassifier` pipeline and stores it in `artifacts/models/pipeline.pkl`.
4. `predict_model`
   Evaluates the model on processed test data and logs metrics and the model to MLflow.

## Current Model Setup

- Vectorization: `CountVectorizer`
- Classifier: `XGBClassifier`
- Label mapping used internally:
  - `-1` -> negative
  - `0` -> neutral
  - `1` -> positive
- Experiment/model tracking: MLflow

The checked-in `metrics.json` reports roughly:

- Accuracy: `0.77`
- Weighted F1: `0.78`

## Backend API

The backend entry point is `backend/main.py`.

Implemented endpoints:

- `GET /health`
  Returns a simple health response.
- `POST /predict`
  Accepts a list of comment strings and returns sentiment predictions.
- `POST /pie-chart`
  Accepts sentiment counts and returns a PNG pie chart.
- `POST /fetch-youtube-comments`
  Accepts a YouTube `video_id` and returns fetched comments.

The backend currently:

- loads environment variables from `.env`
- retrieves the production model from MLflow using `models:/yt-comment-analyzer/Production`
- downloads required NLTK resources at startup

## Frontend

The frontend is a Chrome extension located in `frontend/`.

Key files:

- `frontend/manifest.json`
- `frontend/popup.html`
- `frontend/js/popup.js`
- `frontend/css/styles.css`

The extension:

- detects the active YouTube tab
- extracts the current video ID
- calls the backend for comment retrieval and sentiment prediction
- displays percentage summaries and comment tabs for each sentiment

At the moment, `frontend/js/popup.js` points to a deployed backend load balancer URL. If you want to run everything locally, update that base URL to your local backend address.

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 3. Configure environment variables

Create a `.env` file with the values required by the backend and MLflow setup. Based on the current code, the important variables are:

```env
mlflow_tracking_uri=...
YOUTUBE_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
```

Use the exact names expected by your runtime and deployment setup.

### 4. Pull data if using DVC

```bash
dvc pull
```

### 5. Reproduce the ML pipeline

```bash
dvc repro
```

## Running the Project

### Run the backend locally

```bash
python backend/main.py
```

The API will start on `http://127.0.0.1:8000` by default.

### Load the Chrome extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable Developer Mode
3. Click "Load unpacked"
4. Select the `frontend/` directory

If you are using the backend locally, update the `base_url` in `frontend/js/popup.js` before loading the extension.

## Testing

The repository includes tests under `scripts/tests/`, including checks for:

- model loading
- model performance
- backend API behavior

Run tests with:

```bash
pytest scripts/tests
```

## Notes

- The repository follows a Cookiecutter Data Science-style structure.
- Some older README references to `frontend/main.py` or `backend/app.py` do not match the current codebase; `backend/main.py` is the active backend entry point and the frontend is a Chrome extension.
- The backend currently allows CORS for a specific Chrome extension ID.

## License

MIT
