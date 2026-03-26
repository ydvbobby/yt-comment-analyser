# 📊 YouTube Comment Sentiment Analyser

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![DVC Pipeline](https://img.shields.io/badge/DVC-tracked-blueviolet)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end Machine Learning project to predict and analyze the sentiment of YouTube textual comments. This repository encompasses everything from raw data processing to machine learning model deployment via a web application.

---

## ✨ Features

- **Automated ML Pipeline:** End-to-end data processing and model training driven by DVC (Data Version Control).
- **Sentiment Prediction:** Accurately classifies YouTube comments into positive, negative, or neutral sentiments.
- **RESTful API Backend:** Modular backend serving the trained ML model for predictions.
- **Interactive Web Frontend:** Easy-to-use user interface to explore the results interactively.

---

## 🏗️ Architecture & Pipeline

### Machine Learning Pipeline (DVC tracked)
The core ML tasks are tracked and executed using `dvc.yaml`.
1. `make_datasets`: Downloads/extracts raw datasets and pre-splits components.
2. `preprocess_data`: Cleans and normalizes the textual comment data.
3. `train_model`: Trains the sentiment classifier and logs parameters/metrics.
4. `predict_model`: Validates the model weights against test data.

### Web Application
- **Backend:** Serves prediction endpoints via Python web frameworks (`backend/`).
- **Frontend:** Interactive UI to input user comments and visualize the overall sentiment distribution of a video (`frontend/`).

---

## 📂 Project Organization

```text
yt-comment-analyser/
├── backend/            <- API server and utilities for predictions (app.py)
├── frontend/           <- Web interface for the end-user (main.py)
├── src/                <- Core Python source code for data & models
│   ├── data/           <- Scripts to generate and preprocess data
│   ├── features/       <- Feature engineering scripts
│   ├── models/         <- Training and prediction scripts
│   └── visualization/  <- Exploratory visualizations
├── data/
│   ├── raw/            <- Original, immutable data dump
│   ├── interim/        <- Intermediate transformed data
│   └── processed/      <- Final, canonical data sets for modeling
├── notebooks/          <- Jupyter notebooks for EDA and experimentation
├── artifacts/          <- Contains serialized and trained ML models
├── dvc.yaml            <- DVC pipeline definitions
├── params.yaml         <- Configuration parameters for ML experiments
├── metrics.json        <- Evaluation metrics of the trained models
├── Makefile            <- Helper commands (e.g., `make data`, `make requirements`)
├── requirements.txt    <- Python dependencies
└── README.md           <- The top-level README for developers
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/yt-comment-analyser.git
cd yt-comment-analyser
```

### 2. Create the environment
You can use `make` to initialize everything:
```bash
make requirements
```
*Alternatively, using virtualenv directly:*
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull Data and Reproduce the ML Pipeline
Assuming you have access to the remote DVC storage:
```bash
dvc pull
dvc repro
```

---

## 🎯 Usage

**Using the Web Interface:**
1. Start the backend server:
   ```bash
   python backend/app.py
   ```
2. In a separate terminal, launch the frontend application:
   ```bash
   python frontend/main.py
   ```

*Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).*
