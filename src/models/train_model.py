from xgboost import XGBClassifier
import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import yaml


processed_train = pd.read_csv('data/processed/processed_train.csv')
processed_train.fillna("",inplace=True)

x_train = processed_train['clean_comment']
y_train = processed_train['category'].map({-1:0,0:1,1:2})

max_features = yaml.safe_load(open('params.yaml','r'))['train_model']['max_features']
vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1,2))

model = XGBClassifier(
        objective="multi:softprob",   # multiclass
        num_class=3,                  # number of classes
        max_depth=6,
        n_estimators=500,
        learning_rate=0.1,
        tree_method="hist")

pipeline = Pipeline([
    ('vectorizer',vectorizer),
    ('model',model)
])

pipeline.fit(x_train,y_train)

os.makedirs("artifacts/models",exist_ok=True)

pickle.dump(pipeline, open('artifacts/models/pipeline.pkl', "wb"))


