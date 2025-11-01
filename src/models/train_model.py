from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

vectorized_train_data = pd.read_csv('data/interim/vectorized_train_data.csv')
vectorized_test_data = pd.read_csv('data/interim/vectorized_test_data.csv')

x_train = vectorized_train_data.drop(['labels'],axis=1)
x_test = vectorized_test_data.drop(['labels'],axis=1)
y_train = vectorized_train_data['labels']
y_test = vectorized_test_data['labels']



model = LogisticRegression()

model.fit(x_train, y_train)
os.makedirs("models", exist_ok=True)

pickle.dump(model, open('models/model.pkl', "wb"))


