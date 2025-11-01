import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import yaml

max_features = yaml.safe_load(open('params.yaml','r'))['build_features']['max_features']

processed_train = pd.read_csv('data/processed/processed_train.csv')
processed_test = pd.read_csv('data/processed/processed_test.csv')

processed_test.fillna("",inplace=True)
processed_train.fillna('', inplace=True)

x_train = processed_train['clean_comment'].values
x_test = processed_test['clean_comment'].values

y_train = processed_train['category'].values
y_test = processed_test['category'].values

vectorizer = TfidfVectorizer(max_features=max_features)

x_train_bow = vectorizer.fit_transform(x_train)
x_test_bow = vectorizer.transform(x_test)

train_bow = pd.DataFrame(data=x_train_bow.toarray())
train_bow['labels'] = y_train

test_bow = pd.DataFrame(data=x_test_bow.toarray())
test_bow['labels'] = y_test

data_path = os.path.join('data','interim')
os.makedirs(data_path)

train_bow.to_csv(os.path.join(data_path, 'tfidf_train_data.csv'))
test_bow.to_csv(os.path.join(data_path, 'tfidf_test_data.csv'))
