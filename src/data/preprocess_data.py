import pandas as pd
import os
import re


from src.utils import lemmatize_text




train_data = pd.read_csv('data/raw/raw_train.csv')
train_data.dropna(inplace=True)
test_data = pd.read_csv('data/raw/raw_test.csv')
test_data.dropna(inplace=True)



def preprocess(clean_text:pd.Series):
    clean_text = clean_text.apply(lambda x: x.lower())
    clean_text = clean_text.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    clean_text = clean_text.apply(lambda x: x.strip())
    clean_text = clean_text.apply(lambda x: lemmatize_text(x))
    return clean_text

train_data['clean_comment'] = preprocess(train_data['clean_comment'])


test_data['clean_comment'] = preprocess(test_data['clean_comment'])

print("_________________________________________________________________________")

text = "The striped bats are hanging on their feet for best results."
print(lemmatize_text(text))

print("_________________________________________________________________________")

data_path = os.path.join('data', 'processed')
os.makedirs(data_path,exist_ok=True)

train_data.to_csv(os.path.join(data_path, 'processed_train.csv'),index=False)
test_data.to_csv(os.path.join(data_path, 'processed_test.csv'),index=False)