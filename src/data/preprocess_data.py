import pandas as pd
import os

train_data = pd.read_csv('data/raw/raw_train.csv')
test_data = pd.read_csv('data/raw/raw_test.csv')

def prerocess(df):
    
    #remove empty comments
    df = df[~(df['clean_comment'].str.strip() == '')]
    
    #remove null values
    df = df.dropna()
    
    #remove duplicated rows
    df = df[~(df.duplicated())]
    
    #lowercase every row data
    df['clean_comment'] = df['clean_comment'].str.lower()
    
    #remove spaces in begining and end of sentences
    df['clean_comment'] = df['clean_comment'].str.strip()
    
    #remove \n 
    df['clean_comment'] = df['clean_comment'].apply(lambda x : x.replace("\n", " "))
    
    #remove stopwords
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    to_remove_stopWords = stop_words - {'not','but','however','no','yet'}
    df['clean_comment'] = df['clean_comment'].apply(lambda x : " ".join([word for word in x.split(" ") if word.lower() not in to_remove_stopWords]))
    
    #remove special characters
    import re
    df['clean_comment'] = df['clean_comment'].apply(lambda x : re.sub(r'[^a-zA-Z0-9/s?,.!]'," ", str(x)))
    
    # Lemitization 
    import nltk
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemitizer  = WordNetLemmatizer()
    df['clean_comment'] = df['clean_comment'].apply(lambda x : " ".join([lemitizer.lemmatize(word) for word in x.split()]))
    
    return df

processed_train = prerocess(train_data)
processed_test  = prerocess(test_data)

data_path = os.path.join('data', 'processed')
os.makedirs(data_path)

processed_train.to_csv(os.path.join(data_path, 'processed_train.csv'),index=False)
processed_test.to_csv(os.path.join(data_path, 'processed_test.csv'),index=False)