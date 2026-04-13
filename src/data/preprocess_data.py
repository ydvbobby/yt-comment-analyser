import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

import pandas as pd
import os
import re


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

train_data = pd.read_csv('data/raw/raw_train.csv')
train_data.dropna(inplace=True)
test_data = pd.read_csv('data/raw/raw_test.csv')
test_data.dropna(inplace=True)


lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)

    # Map NLTK POS to WordNet POS
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # default

    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    return " ".join(lemmas)


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
os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, 'processed_train.csv'),index=False)
test_data.to_csv(os.path.join(data_path, 'processed_test.csv'),index=False)