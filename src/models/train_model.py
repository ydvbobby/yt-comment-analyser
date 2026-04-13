import os
import sys
os.environ["PYTHONUTF8"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


import pandas as pd


from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping




processed_train = pd.read_csv('data/processed/processed_train.csv')
processed_test = pd.read_csv('data/processed/processed_test.csv')

# processed_train.fillna("",inplace=True)
# processed_test.fillna("",inplace=True)

x_train = processed_train['clean_comment'].astype(str).to_numpy()
x_test = processed_test['clean_comment'].astype(str).to_numpy()

y_train = processed_train['category'].map({-1:0,0:1,1:2}).astype("int32").to_numpy()
y_test = processed_test['category'].map({-1:0,0:1,1:2}).astype("int32").to_numpy()



vectorizer = TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=400
)

vectorizer.adapt(x_train)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Bidirectional

from tensorflow.keras.layers import LSTM, GRU

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model = Sequential([
    Input(shape=(1,),dtype='string'),
    vectorizer,
    Embedding(10000, 128),
    Bidirectional(GRU(64)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64,activation='tanh'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stop])

os.makedirs('artifacts/models', exist_ok=True)
model.save('artifacts/models/comment_classifier.keras')