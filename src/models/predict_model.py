import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, confusion_matrix

model = pickle.load(open('models/model.pkl', 'rb'))

test_data = pd.read_csv('data/interim/tfidf_test_data.csv')

x_test = test_data.drop(['labels'],axis=1)
y_test = test_data['labels']

predictions = model.predict(x_test)

classification_report = classification_report(predictions, y_test)
confusion_matrix  = confusion_matrix(predictions, y_test)

metrics_dict = {
    'classification_report': classification_report
}

with open('metrics.json','w') as file:
    json.dump(metrics_dict, file, indent=4)



