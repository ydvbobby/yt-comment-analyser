import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

#============================================================================================================
logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('error.log')
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
#================================================================================================================

def load_params(file):
    try:
        test_size = yaml.safe_load(open(file,'r'))['make_dataset']['test_size']
    except Exception as e:
        logger.error(e)
        raise
    return test_size

def load_data(path):
    try:
       df = pd.read_csv(path)
    except Exception as e:
        logger.error(e)
        raise
    return df

def save_data(data_path, train_data,test_data):
    os.makedirs(data_path,exist_ok=True)

    train_data.to_csv(os.path.join(data_path, "raw_train.csv"),index=False)
    test_data.to_csv(os.path.join(data_path, 'raw_test.csv'),index=False)
    
def main():
    data = load_data('data/external/Reddit_Data.csv')
    data.dropna(inplace=True)
    
    logger.info(f'data loaded succesfully with shape {data.shape}')

    test_size = load_params('params.yaml')
    
    train_data, test_data = train_test_split(data, test_size=test_size,random_state=42)
    
    data_path = os.path.join('data', 'raw')
    save_data(data_path,train_data,test_data)
    logger.info('raw data saved succesfully')

if __name__ == '__main__' :
    main()