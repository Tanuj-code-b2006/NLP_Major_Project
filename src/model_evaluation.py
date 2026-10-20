import pandas as pd
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import json
import os
import numpy as np
import  logging


logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)  
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(logs_dir, "feature_engineering.log"))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path:str):
    try:
        logger.debug("Loading model from location %s",file_path)
        with open (file_path,'rb') as file:
            model=pickle.load(file)
            logger.debug("Model loaded from %s",file_path)
            return model
    except FileNotFoundError:
        logger.error("File not found at %s",file_path)
        raise
    except Exception as e:
        logger.error("Unexpected Error occured %s",e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    try:
        logger.debug("Loading Data from %s",file_path)
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s",file_path)
        return df
    except pd.errors.ParserError as p:
        logger.error("Failed to parse the CSV file %s",p)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data")
        raise 

def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,-1]
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)

        metrics_dict={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'auc':auc
    }
        logger.debug("Model evaluation done ")
        return metrics_dict
    except Exception as e:
        logger.error("Unexpected error occured %s",e)


def save_metric(metrics:dict, file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug("Model saved to location %s",file_path)

    except Exception as e:
        logger.error("Error occured while saving metric %s",e)
        raise


def main():
    try:
        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/processed_data/test_tfidf.csv')

        X_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values

        metrics=evaluate_model(clf,X_test,y_test)
        save_metric(metrics,'reports/metrics.json')

    except Exception as e:
        logger.error("Failed to complete the model Evaluation process %s",e)
        print(f"Error: {e}")
        

if __name__=='__main__' :
    main()


