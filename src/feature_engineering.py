import pandas as pd
import os
import logging
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
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

def load_params(params_path:str)->dict:
    try:
        with open (params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters loaded from %s",params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found at %s",e)
        raise 
    except yaml.YAMLError as e:
        logger.error("YAML error %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error %s",e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """
    Function to load data from a given path and return it as a pandas DataFrame.

    Args:
        data_path (str): The path of the data to be loaded.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    """
    try:
        df = pd.read_csv(data_path)
        logger.info("Data loaded successfully.")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while loading data from {data_path}: {e}")
        raise

def apply_tfidf(train_data:pd.DataFrame, test_data:pd.DataFrame, max_features:int) ->pd.DataFrame:
    """
    Function to apply TF-IDF vectorization on the text data.

    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
        max_features (int): The maximum number of features for TF-IDF.

    Returns:
        tuple: Transformed training and testing data as sparse matrices.
    """
    try:
        logger.debug("Starting TF-IDF vectorization.")
        required_columns = {"text", "target"}
        for dataset_name, dataset in (
            ("training", train_data),
            ("test", test_data),
        ):
            missing_columns = required_columns.difference(dataset.columns)
            if missing_columns:
                raise ValueError(
                    f"{dataset_name} data is missing columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )

        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        
        
        # Empty preprocessed messages are written as NaN by read_csv; treat
        # them as documents with no terms so TF-IDF can still process them.
        X_train=train_data['text'].fillna("").astype(str)
        X_test=test_data['text'].fillna("").astype(str)
        y_train=train_data['target']
        y_test=test_data['target']
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        train_df=pd.DataFrame(X_train_tfidf.toarray())
        train_df['target']=y_train
        test_df=pd.DataFrame(X_test_tfidf.toarray())
        test_df['target']=y_test
        logger.debug("TF-IDF vectorization completed.")


        return train_df, test_df
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
        """
        Function to save the DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            path (str): The path where the DataFrame will be saved.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            logger.debug(f"Data saved successfully to {path}.")
        except Exception as e:
            logger.error(f"Error saving data to {path}: {e}")
            raise


def main():
    #max_features = 50
    params=load_params(params_path='params.yaml')
    max_features=params['feature_engineering']['max_features']

    try:
        train_data = load_data("data/interim_data/train_preprocessed.csv")
        test_data = load_data("data/interim_data/test_preprocessed.csv")

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed_data", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed_data", "test_tfidf.csv"))
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()