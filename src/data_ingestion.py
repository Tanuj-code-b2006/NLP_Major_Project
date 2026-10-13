import pandas as pd
import os
import logging

from sklearn.model_selection import train_test_split 

log_dir="logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "data_ingestion.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)->pd.DataFrame:
    """
    Function to load data from a given URL and return it as a pandas DataFrame.

    Args:
        data_url (str): The URL of the data to be loaded.
        
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        logger.debug(f"Attempting to load data from {data_url}")
        data = pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while loading data from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to preprocess the data by handling missing values and duplicates.

    Args:
        df (pd.DataFrame): The input data to be preprocessed.


    Returns:
        pd.DataFrame: The preprocessed data.
    """
    try:
        logger.debug("Starting data preprocessing.")
        # Handle missing values
        df = df.dropna()  # Drop rows with missing values
        logger.debug("Missing values handled by dropping rows with NaN values.")

        # Handle duplicates
        df = df.drop_duplicates()
        logger.debug("Duplicates handled by dropping duplicate rows.")

        df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1, inplace=True)  
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)  # Rename columns
        logger.debug("Data preprocessing completed.")
        return df
    except KeyError as e:
        logger.error(f"Missing column error during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Function to save the preprocessed data to a specified file path.

    Args:
        train_data (pd.DataFrame): The preprocessed training data to be saved.
        test_data (pd.DataFrame): The preprocessed testing data to be saved.
        data_path (str): The file path where the data will be saved.
    """
    try:
        logger.debug(f"Attempting to save data to {data_path}")
        raw_data_path = os.path.join(data_path, "raw_data")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug(f"Data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Error saving data to {data_path}: {e}")
        raise


def main():
    """
    Main function to orchestrate the data ingestion process.
    """
    try:
        data_path ="https://raw.githubusercontent.com/Tanuj-code-b2006/Datasets/refs/heads/main/spam.csv"
        test_size = 0.2

        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()