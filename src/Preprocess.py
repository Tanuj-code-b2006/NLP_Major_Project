import os
import re
import logging

import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("preprocess")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "preprocess.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Function to transform text data by converting to lowercase, removing punctuation,
    removing stopwords, and stemming the remaining words.

    Args:
        text (str): The input text to be transformed.

    Returns:
        str: The transformed text.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = nltk.stem.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def preprocess_data(df, text_column='text', target_column='target', label_encoder=None):
    """
    Function to preprocess the data by handling missing values, removing duplicates,
    encoding the target column, and transforming the text field.

    Args:
        df (pd.DataFrame): The input data to be preprocessed.
        text_column (str): The name of the column containing the text data.
        target_column (str): The name of the column containing the target data.
        label_encoder (LabelEncoder, optional): A fitted label encoder for target values.
    """
    try:
        logger.debug("Starting data preprocessing.")

        df = df.dropna(subset=[text_column, target_column]).copy()
        logger.debug("Dropped rows with missing values.")

        df = df.drop_duplicates()
        logger.debug("Dropped duplicate rows.")

        if label_encoder is None:
            label_encoder = LabelEncoder()
            df[target_column] = label_encoder.fit_transform(df[target_column])
        else:
            df[target_column] = label_encoder.transform(df[target_column])
        logger.debug("Encoded target column.")

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Transformed text data.")
        return df
    except KeyError as e:
        logger.error(f"Missing column error during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def main(text_column='text', target_column='target'):
    """
    Main function to execute the preprocessing steps.

    Args:
        text_column (str): The name of the column containing the text data.
        target_column (str): The name of the column containing the target data.
    """
    try:
        logger.debug("Starting main preprocessing function.")

        train_data = pd.read_csv("data/raw_data/train.csv")
        test_data = pd.read_csv("data/raw_data/test.csv")
        logger.debug("Loaded training and testing data.")

        le = LabelEncoder()
        le.fit(train_data[target_column])

        train_data = preprocess_data(train_data, text_column, target_column, le)
        test_data = preprocess_data(test_data, text_column, target_column, le)

        data_path = os.path.join("./data", "interim_data")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)
        logger.debug("Saved preprocessed data to data/interim_data directory.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing column error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in main preprocessing function: {e}")
        raise


if __name__ == "__main__":
    main()