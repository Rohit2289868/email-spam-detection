import pandas as pd
import numpy as np
from pathlib import Path
import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

handler = logging.StreamHandler()
handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

def load_data(data_path: str) -> tuple:
    train_path = data_path.joinpath("train.csv")
    test_path = data_path.joinpath("test.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    logger.debug("Train and Test data loaded from: %s", data_path)
    return train_data, test_data

def preprocess(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove links (normal + obfuscated)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/\s*\S+', '', text, flags=re.IGNORECASE)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    words = text.split()
    # Remove stopwords + Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_data_path: str) -> None:

    try:
        # Ensure the directory exist
        save_data_path.mkdir(parents=True, exist_ok=True)

        # save data
        train_data.to_csv(save_data_path.joinpath("train_processed_data.csv"), index=False)
        test_data.to_csv(save_data_path.joinpath("test_processed_data.csv"), index=False)
        logger.debug("Train_processed and Test_processed data is saved at: %s", save_data_path)

    except Exception as e:
        logger.error("Unexpected error occured while saving the data: %s", e)
        raise

def main():

    try:
        curr_dir = Path(__file__)
        home_dir = curr_dir.parent.parent.parent
        data_path = home_dir.joinpath("data", "external")
        save_data_path = home_dir.joinpath("data", "interim")

        train_data, test_data = load_data(data_path)

        train_data['text'] = train_data['text'].apply(preprocess)
        logger.debug("Preprocessing on training data is done")

        test_data['text'] = test_data['text'].apply(preprocess)
        logger.debug("Preprocessing on testing data is done")

        save_data(train_data, test_data, save_data_path)

    except Exception as e:
        logger.error("Unexpected error occurred while doing data_preprocessing")
        raise

if __name__ == '__main__':
    main()





