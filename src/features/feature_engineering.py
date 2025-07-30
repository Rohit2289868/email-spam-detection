import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer
import pickle

logger = logging.getLogger('Feature Engineering')
logger.setLevel('DEBUG')

handler = logging.StreamHandler()
handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug("parameters are loaded from: %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File Not Found at: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("yaml error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the parameters: %s", e)
        raise

def load_data(data_path: str, ) -> pd.DataFrame:
    
    data = pd.read_csv(data_path)
    logger.debug("Data is loaded from: %s", data_path)

    return data

def apply_bow(train_data: pd.DataFrame, test_data:pd.DataFrame, save_vectorizer_path: Path, max_features: int) -> tuple:
    vectorizer = CountVectorizer(max_features=max_features)
    
    X_train = train_data['text']
    X_test = test_data['text']
    y_train = train_data['spam']
    y_test = test_data['spam']

    x_train_bow = vectorizer.fit_transform(X_train)
    x_test_bow = vectorizer.transform(X_test)

    train_df = pd.DataFrame(x_train_bow.toarray())
    train_df['spam'] = y_train
    test_df = pd.DataFrame(x_test_bow.toarray())
    test_df['spam'] = y_test

    logger.debug("BoW is applied on the data")
    # Ensure the directory exists
    save_vectorizer_file = save_vectorizer_path.joinpath('vectorizer.pkl')
    save_vectorizer_file.parent.mkdir(parents=True, exist_ok=True)

    with open(save_vectorizer_file, 'wb') as file:
        pickle.dump(vectorizer, file)

    logger.debug("vectorizer is saved at: %s", save_vectorizer_file)

    return train_df, test_df

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_data_path: str) -> None:
    # Ensure the Directory exists
    save_data_path.mkdir(parents=True, exist_ok=True)
    train_data_path = save_data_path.joinpath("final_train_df.csv")
    test_data_path = save_data_path.joinpath("final_test_df.csv")

    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    logger.debug("Train and Test data is saved at: %s", save_data_path)


def main():
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_path = home_dir.joinpath("params.yaml")
    train_data_path = home_dir.joinpath("data", "interim", "train_processed_data.csv")
    test_data_path = home_dir.joinpath("data", "interim", "test_processed_data.csv")
    save_data_path = home_dir.joinpath("data", "processed")
    save_vectorizer_path = home_dir.joinpath("models")

    params = load_params(params_path)
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    max_features = params['feature_engineering']['max_features']

    final_train_data, final_test_data = apply_bow(train_data, test_data, save_vectorizer_path, max_features)

    save_data(final_train_data, final_test_data, save_data_path)
    logger.debug("Feature engineering part is done successfully")

if __name__ == '__main__':
    main()
