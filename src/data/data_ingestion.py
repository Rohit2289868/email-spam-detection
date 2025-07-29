import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

handler = logging.StreamHandler()
handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file"""

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameter retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not Found at: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML Error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected Error: %s", e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug("DataFrame loaded from %s", data_path)
        return df
    except FileNotFoundError:
        logger.error("File Not Found at %s: ", data_path)
        raise
    except Exception as e:
        logger.error("Unexpected Error: %s", e)
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    logger.debug("data cleaning is done")
    return df

def shuffle_data(df: pd.DataFrame, random_state: int, frac: int) -> pd.DataFrame:
    df = df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    logger.debug("shuffling of data is done")
    return df

def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> tuple:

    train_data, test_data= train_test_split(df, test_size=test_size, random_state=random_state)
    logger.debug("data splitting is done")
    return train_data, test_data

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_path: str) -> None:
    try: 
        # Ensure the directory exists
        save_path.mkdir(parents=True, exist_ok=True)

        # save data
        train_data.to_csv(save_path.joinpath("train.csv"), index=False)
        test_data.to_csv(save_path.joinpath("test.csv"), index=False)

        logger.debug("Train and Test data is saved at: %s", save_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise



def main():
    try:
        curr_dir = Path(__file__)
        home_dir = curr_dir.parent.parent.parent
        params_path = home_dir/"params.yaml"
        data_path = home_dir.joinpath("data", "raw", "emails.csv")
        save_data_path = home_dir.joinpath("data", "external")

        # load parameters
        params = load_params(params_path)
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        frac = params['data_ingestion']['frac']

        # load data
        df = load_data(data_path)

        # data cleaning
        df = clean_data(df)

        # shuffle data
        df = shuffle_data(df, random_state, frac)

        # split data
        train_data, test_data = split_data(df, test_size, random_state)

        # save data
        save_data(train_data, test_data, save_data_path)
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        raise
if __name__ == '__main__':
    main()