import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
from sklearn.ensemble import RandomForestClassifier
import pickle

logger = logging.getLogger('Model Building')
logger.setLevel('DEBUG')

handler = logging.StreamHandler()
handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

def load_params(params_path: Path) -> dict:
    
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logger.debug("parameter are loaded from: %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File Not Found at: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the parameters: %s", e)
        raise

def load_data(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    logger.debug("Train data is loaded from: %s", data_path)
    return data

def train_model(train_data: pd.DataFrame, n_estimators, min_samples_split) -> RandomForestClassifier:
    X_train = train_data.drop(columns=['spam'], axis=1)
    y_train = train_data['spam']
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    logger.debug("Model training is done")

    return model

def save_model(model: RandomForestClassifier, save_model_path: Path) -> None:
    
    # Ensure the directory exists
    save_model_path.mkdir(parents=True, exist_ok=True)
    save_model_file = save_model_path.joinpath('model.pkl')

    with open(save_model_file, 'wb') as file:
        pickle.dump(model, file)

    logger.debug("model is saved at: %s", save_model_file)


def main():
    try:
        curr_dir = Path(__file__)
        home_dir = curr_dir.parent.parent.parent
        params_path = home_dir.joinpath("params.yaml")
        data_path = home_dir.joinpath("data", "processed", "final_train_df.csv")
        save_model_path = home_dir.joinpath("models")

        params = load_params(params_path)['model_building']
        data = load_data(data_path)

        model = train_model(data, params['n_estimators'], params['min_samples_split'])
        save_model(model, save_model_path)

        logger.debug("model training is done.")

    except Exception as e:
        logger.error("Unexpected error occurred while training the model: %s", e)


if __name__ == '__main__':
    main()