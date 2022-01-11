import pandas as pd
import logging
import pytest
import os
import yaml
from pipeline.ml.data import load_data, process_data
from pipeline.ml.model import train_model
from sklearn.model_selection import train_test_split


with open("config.yml", "r") as config:
    cfg = yaml.safe_load(config)


logging.basicConfig(
    filename='./pipeline.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def data():
    """ 
    Load data for testing.
    """
    PATH = cfg["main"]["data_path"]

    assert os.path.exists(
        os.path.join(os.getcwd(), PATH)
    ), f"Path doesn't exist. Please, verify."

    df = load_data(PATH)

    return df


def test_data(data):
    """ Basic data testing. """

    required_columns = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'salary']

    assert data is not None, "No data was loaded."
    assert data.shape[0] > 0, "Testing import_data: The file doesn't appear to have rows."
    
    # Check columns
    assert set(data.columns.values).issuperset(set(required_columns)), "Columns doesn't match."

@pytest.fixture
def split_data(data):
    train, test = train_test_split(
                    data, 
                    test_size=cfg["main"]["test_size"], 
                    random_state=cfg["main"]["random_seed"]
                    )
    return (train, test)


def test_model_training(split_data):
    """ Checks process_data outputs """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
                                        split_data[0], 
                                        cat_features, 
                                        label="salary", 
                                        training=True
                                        )
    
    assert X_train is not None, "Error spliting data. No X_train data."
    assert y_train is not None, "Error spliting data. No y_train data."
    assert encoder is not None, "Error spliting data. Encoder not created."
    assert lb is not None, "Error spliting data. Label Encoder not created."

    model = train_model(X_train, y_train, cfg["model"], 'pipeline/ml/' + cfg["main"]["model_path"])
    
    


def test_is_model_saved():
    """ Checks if the model has been saved """
    assert os.path.exists('pipeline/ml/' + cfg["main"]["model_path"])
