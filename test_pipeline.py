import pandas as pd
import logging
import pytest
import os
import yaml
from ml.data import load_data, process_data
from ml.model import train_model
from sklearn.model_selection import train_test_split


with open("config.yaml", "r") as config:
    cfg = yaml.safe_load(config)

@pytest.fixture
def data():
    """ 
    Load data for testing.
    """
    PATH = cfg["main"]["data_path"]

    if not os.path.exists(
        os.path.join(os.getcwd(), PATH)
    ):
        pytest.fail("Path doesn't exist. Please, verify.")

    df = load_data(PATH)

    return df


def test_data(data):
    """ Basic data testing. """

    required_columns = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'salary']

    if data is None:
        pytest.fail("No data was loaded.")

    if data.shape[0] == 0: 
        pytest.fail("Testing import_data: The file doesn't appear to have rows.")
    
    # Check columns
    if not set(data.columns.values).issuperset(set(required_columns)): 
        pytest.fail("A Feature is missing.")

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
    if X_train is None:
        pytest.fail("Error spliting data. No X_train data.")
    if y_train is None:
        pytest.fail("Error spliting data. No y_train data.")
    if encoder is None:
        pytest.fail("Error spliting data. Encoder not created.")
    if lb is None:
        pytest.fail("Error spliting data. Label Encoder not created.")     

    model = train_model(
        X_train=X_train, 
        y_train=y_train, 
        model_params=cfg["model"], 
        model_path=cfg["main"]["model_path"]
    )
    

def test_is_model_saved():
    """ Checks if the model has been saved """
    if not os.path.exists(cfg["main"]["model_path"]):
        pytest.fail("Model was not saved.")
