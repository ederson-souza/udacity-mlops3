import pandas as pd
import logging
import pytest
import os
import yaml
from pipeline.ml.data import load_data


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
    PATH = cfg["data_path"]

    assert os.path.exists(
        os.path.join(os.getcwd(), PATH)
    ), f"Path doesn't exist. Please, verify."

    df = load_data(PATH)

    return df


def test_data(data):

    required_columns = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'salary']

    assert data is not None, "No data was loaded."
    assert data.shape[0] > 0, "Testing import_data: The file doesn't appear to have rows."
    
    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys())), "Columns doesn't match."