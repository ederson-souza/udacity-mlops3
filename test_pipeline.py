import pandas as pd
import logging
import pytest
import os
from pipeline.ml.data import load_data


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@pytest.fixture
def data():
    """ 
    Load data for testing.
    """
    PATH = "data/census.csv"

    assert os.path.exists(
        os.path.join(os.getcwd(), PATH)
    ), f"Path doesn't exist. Please, verify."

    df = load_data(PATH)

    return df


def test_data(data):

    features = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race',
               'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'salary']

    assert data is not None, "No data was loaded."
    assert data.shape[0] > 0, "Testing import_data: The file doesn't appear to have rows."
    assert list(data.columns) == features, "Features doesn't match."

