import pandas as pd
import logging
import pytest
import os
from pipeline.ml.data import load_data
 

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

PATH = "data/census.csv"

@pytest.fixture
def data(PATH):
    """ 
    Load data for testing.
    """

    assert os.path.exists(PATH), f"Path doesn't exist. Please, verify."

    df = load_data(PATH)

    assert df is not None, "No data was loaded."
    assert df.shape[0] > 0 , "Testing import_data: The file doesn't appear to have rows."
    assert df.shape[1] > 0, "Testing import_data: The file doesn't appear to have columns."

    return df

    


