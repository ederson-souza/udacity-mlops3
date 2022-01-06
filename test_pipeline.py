import pandas as pd
import pytest


@pytest.fixture
def data():
    """ Load data for testing."""
    
    df = pd.read_csv("../data/census.csv")

    return df


