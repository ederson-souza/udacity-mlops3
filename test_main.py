import pytest
from fastapi.testclient import TestClient
from main import app



home = TestClient(app)


def test_index():
    connect = home.get("/")
    assert connect.status_code == 200
    assert connect.json() == ["Welcome to the Salary Predictor"]

def test_predict_1():
    data1 = {
        "workclass": "Private",
        "education": "HS-grad",
        "marital_status": "Never-married",
    }

    connect = home.post('/salary/', json=data1)
    assert connect.status_code == 200
