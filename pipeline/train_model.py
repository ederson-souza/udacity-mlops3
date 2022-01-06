# Script to train machine learning model.
import logging
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from ml.data import process_data, load_data
from ml.model import compute_model_metrics, inference, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add code to load in the data.
logger.info("Reading data.")
data = load_data("../data/census.csv")
print(data.shape)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Spliting data.")
train, test = train_test_split(data, test_size=0.20)

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
logger.info("Processing data.")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    X=test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False,
    encoder=encoder,
    lb=lb

)

logger.info("Training the model.")
# Train and save a model.
model = train_model(X_train, y_train)
logger.info("Model saved.")
dump(model, '../model/model.joblib')

# Get predictions.
logger.info("Get predictions.")
preds = inference(model, X_test)

# Get model metrics.
precision, recall, f_beta = compute_model_metrics(y_test, preds)
logger.info(
    f"Model Metrics: Precision {precision:.3f}, Recall {recall:.3f}, F_Beta {f_beta:.3f}"
)

