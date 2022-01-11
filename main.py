import os
import logging
import hydra
from ml.data import load_data, process_data
from ml.model import compute_model_metrics, inference, train_model
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# This automatically reads in the configuration
@hydra.main(config_name='config')
def main(cfg):
    print(cfg)


    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Add code to load in the data.
    logging.info("Reading data.")
    data = load_data(os.path.join(root_path, cfg.main.data_path))

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logging.info("Spliting data.")
    train, test = train_test_split(
        data, 
        test_size=cfg.main.test_size,
        random_state=cfg.main.random_seed
    )

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
    logging.info("Processing data.")
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

    logging.info("Training the model.")
    # Train and save a model.
    model = train_model(
        X_train=X_train, 
        y_train=y_train, 
        model_params=cfg.model, 
        model_path=os.path.join(root_path, cfg.main.model_path)
        )

    logging.info("Saving model.")


    # Get predictions.
    logging.info("Get predictions.")
    preds = inference(model, X_test)

    # Get model metrics.
    precision, recall, f_beta = compute_model_metrics(y_test, preds)
    logging.info(
        f"Model Metrics: Precision {precision:.3f}, Recall {recall:.3f}, F_Beta {f_beta:.3f}"
    )



if __name__ == "__main__":
    main()
