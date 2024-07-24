import argparse
import pandas as pd
import hydra
import torch
import zenml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
from nn_model import nn_run, set_seed
from omegaconf import OmegaConf
import mlflow
from mlflow.tracking import MlflowClient
from nn_model import train_and_evaluate_model
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_by_alias(model_name, alias):
    client = MlflowClient()

    # Get the model version associated with the alias
    model_version = client.get_model_version_by_alias(model_name, alias).version

    # Load the model
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

    return model


def evaluate(data_sample_version, model_alias):
    print(f"Evaluating data sample version: {data_sample_version}")
    print(f"Using model alias: {model_alias}")
    try:
        data = zenml.load_artifact(name_or_id='features_target', version=str(data_sample_version))
    except Exception as e:
        print('Error loading zenml artifacts\n')
        raise e
    client = MlflowClient()
    registered_models = client.search_registered_models()
    needed_model = None
    for model in registered_models:
        model_name = model.name


        # Get all versions of the current model
        versions = client.search_model_versions(f"name='{model_name}'")

        # Iterate through each version
        for version in versions:
            version_number = version.version

            # Retrieve the model version details
            model_version = client.get_model_version(model_name, version_number)

            # Get the aliases for this model version
            aliases = model_version.aliases
            # Download the model for each alias
            for alias in aliases:
                if alias == model_alias:
                    needed_model = model
                    break
            if needed_model is not None:
                break
        if needed_model is not None:
            break
    X_test = data.drop(columns=['price'])
    y_test = data['price']
    device = next(model.model.parameters()).device
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_pred = needed_model(X_test_tensor).detach().cpu().numpy()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

@hydra.main(config_path="../configs", config_name='main', version_base=None)
def main(cfg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_sample_version", type=str, default="sample_1", help="Data sample version")
    parser.add_argument("--model_alias", type=str, default="champion", help="Model alias")

    args = parser.parse_args()
    mae,mse,r2 = evaluate(args.data_sample_version, args.model_alias)
    print(f'{mae},{mse},{r2}')


if __name__ == "__main__":
    main()
