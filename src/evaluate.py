import argparse
import torch
import zenml
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_models_info_by_alias(client, alias):
    models_with_alias = []
    # Search for all registered models
    for model_info in client.search_registered_models(max_results=1000):
        model_aliases = model_info.aliases
        if any(alias in key for key in model_aliases.keys()):
            models_with_alias.append(model_info)

    return models_with_alias


def evaluate(data_sample_version, model_alias):
    print(f"Evaluating data sample version: {data_sample_version}")
    print(f"Using model alias: {model_alias}")
    try:
        data = zenml.load_artifact(
            name_or_id="features_target", version=str(data_sample_version)
        )
    except Exception as e:
        print("Error loading zenml artifacts\n")
        raise e

    # Load the client
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
    except Exception:
        pass
    client = mlflow.MlflowClient()

    X_test = data.drop(columns=["price"])
    y_test = data["price"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    models_with_alias = get_models_info_by_alias(client, model_alias)
    for model_info in models_with_alias:
        model = mlflow.pytorch.load_model(
            model_info.latest_versions[0].source, map_location=device
        )
        with torch.no_grad():
            y_pred = model(X_test_tensor).detach().cpu().numpy()
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model {model_info.name} with alias {model_alias} metrics are:")
        print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", type=str, default="sample_1", help="Data sample version"
    )
    parser.add_argument("--alias", type=str, default="champion", help="Model alias")

    args = parser.parse_args()
    evaluate(args.version, args.alias)


if __name__ == "__main__":
    main()
