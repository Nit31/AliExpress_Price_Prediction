import mlflow
import hydra
import giskard
import torch
import zenml


def get_models_info_by_alias(client, alias):
    models_with_alias = []
    # Search for all registered models
    for model_info in client.search_registered_models(max_results=1000):
        model_aliases = model_info.aliases
        if any(alias in key for key in model_aliases.keys()):
            models_with_alias.append(model_info)

    return models_with_alias


def extract_data(cfg, version):
    try:
        df = zenml.load_artifact(
            name_or_id="features_target", version=version
        )
    except Exception as e:
        print("Error loading zenml artifacts\n")
        raise e

    return df


def test_model(client, cfg, model_info, giskard_dataset):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(
        model_info.latest_versions[0].source, map_location=device
    )

    def predict(df):
        X = df
        if cfg.zenml.features.target in df.columns:
            X = df.drop(columns=[cfg.zenml.features.target])
        # Predict using the model
        with torch.no_grad():
            output = (
                model(torch.tensor(X.values, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )
        
        return output.tolist()

    # print(predict(giskard_dataset.df.head()))
    # print(giskard_dataset.df.price.head())
    # Prepare feature names (excluding target column)
    feature_names = [col for col in giskard_dataset.df.columns if col != cfg.zenml.features.target]

    # Wrap the model
    giskard_model = giskard.Model(
        model=predict,
        model_type="regression",
        feature_names=feature_names,
    )
    
    # Create a test suite
    test_suite = giskard.Suite(name=f"test_suite_{model_info.name}")
    test_r2 = giskard.testing.test_r2(
        model=giskard_model, dataset=giskard_dataset, threshold=cfg.giskard.r2_threshold
    )
    test_suite.add_test(test_r2)
    # Run the test suite
    test_results = test_suite.run()
    assert test_results.passed


def main():
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
    except Exception:
        pass
    # Initialize the MLflow client
    client = mlflow.MlflowClient()

    # Initialize hydra
    hydra.initialize(
        config_path="../configs", job_name="preprocess_data", version_base=None
    )
    cfg = hydra.compose(config_name="main")

    # Get all models with the alias "Challenger"
    models_with_champion_alias = get_models_info_by_alias(client, "champion")

    # Wrap the raw dataset
    version = cfg.mlflow.test_data_version
    df = extract_data(cfg, version)


    giskard_dataset = giskard.Dataset(
        df=df,
        target=cfg.zenml.features.target
    )
    
    # X = df.drop(columns=cfg.zenml.features.target)

    # Raise exception if there are several champion models or zero
    if len(models_with_champion_alias) > 1:
        raise Exception("There should be exactly one champion model!")
    elif len(models_with_champion_alias) == 0:
        raise Exception("There should be at least one champion model!")

    # Test all the challenger models
    test_model(client, cfg, models_with_champion_alias[0], giskard_dataset)


if __name__ == "__main__":
    main()
