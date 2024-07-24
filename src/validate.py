import mlflow
import hydra
from data import (
    sample_data,
    validate_initial_data,
    handle_initial_data,
    preprocess_data,
)
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
    # take a sample
    sample = sample_data(cfg)

    # update the version of the sample
    cfg.data_version.version = version

    # validate the sample
    try:
        # if the validation failed, then try to handle the initial data
        assert validate_initial_data(cfg, sample)
    except Exception:
        sample = handle_initial_data(sample)
    assert validate_initial_data(cfg, sample)

    return sample


def test_model(client, cfg, model_info, giskard_dataset):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(
        model_info.latest_versions[0].source, map_location=device
    )

    def predict(df):
        # Preprocess the data
        X, _ = preprocess_data(df, cfg, skip_target=True)

        # Predict using the model
        with torch.no_grad():
            output = (
                model(torch.tensor(X.values, dtype=torch.float32).to(device))
                .cpu()
                .numpy()
                .flatten()
            )

        # Load the target preprocessor
        target_preprocessor = zenml.load_artifact(
            name_or_id="target_preprocessor", version="1"
        )

        # Extract the numerical transformer
        num_transformer = target_preprocessor.transformers_[0][1]

        # Ensure output is a 2D array with shape (n_samples, n_features)
        # If output is one-dimensional, reshape it to 2D
        output_reshaped = output.reshape(-1, 1)  # Reshape to 2D if needed

        # Perform the inverse transformation
        inverse_transformed_output = num_transformer.inverse_transform(output_reshaped)

        return inverse_transformed_output

    # print(predict(giskard_dataset.df.head()))
    # Wrap the model
    giskard_model = giskard.Model(
        model=predict,
        model_type="regression",
        feature_names=giskard_dataset.df.columns,
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

    TARGET_COLUMN = cfg.zenml.features.target
    CATEGORICAL_COLUMNS = list(cfg.zenml.features.categorical)

    giskard_dataset = giskard.Dataset(
        df=df, target=TARGET_COLUMN, cat_columns=CATEGORICAL_COLUMNS
    )

    # Raise exception if there are several champion models or zero
    if len(models_with_champion_alias) > 1:
        raise Exception("There should be exactly one champion model!")
    elif len(models_with_champion_alias) == 0:
        raise Exception("There should be at least one champion model!")

    # Test all the challenger models
    for model_info in models_with_champion_alias:
        test_model(client, cfg, model_info, giskard_dataset)


if __name__ == "__main__":
    main()
