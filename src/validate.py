import mlflow
import hydra
from data import sample_data, validate_initial_data, handle_initial_data
import giskard


def get_models_info_by_alias(client, alias):
    models_with_alias = []
    # Search for all registered models
    for model_info in client.search_registered_models():
        model_aliases = model_info.aliases
        if alias in model_aliases.keys():
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


def test_model(cfg, model_info, giscard_dataset):
    # Load the model
    model = mlflow.pytorch.load_model(model_info.latest_versions[0].source)
    print(model)
    
    
def main():
    # Initialize the MLflow client
    client = mlflow.MlflowClient()

    # Initialize hydra
    hydra.initialize(config_path="../configs", job_name="preprocess_data", version_base=None)
    cfg = hydra.compose(config_name="main")

    # Get all models with the alias "Challenger"
    models_with_challenger_alias = get_models_info_by_alias(client, "challenger")
    
    # Wrap the raw dataset
    version = cfg.mlflow.test_data_version
    df = extract_data(cfg, version)
    
    TARGET_COLUMN = cfg.zenml.features.target
    CATEGORICAL_COLUMNS = list(cfg.zenml.features.categorical)
    
    giskard_dataset = giskard.Dataset(
        df=df,
        target=TARGET_COLUMN,
        cat_columns=CATEGORICAL_COLUMNS
    )
    
    # Test all the challenger models
    for model_info in models_with_challenger_alias:
        test_model(cfg, model_info, giskard_dataset)
    

if __name__ == "__main__":
    main()