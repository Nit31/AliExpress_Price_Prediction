import mlflow
import hydra
from data import sample_data, validate_initial_data, handle_initial_data, preprocess_data
import giskard
import torch


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

def test_model(client, cfg, model_info, giskard_dataset):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(model_info.latest_versions[0].source, map_location=device)
    
    def predict(df):
        X, _ = preprocess_data(df, cfg, skip_target=True)
        with torch.no_grad():
            return model(torch.tensor(X.values, dtype=torch.float32).to(device)).cpu().numpy().flatten()
        
    # print(predict(giskard_dataset.df.head()))
    # Wrap the model
    giskard_model = giskard.Model(
        model=predict,
        model_type = "regression",
        feature_names = giskard_dataset.df.columns,
    )
    # Perform scan
    scan_results = giskard.scan(giskard_model, giskard_dataset)
    # Save the results
    scan_results.to_html(f"reports/test_suite_{model_info.name}_testdata_version_{cfg.data_version.version}.html")
    # Create a test suite
    test_suite = giskard.Suite(name = f"test_suite_{model_info.name}")
    test_r2 = giskard.testing.test_r2(model = giskard_model, 
                                dataset = giskard_dataset,
                                threshold=cfg.giskard.r2_threshold)
    test_suite.add_test(test_suite.add_test(test_r2))
    # Run the test suite
    test_results = test_suite.run()
    print(f"Model {model_info.name}:")
    if (test_results.passed):
        print("Passed model validation!")
    else:
        print("Model has vulnerabilities!")
        
        
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
        test_model(client, cfg, model_info, giskard_dataset)
    

if __name__ == "__main__":
    main()