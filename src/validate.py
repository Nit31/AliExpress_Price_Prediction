import mlflow
from mlflow import MlflowClient
import hydra


def get_models_info_by_alias(client, alias):
        models_with_alias = []

        # Search for all registered models
        for model_info in client.search_registered_models():
            print(model_info.aliases)
            model_aliases = model_info.aliases
            if alias in model_aliases.keys():
                models_with_alias.append(model_info)

        return models_with_alias
    
    
def main():
    # Initialize the MLflow client
    client = MlflowClient()

    # Initialize hydra
    hydra.initialize(config_path="../configs", job_name="preprocess_data", version_base=None)
    cfg = hydra.compose(config_name="main")

    # Get all models with the alias "Challenger"
    models_with_challenger_alias = get_models_info_by_alias(client, "challenger")
    
    

    # Print all models with the alias
    for model in models_with_challenger_alias:
        print(f"Model Name: {model.name}, Alias: {model.aliases}")
