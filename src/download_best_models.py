import mlflow
from mlflow.tracking import MlflowClient
import os
try:
    mlflow.set_tracking_uri("http://localhost:5000")
except:
    pass
# Initialize MLflow client
client = MlflowClient()

# Define the base directory where models will be saved
models_directory = 'models'
os.makedirs(models_directory, exist_ok=True)

# Retrieve all registered models
registered_models = client.search_registered_models()

# Iterate through all registered models
for model in registered_models:
    model_name = model.name
    print(f"Processing model: {model_name}")

    # Get all versions of the current model
    versions = client.search_model_versions(f"name='{model_name}'")

    # Iterate through each version
    for version in versions:
        version_number = version.version
        print(f"  Version: {version_number}")

        # Retrieve the model version details
        model_version = client.get_model_version(model_name, version_number)

        # Get the aliases for this model version
        aliases = model_version.aliases
        print(f"    Aliases: {aliases}")

        # Download the model for each alias
        for alias in aliases:
            # Create a directory for the alias if it does not exist
            alias_directory = os.path.join(models_directory, alias)
            os.makedirs(alias_directory, exist_ok=True)

            # Define the artifact path for this version
            artifact_path = f"{'_'.join(model_name.split('_')[:4])}"
            print(artifact_path)
            print(alias_directory)
            # Download the model artifacts to the alias directory
            print(f"    Downloading to: {alias_directory}")
            client.download_artifacts(model_version.run_id, artifact_path, alias_directory)
