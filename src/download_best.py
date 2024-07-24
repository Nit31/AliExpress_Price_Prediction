import mlflow
from mlflow.tracking import MlflowClient
import re

# Function to download a model from a run
def download_model(artifact_uri, dest_path):
    mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri,
        dst_path=dest_path
    )


def main():
    # Define the regex pattern for champion and challenger aliases
    alias_pattern = re.compile(r'^(champion|challenger\d*)$')
    
    # Set the path for the local models folder
    models_folder = "models"

    # Connect to the MLflow tracking server
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
    except Exception:
        pass
    client = MlflowClient()

    for model_info in client.search_registered_models(max_results=1000):
        model_aliases = model_info.aliases
        if any(alias_pattern.match(key) for key in model_aliases.keys()):
            print(model_info)
            print(f"Downloading '{model_info.latest_versions[0].name}' model with alias '{model_aliases}'")
            download_model(model_info.latest_versions[0].source, models_folder)
            

    print(f"Models downloaded successfully to the '{models_folder}' folder.")



if __name__ == "__main__":
    main()