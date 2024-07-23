import mlflow
from mlflow.tracking import MlflowClient
import os
try:
    mlflow.set_tracking_uri("http://localhost:5000")
except:
    pass
# Initialize MLflow client
client = MlflowClient()

# Define the run ID and the artifact name
run_id = "c77c9f07b6384284a2db164163a309d6"
artifact_name = "combined_metrics_plot.png"

# Define the directory where you want to save the result
result_directory = 'results'
os.makedirs(result_directory, exist_ok=True)

# Download the artifact
artifact_local_path = client.download_artifacts(run_id, artifact_name, result_directory)

print(f"Downloaded artifact saved to: {artifact_local_path}")
