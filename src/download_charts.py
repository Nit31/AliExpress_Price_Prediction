import mlflow
from mlflow.tracking import MlflowClient
import os

try:
    mlflow.set_tracking_uri("http://localhost:5000")
except:
    pass

# Initialize MLflow client
client = MlflowClient()

# Get the default experiment (usually with ID '0')
experiment_name = "Default"
experiment = client.get_experiment_by_name(experiment_name)

# Get the latest run in the default experiment
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

if not runs:
    raise Exception("No runs found in the default experiment.")

latest_run = runs[0]
run_id = latest_run.info.run_id
artifact_name = "combined_metrics_plot.png"

# Define the directory where you want to save the result
result_directory = 'results'
os.makedirs(result_directory, exist_ok=True)

# Download the artifact
artifact_local_path = client.download_artifacts(run_id, artifact_name, result_directory)

print(f"Downloaded artifact saved to: {artifact_local_path}")
