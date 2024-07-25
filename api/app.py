from flask import Flask, request, make_response

import mlflow
import mlflow.pyfunc
import os
import torch
import json
import zenml
import hydra
from hydra.core.global_hydra import GlobalHydra

def get_champion_metadata(client):
    models_with_alias = []
    # Search for all registered models
    for model_info in client.search_registered_models(max_results=1000):
        model_aliases = model_info.aliases
        if any('champion' in key for key in model_aliases.keys()):
            return model_info

    return 'Champion not found'

# Load the model
BASE_PATH = os.path.expandvars("$PWD")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlflow.pytorch.load_model(f'{BASE_PATH}/api/model_dir', map_location=device)
app = Flask(__name__)

# info endpoint for taking metadata of the model
@app.route("/info", methods = ["GET"])
def info():
    """API endpoint to retrieve metadata of the working(champion) model

    Returns:
        response: HTTP responce with metadata
    """    
    client = mlflow.MlflowClient()
    
    champion_info = get_champion_metadata(client)
    
    response = make_response(str(champion_info), 200)
    response.content_type = "text/plain"
    return response


# /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    """API end point for prediction based on given product info

    Returns:
        response: HTTP responce with predicted price
    """
    # Extract data
    content = request.data
    decoded_string = content.decode("utf-8")
    data = json.loads(decoded_string)
    inputs = torch.tensor(list(data['inputs'].values()))
    
    # Predict
    with torch.no_grad():
        prediction = model(inputs.to(device)).cpu().numpy().flatten()
    
    # Invers target transormations
    target_preprocessor = zenml.load_artifact(name_or_id='target_preprocessor', version='1')
    # Extract the numerical transformer
    num_transformer = target_preprocessor.transformers_[0][1]
    # Ensure output is a 2D array with shape (n_samples, n_features)
    # If output is one-dimensional, reshape it to 2D
    output_reshaped = prediction.reshape(-1, 1)  # Reshape to 2D if needed

    # Perform the inverse transformation
    inverse_transformed_output = num_transformer.inverse_transform(output_reshaped)
    
    # Convert data into response
    response = make_response(str(inverse_transformed_output[0][0]), 200)
    response.headers["content-type"] = "application/json"
    return response


# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    
    # Load the configs
    if GlobalHydra.instance().is_initialized():
        print("Using existing Hydra global instance.")
        cfg = hydra.compose(config_name="main")
    else:
        print("Initializing a new Hydra global instance.")
        hydra.initialize(
            config_path="../configs", job_name="streamlit", version_base=None
        )
        cfg = hydra.compose(config_name="main")

    port = int(os.environ.get("PORT", cfg.api_port))
    app.run(debug=True, host="0.0.0.0", port=port)
