from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os
import torch
import json
import zenml
import hydra
from hydra.core.global_hydra import GlobalHydra


BASE_PATH = os.path.expandvars("$PWD")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mlflow.pytorch.load_model(f'{BASE_PATH}/api/model_dir', 
                                  map_location=device)

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():

  # response = make_response(str(model.metadata), 200)
  response = make_response(str('Hi').encode('utf-8'), 200)
  response.content_type = "text/plain"
  return response

# /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    content = request.data
    decoded_string = content.decode('utf-8')
    data = json.loads(decoded_string)
    inputs = torch.tensor(list(data['inputs'].values()))
    print(inputs.shape)
    with torch.no_grad():
        prediction = model(inputs.to(device)).cpu().numpy().flatten()
            
    target_preprocessor = zenml.load_artifact(name_or_id='target_preprocessor', version='1')
    # Extract the numerical transformer
    num_transformer = target_preprocessor.transformers_[0][1]
    # Ensure output is a 2D array with shape (n_samples, n_features)
    # If output is one-dimensional, reshape it to 2D
    output_reshaped = prediction.reshape(-1, 1)  # Reshape to 2D if needed
    
    # Perform the inverse transformation
    inverse_transformed_output = num_transformer.inverse_transform(output_reshaped)
    response = make_response(str(inverse_transformed_output[0][0]), 200)
    response.headers["content-type"] = "application/json"
    return response

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    
    if GlobalHydra.instance().is_initialized():
            print("Using existing Hydra global instance.")
            cfg = hydra.compose(config_name="main")
    else:
        print("Initializing a new Hydra global instance.")
        hydra.initialize(config_path="../configs", job_name="streamlit", version_base=None)
        cfg = hydra.compose(config_name="main")
        
    port = int(os.environ.get('PORT', cfg.api_port))
    app.run(debug=True, host='0.0.0.0', port=port)