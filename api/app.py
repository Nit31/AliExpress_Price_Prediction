from flask import Flask, request, jsonify, abort, make_response

import mlflow
import mlflow.pyfunc
import os
import torch

BASE_PATH = os.path.expandvars("$PWD")

# def get_model_info(client):
#     models = []
#     # Search for all registered models
#     for model_info in client.search_registered_models():
#         models.append(model_info)
#     return models[0]
# client = mlflow.MlflowClient()
# model_info = get_model_info(client)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mlflow.pytorch.load_model(f'{BASE_PATH}/mlruns/452294063850305597/a11fe7218c8742ea9e130a7b148a54a6/artifacts/models/Model_dim[512]_layers[3]_experiment/r2_score0.391447074233853_run1', 
                                  map_location=device)

# def test_model(model_info):
#     # Load the model
#     print(BASE_PATH, model_info.latest_versions[0].source)
#     model = mlflow.pytorch.load_model(model_info.latest_versions[0].source)
#     return model

# print(test_model(model_info))

app = Flask(__name__)

@app.route("/info", methods = ["GET"])
def info():

  # response = make_response(str(model.metadata), 200)
  response = make_response(str('Hi').encode('utf-8'), 200)
  response.content_type = "text/plain"
  return response

# /predict endpoint
@app.route("/predict", methods = ["POST"])
def predict():
	
    # EDIT THIS ENDPOINT
    
    # EXAMPLE
	content = str(request.data)
	response = make_response(content, 200)
	response.headers["content-type"] = "application/json"
	return jsonify({'result': 'yes', 'prob': '0.7'})

# This will run a local server to accept requests to the API.
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)