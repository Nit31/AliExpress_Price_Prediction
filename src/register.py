import mlflow

# Path to the directory containing the model artifacts
model_uri = "models/champion/artifacts/models/Model_dim512_layers3_experiment/r2_score0.5740082318163814_run20"

# Name you want to give to the model in the registry
model_name = "my_champion_model"

# Register the model
result = mlflow.register_model(model_uri=model_uri, name=model_name)

# Output the result
print(f"Model registered with name: {model_name}")
print(f"Model version: {result.version}")
