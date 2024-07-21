import pandas as pd
import hydra
import torch
import zenml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
from nn_model import nn_run
from omegaconf import OmegaConf
import mlflow
from nn_model import train_and_evaluate_model
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_split_data(cfg):
    try:
        train = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.train_val_data_version))
        test = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.test_data_version))
    except Exception as e:
        print('Error loading zenml artifacts\n')
        raise e

    X_train, y_train = train.drop(columns=['price']), train['price']
    
    # sample out of test data
    test = pd.DataFrame(test)
    test = test.sample(frac=cfg.mlflow.test_size, random_state=cfg.experiment.random_state)
    
    X_test = test.drop(columns=['price'])
    y_test = test['price']
    return X_train, X_test, y_train, y_test


def log_metadata(cfg, models, X_train, X_test, y_train, y_test):
    # FIXME: Need to start mlflow server. Maybe need to reconsider
    mlflow.set_tracking_uri("http://localhost:5000")
    architecture_params = models[0]['params']
    experiment_name = f"Model_dim{architecture_params['hidden_dim']}_layers{architecture_params['hidden_layers']}_experiment"
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
        #experiment_id = mlflow.create_experiment(experiment_name).experiment
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(experiment_id)
    # FIXME: In first run mlflow always save in default experiment
    with mlflow.start_run(run_name='first_run', experiment_id=experiment_id) as run:
        pass
    for i, whole_params in enumerate(models):
        model = whole_params['model']
        mean_score = whole_params['mean_score']
        params = whole_params['params']
        run_name = f'r2_score{mean_score}_run{i}'

        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            print(f'exp:{mlflow.get_run(run.info.run_id).info.experiment_id}')
            try:
                mlflow.log_params(params)
            except mlflow.exceptions.MlflowException as e:
                if "Changing param values is not allowed" in str(e):
                    print(f"Parameter already logged. Skipping...")
            mlflow.log_metrics({"train r2": mean_score})
            mlflow.set_tag("Training Info", f"Fully-connected model architecture for aliexpress")
            # Infer the model signature
            signature = infer_signature(X_train, y_train)
            # Log the model
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model.model,
                artifact_path=f'models/{experiment_name}/{run_name}',
                signature=signature,
                input_example=X_train,  # Use X_train or X_test as needed
                registered_model_name=f'{experiment_name}_{run_name}'
            )
            eval_data = pd.DataFrame(y_test.values, columns=["real"])
            # Get predictions from the model
            device = next(model.model.parameters()).device
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

            # Get predictions from the model
            eval_data["predictions"] = model.model(X_test_tensor).detach().cpu().numpy()
            mae = mean_absolute_error(eval_data["real"], eval_data["predictions"])
            mse = mean_squared_error(eval_data["real"], eval_data["predictions"])
            r2 = r2_score(eval_data["real"], eval_data["predictions"])

            # Print the metrics
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            print(f"R² Score: {r2}")

            # Log metrics to MLflow
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            print('Done!')


def train(X_train, y_train, cfg):
        param_grid = dict(cfg.model.params)
        models = list()
        #experiment_name = f"Model_dim{param_grid['hidden_dim']}_layers{param_grid['hidden_layers']}_experiment"
        # try:
        #     # Create a new MLflow Experiment
        #     experiment_id = mlflow.create_experiment(name=experiment_name)
        # except mlflow.exceptions.MlflowException as e:
        #     experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f'Run {i}')
            model, mean_score = train_and_evaluate_model(params, X_train, y_train)
            models.append({'model':model, 'mean_score':mean_score, 'params':params})
        return models
        # run_name = f'r2_score{mean_score}_run{i}'
        # with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:


            # mlflow.log_params(params)
            # mlflow.log_metrics({"r2": mean_score})
            # mlflow.set_tag("Training Info", f"Fully-connected model architecture for aliexpress using")
            # # Fit the model on the entire training set
            #
            # # Infer the model signature
            # signature = infer_signature(X_train, y_train)
            # # Log the model
            # model_info = mlflow.pytorch.log_model(
            #     pytorch_model=model.model,
            #     artifact_path=f'models/{experiment_name}/{run_name}',
            #     signature=signature,
            #     input_example=X_train,  # Use X_train or X_test as needed
            #     registered_model_name=f'{experiment_name}_{run_name}'
            # )



@hydra.main(config_path="../configs", config_name='main', version_base=None)
def main(cfg=None):
    print(OmegaConf.to_yaml(cfg))
    
    # Extract data
    X_train, X_test, y_train, y_test = get_split_data(cfg)

    # Train the models
    models = train(X_train.to_numpy(), y_train.to_numpy(), cfg=cfg)

    log_metadata(cfg, models, X_train, X_test, y_train, y_test)

    # # nn_run(cfg,X_train.to_numpy(),X_val.to_numpy(),X_test.to_numpy(),y_train.to_numpy(),y_val.to_numpy(),y_test.to_numpy())



    # # Log the metadata
    # log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
