import pandas as pd
import hydra
import zenml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
from nn_model import nn_run
from omegaconf import OmegaConf
import mlflow
from nn_model import train_and_evaluate_model
from mlflow.models import infer_signature


def get_split_data(cfg):
    try:
        train_val = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.train_val_data_version))
        test = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.test_data_version))
    except Exception as e:
        print('Error loading zenml artifacts\n')
        raise e

    X_train, X_val, y_train, y_val = train_test_split(train_val.drop(columns=['price']), train_val['price'],\
        test_size=cfg.mlflow.val_size, random_state=cfg.experiment.random_state)
    
    # sample out of test data
    test = pd.DataFrame(test)
    test = test.sample(frac=cfg.mlflow.test_size, random_state=cfg.experiment.random_state)
    
    X_test = test.drop(columns=['price'])
    y_test = test['price']
    return X_train, X_val, X_test, y_train, y_val, y_test

def train(X_train, y_train, cfg):
    param_grid = dict(cfg.model.params)
    experiment_name = f"Model_dim{param_grid['hidden_dim']}_layers{param_grid['hidden_layers']}_experiment"
    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id    
    for i, params in enumerate(ParameterGrid(param_grid)):
        model, mean_score = train_and_evaluate_model(params, X_train, y_train)
        run_name = f'r2_score{mean_score}_run{i}'
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"r2": mean_score})
            mlflow.set_tag("Training Info", f"Fully-connected model architecture for aliexpress using")
            # Fit the model on the entire training set
            model.fit(X_train, y_train)
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
    return None
        
def log_metadata():
    ...


@hydra.main(config_path="../configs", config_name='main', version_base=None)
def main(cfg=None):
    print(OmegaConf.to_yaml(cfg))
    
    # Extract data
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(cfg)

    # # nn_run(cfg,X_train.to_numpy(),X_val.to_numpy(),X_test.to_numpy(),y_train.to_numpy(),y_val.to_numpy(),y_test.to_numpy())

    # Train the models
    # train(X_train.to_numpy(), y_train.to_numpy(), cfg=cfg)

    # # Log the metadata
    # log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
