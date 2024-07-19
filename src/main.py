import pandas as pd
import hydra
import zenml
import yaml
from sklearn.model_selection import train_test_split
from ridge_reg import run_ridge_regression
from nn_model import nn_run
from omegaconf import OmegaConf


def get_split_data(cfg):
    try:
        print("!!!!!!!!!!!!!!!!")
        print(cfg.mlflow.train_val_data_version)
        train_val = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.train_val_data_version))
        test = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.test_data_version))
    except Exception as e:
        print('Error loading zenml artifacts\n')
        raise e

    X_train, X_val, y_train, y_val = train_test_split(train_val.drop(columns=['price']), train_val['price'],\
        test_size=cfg.mlflow.val_size, random_state=42)
    
    # sample out of test data
    test = pd.DataFrame(test)
    test = test.sample(frac=cfg.mlflow.test_size, random_state=42)
    
    X_test = test.drop(columns=['price'])
    y_test = test['price']
    return X_train, X_val, X_test, y_train, y_val, y_test

def train():
    pass

def log_metadata():
    ...


@hydra.main(config_path="../configs", config_name='main', version_base=None)
def main(cfg=None):
    # print(OmegaConf.to_yaml(cfg))
    
    # Extract data
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(cfg)
    print(len(X_train), len(X_val), len(X_test))

    # # nn_run(cfg,X_train.to_numpy(),X_val.to_numpy(),X_test.to_numpy(),y_train.to_numpy(),y_val.to_numpy(),y_test.to_numpy())

    # # Train the models
    # gs = train(X_train, y_train, cfg=cfg)

    # # Log the metadata
    # log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
