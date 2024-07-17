import hydra
import zenml
import yaml
from sklearn.model_selection import train_test_split


# Function that load train_val and test samples and split them by ratio
def get_split_data(cfg):
    try:
        train_val = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.train_val_data_version))
        test = zenml.load_artifact(name_or_id='features_target', version=str(cfg.mlflow.test_data_version))
    except Exception as e:
        print('Error loading zenml artifacts\n')
        raise e

    X_train, X_val, y_train, y_val = train_test_split(train_val.drop(columns=['price']), train_val['price'],\
        test_size=cfg.mlflow.val_size, random_state=42)
    
    # Cut the test data
    test = test.sample(cfg.mlflow.test_size, random_state=42)
    
    X_test = test.drop(columns=['price'])
    y_test = test['price']
    return X_train, X_val, X_test, y_train, y_val, y_test

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    # Extract data
    X_train, X_val, X_test, y_train, y_val, y_test = get_split_data(cfg)
    # TODO: Train the model
    
    # TODO: Evaluate the model
    
    # TODO: Log metadata

if __name__ == "__main__":
    main()
