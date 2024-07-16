# src/main.py


import hydra
# from model import train, load_features, log_metadata
from omegaconf import OmegaConf
import zenml
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function for extracting features from zenml artifact
def extract_data(is_test: bool = False):
    with open("configs/data_version.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    if is_test:
        version = str(cfg['version_for_test'])
    else:
        version = str(cfg['version'])
    try:
        df = zenml.load_artifact(name_or_id='features_target', version=version)
    except:
        print('Error loading')
        df = None
    if df is None:
        print('Errors with extraction features data!')
    return df

# Function that load train_val and test samples and split them by ratio
def get_split_data():
    with open("configs/split_ratio.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)
    train_ratio = cfg['train']
    validation_ratio = cfg['validation']
    test_ratio = cfg['test']
    train_val_set_sample = extract_data(False)
    test_set_sample = extract_data(True)
    X_train_val_set_sample = train_val_set_sample.drop(columns=['price'])
    y_train_val_set_sample = train_val_set_sample['price']
    X_test_set_sample = test_set_sample.drop(columns=['price'])
    y_test_set_sample = test_set_sample['price']
    X_train, X_val, y_train, y_val = train_test_split(X_train_val_set_sample, y_train_val_set_sample, test_size=validation_ratio, random_state=42)
    X_test, _, y_test, _ = train_test_split(X_test_set_sample, y_test_set_sample, test_size=1-test_ratio, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

@hydra.main(config_path="../configs", config_name="main", version_base=None)  # type: ignore
def main(cfg=None):
    print(len(get_split_data()))


if __name__ == "__main__":
    main()
