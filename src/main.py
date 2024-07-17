import hydra
import zenml
import yaml
from sklearn.model_selection import train_test_split


# # Function for extracting features from zenml artifact
# def extract_data(cfg, is_test: bool = False):
#     with open("configs/data_version.yaml", "r") as stream:
#         cfg = yaml.safe_load(stream)
#     if is_test:
#         version = str(cfg['version_for_test'])
#     else:
#         version = str(cfg['version'])
#     try:
#         df = zenml.load_artifact(name_or_id='features_target', version=version)
#     except Exception as e:
#         print('Error loading\n', e)
#         df = None
#     if df is None:
#         raise Exception('Errors with extraction data!')
#     return df

# # Function that load train_val and test samples and split them by ratio
# def get_split_data(cfg):
#     # Parse data_version
#     with open(cfg.dvc.data_version_yaml_path) as stream:
#         try:
#             data_version = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#             raise
#     with open_dict(cfg):
#         cfg.db.sample_part = data_version['version']
    
#     with open(cfg.mlflow.split_ratio_path, "r") as stream:
#         cfg = yaml.safe_load(stream)
#     train_ratio = cfg['train']
#     validation_ratio = cfg['validation']
#     test_ratio = cfg['test']
#     train_val_set_sample = extract_data(False)
#     test_set_sample = extract_data(True)
#     X_train_val_set_sample = train_val_set_sample.drop(columns=['price'])
#     y_train_val_set_sample = train_val_set_sample['price']
#     X_test_set_sample = test_set_sample.drop(columns=['price'])
#     y_test_set_sample = test_set_sample['price']
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val_set_sample, y_train_val_set_sample, test_size=validation_ratio, random_state=42)
#     X_test, _, y_test, _ = train_test_split(X_test_set_sample, y_test_set_sample, test_size=1-test_ratio, random_state=42)
#     return X_train, X_val, X_test, y_train, y_val, y_test

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    # Extract data
    print(len(get_split_data(cfg)))
    # TODO: Train the model
    
    # TODO: Evaluate the model
    
    # TODO: Log metadata

if __name__ == "__main__":
    main()
