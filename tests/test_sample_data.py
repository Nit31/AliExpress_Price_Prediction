import pytest
import types
from src.data import sample_data
import os
import yaml

# Load the YAML file
with open("configs/main.yaml", "r") as stream:
    config_main_data = yaml.safe_load(stream)

# Extract the data from the YAML file
db_main_data = config_main_data["db"]

# Generate the testdata list for @pytest.mark.parametrize
testdata = [
    (db_main_data["db_creator"], db_main_data["db_name"], db_main_data["sample_path"], db_main_data['data_path'], db_main_data["kaggle_filename"], config_main_data['dvc']['data_version_yaml_path'], 3, 42)
]

@pytest.mark.parametrize("db_creator, db_name, sample_path, data_path, kaggle_filename, data_version_yaml_path, data_version, seed", testdata)
def test_sample_data(db_creator, db_name, sample_path, data_path, kaggle_filename, data_version_yaml_path, data_version, seed):
    """
    Mock the hydra config

    Args:
        db_creator (_type_): _description_
        db_name (_type_): _description_
        sample_path (_type_): _description_
        data_path (_type_): _description_
        kaggle_filename (_type_): _description_
        data_version_yaml_path (_type_): _description_
        data_version (_type_): _description_
        seed (_type_): _description_
    """
    cfg = types.SimpleNamespace(
        db = types.SimpleNamespace(
            db_creator = db_creator,
            db_name = db_name,
            kaggle_filename = kaggle_filename,
            data_path = data_path,
            sample_path = sample_path
        ),
        dvc = types.SimpleNamespace(
            data_version_yaml_path = data_version_yaml_path
        ),
        data_version = types.SimpleNamespace(
            version = data_version
        ),
        experiment = types.SimpleNamespace(
            random_state = seed
        ),
    )
    print(cfg.db)  # Call the function with the test configuration

    sample_data(cfg)

    # Assert some expected behavior after calling the function
    # For example, check if the sample file was created correctly

    try:
        assert os.path.exists(sample_path)
    except AssertionError:
        pytest.fail("File is not saved in data/samples")

def test_sample_data_exception():
    """
    Test sample_data fucntion for exception
    """
    with pytest.raises(Exception):
        cfg = {
            "db": {
                "db_creator": "invalid_creator",
                "db_name": "invalid_name",
                "kaggle_filename": "invalid_filename",
                "sample_path": "invalid_sample.csv",
                "data_path": "data_path"
            }
        }
        sample_data(cfg)  # Call the function with invalid data which should raise an exception

