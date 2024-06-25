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
    (db_main_data["db_creator"], db_main_data["db_name"], db_main_data["kaggle_filename"], db_main_data['sample_part'],
        db_main_data["sample_path"])
]

@pytest.mark.parametrize("db_creator, db_name, kaggle_filename, sample_part, sample_path", testdata)
def test_sample_data(db_creator, db_name, kaggle_filename, sample_part, sample_path):
    cfg = types.SimpleNamespace(
        db = types.SimpleNamespace(
            db_creator = db_creator,
            db_name = db_name,
            kaggle_filename = kaggle_filename,
            sample_part = sample_part,
            sample_path = sample_path
        )
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
    with pytest.raises(Exception):
        cfg = {
            "db": {
                "db_creator": "invalid_creator",
                "db_name": "invalid_name",
                "kaggle_filename": "invalid_filename",
                "sample_part": 2,
                "sample_path": "invalid_sample.csv"
            }
        }
        sample_data(cfg)  # Call the function with invalid data which should raise an exception

