import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
#from data import transform_data, extract_data, load_features, validate_transformed_data
#from utils import get_sample_version
import sys
import os

# Add the parent directory of 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src import data

import os

BASE_PATH = os.path.expandvars("$PROJECTPATH")

@step(enable_cache=False)
def extract()-> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="extracted_data",
                   tags=["data_preparation"]
                   )
    ],
    Annotated[int,
    ArtifactConfig(name="data_version",
                   tags=["data_preparation"])]
]:

    df, version = data.read_datastore()
    return df, version

@step(enable_cache=False)
def transform(df: pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="input_features",
                   tags=["data_preparation"])],
    Annotated[pd.DataFrame,
    ArtifactConfig(name="input_target",
                   tags=["data_preparation"])]
]:

    # Your data transformation code
    X, y = data.preprocess_data(df)

    return X, y

@step(enable_cache=False)
def validate(X:pd.DataFrame,
             y:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="valid_input_features",
                   tags=["data_preparation"])],
    Annotated[pd.DataFrame,
    ArtifactConfig(name="valid_target",
                   tags=["data_preparation"])]
]:
    success = data.validate_features(X, y)
    if success:
        return X, y
    else:
        raise Exception('Errors with Expectations!')

@step(enable_cache=False)
def load(X:pd.DataFrame, y:pd.DataFrame, version: str)-> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="features",
                   tags=["data_preparation"])],
    Annotated[pd.DataFrame,
    ArtifactConfig(name="target",
                   tags=["data_preparation"])]
]:

    data.load_features(X, y, version)

    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    try:
        X, y = validate(X, y)
    except Exception as e:
        print(e)
    X, y = load(X, y, version)


if __name__ == "__main__":

    run = prepare_data_pipeline()