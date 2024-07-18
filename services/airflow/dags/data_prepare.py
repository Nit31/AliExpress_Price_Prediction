import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# Go up two levels to get the parent directory (which contains 'src')
# Add the 'src' folder to the Python path
# src_path = '.'
# sys.path.append(str(src_path))
# from src import data
import data
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
def load(X:pd.DataFrame, y:pd.DataFrame, version: int)-> Tuple[
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