import os
import pytest
import pandas as pd
from data import read_datastore

def test_read_datastore():
    # Mock the `os.getcwd()` function to return the expected current working directory
        # Call the function under test
    df, version = read_datastore()
    
    # Assert that the DataFrame is read from the expected path
    assert df is not None
    assert isinstance(df, pd.DataFrame)