import pandas as pd
import pytest
import yaml
from unittest.mock import patch, mock_open
from data import read_datastore

def test_read_datastore():
    # Sample DataFrame to return when pd.read_csv is called
    sample_data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['A', 'B', 'C']
    })
    
    # Mock the pd.read_csv and yaml.safe_load methods
    with patch('pandas.read_csv', return_value=sample_data), \
         patch('builtins.open', mock_open(read_data='version: 1.0.0')) as mock_file:

        # Call the function
        df, version = read_datastore()

        # Check that the CSV was read correctly
        pd.testing.assert_frame_equal(df, sample_data)

        # Check that the version is read correctly
        assert version == '1.0.0', "The version should be '1.0.0'."

        # Ensure that the open function was called with the correct file path
        mock_file.assert_called_once_with('configs/data_version.yaml', "r")
