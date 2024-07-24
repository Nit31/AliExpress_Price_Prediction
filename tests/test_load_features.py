import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data import load_features  # Replace with the actual module name

# Sample DataFrames for testing
@pytest.fixture
def sample_data():
    X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    y = pd.DataFrame({'target': [5, 6]})
    return X, y

# Test for load_features function
@patch('zenml.save_artifact')  # Replace with the actual import path for zenml
@patch('zenml.load_artifact')  # Replace with the actual import path for zenml
def test_load_features(mock_load_artifact, mock_save_artifact, sample_data):
    X, y = sample_data
    version = 1

    # Mock the return value of load_artifact to simulate it loading successfully
    mock_load_artifact.return_value = True

    # Call the function
    load_features(X, y, version)

    # Assert that save_artifact was called with the correct parameters
    pd.testing.assert_frame_equal(
        mock_save_artifact.call_args[1]['data'],
        pd.concat([X, y], axis=1)  # Check that the concatenated DataFrame is passed
    )
    assert mock_save_artifact.call_args[1]['name'] == "features_target"
    assert mock_save_artifact.call_args[1]['version'] == str(version)
    assert mock_save_artifact.call_args[1]['tags'] == [str(version)]

    # Assert that load_artifact was called with the correct parameters
    assert mock_load_artifact.call_args[1]['name_or_id'] == "features_target"
    assert mock_load_artifact.call_args[1]['version'] == str(version)

def test_load_features_artifact_not_loaded(sample_data):
    X, y = sample_data
    version = 1

    with patch('zenml.save_artifact') as mock_save_artifact, \
         patch('zenml.load_artifact') as mock_load_artifact:
        
        # Mock load_artifact to return None, simulating an artifact loading failure
        mock_load_artifact.return_value = None
        
        with pytest.raises(Exception, match="Artifact not loaded"):
            load_features(X, y, version)
