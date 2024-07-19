import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data import preprocess_data  # Replace 'your_module' with the actual module name

# Mock data for testing
@pytest.fixture
def mock_data():
    return pd.DataFrame({
        'sold': [15, 20, 5, 6, 11],
        'rating': [4.5, 4.0, 0.0, 3.5, 5.0],
        'storeName': ['Store A', 'Store A', 'Store B', 'Store A', 'Store C'],
        'lunchTime': ['2022-01-01 12:00:00'] * 5,
        'type': ['A', 'B', 'A', 'C', 'B'],
        'category_name': ['Cat1', 'Cat2', 'Cat1', 'Cat3', 'Cat2'],
        'price': [10, 20, 30, 40, 50],
        'title': ['Title 1', 'Title 2', 'Title 3', 'Title 4', 'Title 5']
    })

# Mocking external dependencies
@patch('data.hydra')
@patch('data.zenml')
@patch('data.RobertaModel')
@patch('data.generate_embeddings')
def test_preprocess_data(mock_generate_embeddings, mock_RobertaModel, mock_zenml, mock_hydra, mock_data):
    
    # Configure mocks
    mock_hydra.compose.return_value.features.all = ['sold', 'rating', 'storeName', 'lunchTime', 'type', 'category_name']
    mock_hydra.compose.return_value.features.target = 'price'
    mock_hydra.compose.return_value.features.numerical = ['sold', 'rating']
    
    mock_zenml.load_artifact.side_effect = [None, None]  # Simulate that no artifacts are loaded
    mock_generate_embeddings.return_value = MagicMock()  
    mock_generate_embeddings.return_value.numpy.return_value = [[1.0, 2.0, 3.0]]  # Example output shape

    # Call the preprocessing function
    df_transformed, y_df = preprocess_data(mock_data)  # Call the fixture directly
    
    # Assertions
    assert df_transformed is not None
    assert isinstance(df_transformed, pd.DataFrame)
    assert not df_transformed.empty
    assert 'title_0' in df_transformed.columns  # Check if title embeddings are added
    assert len(y_df) == 5  # Check if target variable size matches input size
    assert (df_transformed.shape[1] > 0)  # Ensure some features are created

# To run tests, use the command: pytest -v
