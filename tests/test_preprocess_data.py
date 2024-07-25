import pytest
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
from transformers import RobertaModel
from src.data import preprocess_data

# Mock configuration class to simulate the expected 'cfg' structure
class MockConfig:
    class zenml:
        class features:
            target = 'target'
            all = ['feature1', 'feature2', 'year', 'month', 'type', 'category_name', 'title']
            numerical = ['feature1', 'feature2']

    data_version = MagicMock(version=1)  # Set the version to 1

# Sample DataFrame for testing
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'lunchTime': ['2023-01-01 12:30:00', '2023-01-02 13:30:00'],
        'feature1': [10, 20],
        'feature2': [1.5, 2.5],
        'type': ['A', 'B'],
        'category_name': ['cat1', 'cat2'],
        'target': [100, 200],
        'title': ['Title 1', 'Title 2']
    })

# Mock function to simulate the generate_embeddings function
def mock_generate_embeddings(titles, model, device):
    return torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # Mocking embeddings

# Tests for preprocess_data function
def test_preprocess_data_with_target(sample_df):
    """
    Test preprocessing of the data

    Args:
        sample_df (pandas.Dataframe): Sample df
    """
    cfg = MockConfig()
    global generate_embeddings
    generate_embeddings = mock_generate_embeddings  # Substitute the actual function with the mock
    
    with MagicMock() as mock_zenml:
        mock_zenml.load_artifact = MagicMock(side_effect=[None, None, None])  # Load preprocessor, target_preprocessor, roberta_model
        mock_zenml.save_artifact = MagicMock()
        preprocess_data.__globals__['zenml'] = mock_zenml  # Allow the function to access the mock

        df_transformed, y_transformed = preprocess_data(sample_df, cfg, skip_target=False)

        # Check the transformed DataFrame structure
        assert df_transformed is not None
        assert isinstance(df_transformed, pd.DataFrame)

        # Test if the target is transformed and included
        assert y_transformed is not None
        assert isinstance(y_transformed, pd.DataFrame)
        assert y_transformed.shape[1] == 1  # One column for the target
        assert 'price' in y_transformed.columns
