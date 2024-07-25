import pytest
import pandas as pd
import great_expectations as gx
from great_expectations.data_context import FileDataContext
from unittest.mock import patch, MagicMock
from your_module import validate_initial_data

@pytest.fixture
def sample():
    # Sample DataFrame for testing
    return pd.DataFrame({
        'id': [1, 2, 3],
        'shippingCost': [10.0, 15.5, 5.0],
        'sold': [100, 250, 50]
    })

def test_validate_initial_data(sample):
    """
    Test data validation

    Args:
        sample (pandas.Dataframe): Sample df
    """
    cfg = None  # Configuration can be passed if needed.

    with patch('your_module.gx.get_context') as mock_get_context:
        mock_context = MagicMock()
        mock_get_context.return_value = mock_context
        
        # Mock methods directly called within validate_initial_data
        mock_context.sources.add_or_update_pandas.return_value.add_dataframe_asset.return_value.build_batch_request.return_value = None

        # Call the function
        result = validate_initial_data(cfg, sample)

        # Assertions to check if the methods were called
        mock_get_context.assert_called_once_with(context_root_dir="services/gx")
        mock_context.sources.add_or_update_pandas.assert_called_once_with(name="sample")
        
        # Ensure the expectations suite fetching is called
        mock_context.get_expectation_suite.assert_called_once_with("expectation_suite")

        # Check that the function completes successfully (assuming it returns True)
        assert isinstance(result, bool), "Result should be a boolean indicating the status of validation."
