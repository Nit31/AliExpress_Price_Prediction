import pytest
import pandas as pd
import numpy as np
from data import handle_initial_data

# Sample data for testing
sample_data = {
    'id': [1, 2, 2, 3, 4],
    'shippingCost': ['None', '10.0', '10.0', '5.0', '4.2'],
    'sold': ['100 sold', '200 sold', '200 sold', '50 sold', '1000 sold'],
    'rating': [1, 2, 2, 3, 4],
    'storeName': ['A', 'A', 'A', 'A', 'A']
}
sample_df = pd.DataFrame(sample_data)

@pytest.fixture
def sample():
    return sample_df

def test_handle_initial_data(sample):
    """
    Test functionon sample 
    Args:
        sample (pandas.Dataframe): Input sample
    """
    cleaned_df = handle_initial_data(sample)

    # Check for duplicates in 'id'
    print(cleaned_df.shape)
    assert cleaned_df['id'].nunique() == 4, "Duplicates were not removed correctly."

    # Check that 'shippingCost' column has no None values and the mean is calculated correctly
    assert cleaned_df['shippingCost'].isnull().sum() == 0, "There are still None values in 'shippingCost'."
    
    # Check that 'sold' values are converted to integers correctly
    expected_sold_values = [100, 200, 50, 1000]  # Expected cleaned values
    assert all(cleaned_df['sold'].values == expected_sold_values), "Sold values were not cleaned correctly."
