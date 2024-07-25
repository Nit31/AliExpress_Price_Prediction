import numpy as np
import pytest
from unittest.mock import patch
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.binary import BinaryEncoder 
from data import FixedBinsBinaryEncoder

# Mock BinaryEncoder to test FixedBinsBinaryEncoder
class MockBinaryEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, X, y=None):
        self.classes_ = np.unique(X)
        return self

    def transform(self, X):
        num_bits = len(self.classes_) * 2  # Just to simulate; adjust based on actual encoding
        # Create a mock binary matrix
        return np.random.randint(0, 2, size=(len(X), num_bits))

# Replace BinaryEncoder with MockBinaryEncoder for testing
@pytest.fixture
def fixed_bins_binary_encoder():
    """
    Fixture for FixedBinsBinaryEncoder with a mocked BinaryEncoder.

    """
    with patch('category_encoders.binary.BinaryEncoder', MockBinaryEncoder):
        yield FixedBinsBinaryEncoder(max_bins=4)  # Example with max_bins set to 4


def test_fit(fixed_bins_binary_encoder):
    """
    Test fit method of encoder

    Args:
        fixed_bins_binary_encoder (FixedBinsBinaryEncoder): fixture of FixedBinsBinaryEncoder
    """
    # Test that fit does not raise any exceptions
    X = np.array(['a', 'b', 'c'])
    encoder = fixed_bins_binary_encoder.fit(X)
    assert encoder is fixed_bins_binary_encoder


def test_transform(fixed_bins_binary_encoder):
    """
    Test transform method of encoder
    Args:
        fixed_bins_binary_encoder (FixedBinsBinaryEncoder): fixture of FixedBinsBinaryEncoder
    """
    # Test that transform works correctly when within max_bins
    X = np.array(['a', 'b', 'c'])
    transformed = fixed_bins_binary_encoder.fit_transform(X)
    
    assert transformed.shape == (3, 4)  # Check shape after transformation

    
def test_transform_exceeds_max_bins(fixed_bins_binary_encoder):
    """
    Specially exceed maxbins to test function in error case
    Args:
        fixed_bins_binary_encoder (FixedBinsBinaryEncoder): fixture of FixedBinsBinaryEncoder
    """
    # Test that transform raises ValueError when exceeding max_bins
    fixed_bins_binary_encoder.max_bins = 2  # Set max_bins lower
    X = np.array(['a', 'b', 'c', 'd'])  # This should create more than 2 bits
    fixed_bins_binary_encoder.fit(X)

    with pytest.raises(ValueError, match="Number of required bits"):
        fixed_bins_binary_encoder.transform(X)
