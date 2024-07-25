import pandas as pd
import pytest
from src.data import TextDataset

# Test each method of the class

def test_text_dataset_length():
    """Tests the length of the TextDataset."""
    text_series = pd.Series(["This is a test.", "Another test data."])
    dataset = TextDataset(text_series)
    assert len(dataset) == 2

def test_text_dataset_getitem():
    """Tests the __getitem__ method of TextDataset."""
    text_series = pd.Series(["First text.", "Second text."])
    dataset = TextDataset(text_series)
    assert dataset[0] == "First text."
    assert dataset[1] == "Second text."

def test_text_dataset_getitem_out_of_bounds():
    """Tests that __getitem__ raises an IndexError for out-of-bounds indices."""
    text_series = pd.Series(["Text 1", "Text 2"])
    dataset = TextDataset(text_series)
    with pytest.raises(IndexError):
        dataset[2] 
