import pytest
import torch
from transformers import RobertaTokenizer
from src.data import collate_fn

def test_collate_fn():
    """
    Test that the collate_fn function correctly tokenizes a batch of text and returns a PyTorch tensor.

    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Sample batch of text
    batch = ["first sentence", "second sentence"]

    # Call the collate_fn function
    output = collate_fn(batch)

    # Assert that the output is a dictionary containing PyTorch tensors
    assert 'input_ids' in output
    assert 'attention_mask' in output
    assert isinstance(output['input_ids'], torch.Tensor)
    assert isinstance(output['attention_mask'], torch.Tensor)