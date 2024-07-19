import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from data import generate_embeddings

# Sample data for testing
sample_texts = pd.Series(["Hello world!", "This is a test.", "Generating embeddings for testing."])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture
def setup_model():
    model = RobertaModel.from_pretrained('roberta-base')
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

@pytest.mark.parametrize("texts,expected_shape", [
    (sample_texts, (len(sample_texts), 768)),  # Assuming embedding size of 768 for RoBERTa
])
def test_generate_embeddings(setup_model, texts, expected_shape):
    embeddings = generate_embeddings(texts, setup_model, device)
    
    assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, but got {embeddings.shape}"
