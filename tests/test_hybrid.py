import pytest
import torch
import numpy as np
from src.models.hybrid import HybridConfig, HybridPricePredictor, TabularMLP, HybridDataset
from src.models.deep import get_tokenizer

def test_tabular_mlp_dimensions():
    input_dim = 11
    output_dim = 64
    batch_size = 8
    
    mlp = TabularMLP(input_dim=input_dim, hidden_dim=32, output_dim=output_dim)
    x = torch.randn(batch_size, input_dim)
    output = mlp(x)
    
    assert output.shape == (batch_size, output_dim)

def test_hybrid_predictor_forward():
    config = HybridConfig(device="cpu")
    tokenizer = get_tokenizer(config.nlp_config)
    num_tabular = 11
    
    model = HybridPricePredictor(config, tokenizer, num_tabular_features=num_tabular)
    
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    tabular = torch.randn(batch_size, num_tabular)
    
    output = model(input_ids, attention_mask, tabular)
    
    assert output.shape == (batch_size,)

def test_hybrid_dataset():
    texts = ["Text 1", "Text 2"]
    tabular = np.random.randn(2, 5)
    targets = np.array([1.0, 2.0])
    config = HybridConfig()
    tokenizer = get_tokenizer(config.nlp_config)
    
    dataset = HybridDataset(texts, tabular, targets, tokenizer, max_length=32)
    
    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "tabular" in item
    assert "target" in item
    assert item["tabular"].shape == (5,)
