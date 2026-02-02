import pytest
import torch
import os
from src.models.deep import NLPConfig, get_tokenizer, CardTextEncoder, CardTextDataset, MTG_SPECIAL_TOKENS
from transformers import DistilBertTokenizerFast

def test_tokenizer_mtg_tokens():
    config = NLPConfig()
    tokenizer = get_tokenizer(config)
    
    # Check if some MTG tokens are in the vocab
    for token in ["{W}", "Prowess", "Trample"]:
        # Tokenizer might lowercase them if it's an uncased model
        assert token in tokenizer.get_vocab() or token.lower() in tokenizer.get_vocab()
    
    # Check if it tokenizes correctly
    text = "{W}: Gain Lifelink."
    tokens = tokenizer.tokenize(text)
    # For uncased models, it will be '{w}'
    assert "{W}" in tokens or "{w}" in tokens

def test_card_text_encoder_dimensions():
    config = NLPConfig(device="cpu")
    tokenizer = get_tokenizer(config)
    encoder = CardTextEncoder(config, tokenizer)
    
    # Create mock batch
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    
    output = encoder(input_ids, attention_mask)
    
    assert output.shape == (batch_size, 768) # 768 is DistilBERT hidden size

def test_card_text_dataset():
    texts = ["Card 1 text", "Card 2 with {U}"]
    targets = [1.0, 2.0]
    config = NLPConfig()
    tokenizer = get_tokenizer(config)
    
    dataset = CardTextDataset(texts, targets, tokenizer, max_length=64)
    
    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "target" in item
    assert item["input_ids"].shape == (64,)
