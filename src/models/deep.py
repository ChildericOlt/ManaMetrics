"""
ManaMetrics - Deep Learning NLP Component
==========================================
This script implements the NLP component for MTG card price prediction using DistilBERT.
It handles:
- Text tokenization with MTG-specific tokens
- DistilBERT fine-tuning for text embeddings
- Embedding extraction for downstream fusion

This script is designed to be imported by `hybrid.py` for the multi-modal fusion.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    DistilBertConfig,
)
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ManaMetrics.Deep")


# --- Configuration ---
@dataclass
class NLPConfig:
    """Configuration for the NLP component."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    embedding_dim: int = 768  # DistilBERT hidden size
    freeze_bert: bool = False  # Whether to freeze BERT weights during training
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --- MTG-Specific Tokenizer ---
MTG_SPECIAL_TOKENS = [
    "{W}", "{U}", "{B}", "{R}", "{G}",  # Mana colors
    "{C}", "{X}", "{T}",  # Colorless, X, Tap
    "{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}", "{9}", "{10}",  # Generic mana
    "Prowess", "Trample", "Flying", "Lifelink", "Deathtouch", "Vigilance",  # Keywords
    "Hexproof", "Indestructible", "Haste", "First Strike", "Double Strike",
    "Flash", "Menace", "Reach", "Scry", "Mill", "Exile",
]


def get_tokenizer(config: NLPConfig) -> DistilBertTokenizerFast:
    """Initialize tokenizer with MTG-specific tokens."""
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.model_name)

    # Add MTG-specific tokens
    num_added = tokenizer.add_tokens(MTG_SPECIAL_TOKENS)
    logger.info(f"Added {num_added} MTG-specific tokens to vocabulary.")

    return tokenizer


# --- Card Text Dataset ---
class CardTextDataset(Dataset):
    """PyTorch Dataset for MTG card text."""

    def __init__(
        self,
        texts: List[str],
        targets: Optional[np.ndarray] = None,
        tokenizer: DistilBertTokenizerFast = None,
        max_length: int = 128,
    ):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx] if self.texts[idx] else ""  # DistilBERT can handle None strings

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


# --- NLP Model (DistilBERT Encoder) ---
class CardTextEncoder(nn.Module):
    """
    DistilBERT-based text encoder for MTG card text.
    
    Outputs the [CLS] embedding for downstream fusion.
    """

    def __init__(self, config: NLPConfig, tokenizer: DistilBertTokenizerFast):
        super().__init__()
        self.config = config
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(config.model_name)
        
        # Resize embeddings to account for MTG tokens
        self.bert.resize_token_embeddings(len(tokenizer))

        # Optional: Freeze BERT weights
        if config.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("BERT weights frozen.")

        # Projection layer (optional, for dimensionality reduction)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            [CLS] embedding (batch_size, embedding_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Apply projection
        projected = self.projection(cls_embedding)

        return projected


# --- Standalone Regression Head (for fine-tuning) ---
class TextOnlyRegressor(nn.Module):
    """
    Standalone text-based regressor for initial fine-tuning.
    
    This can be used to pre-train the text encoder before fusion. 
    Prepare a generic BERT to a MTG BERT.
    """

    def __init__(self, config: NLPConfig, tokenizer: DistilBertTokenizerFast):
        super().__init__()
        self.encoder = CardTextEncoder(config, tokenizer)
        self.regressor = nn.Sequential(
            nn.Linear(config.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning log-price prediction."""
        embeddings = self.encoder(input_ids, attention_mask)
        return self.regressor(embeddings).squeeze(-1)


# --- Embedding Extraction Utility ---
def extract_embeddings(
    model: CardTextEncoder,
    dataloader: DataLoader,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract [CLS] embeddings from the encoder for all samples in order to use them for fusion with card statistics features.
    
    Args:
        model: Trained CardTextEncoder.
        dataloader: DataLoader with card texts.
        device: Device to run inference on.
    
    Returns:
        numpy array of embeddings (n_samples, embedding_dim)
    """
    model.eval()
    model.to(device)
    
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# --- Training Utilities ---
def train_text_regressor(
    model: TextOnlyRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: NLPConfig,
    epochs: int = 5,
    lr: float = 2e-5,
) -> TextOnlyRegressor:
    """
    Train the text-only regressor (pre-training for fusion).
    
    Uses MSLE loss for price prediction.
    """
    device = config.device
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Applied on log-transformed prices

    logger.info(f"Training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)

                predictions = model(input_ids, attention_mask)
                val_loss += criterion(predictions, targets).item()

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss / len(train_loader):.4f} - "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )

    return model


# --- Main Entry Point ---
def prepare_text_encoder(
    input_path: str,
    output_dir: str = "models",
    epochs: int = 3,
) -> Tuple[CardTextEncoder, DistilBertTokenizerFast]:
    """
    Prepare and optionally pre-train the text encoder.
    
    Args:
        input_path: Path to processed parquet file.
        output_dir: Directory to save the encoder.
        epochs: Number of pre-training epochs.
    
    Returns:
        Trained encoder and tokenizer.
    """
    config = NLPConfig()
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    texts = df["oracle_text"].fillna("").tolist()
    targets = np.log1p(df["price_usd"].values)

    # Initialize tokenizer and model
    tokenizer = get_tokenizer(config)
    model = TextOnlyRegressor(config, tokenizer)

    # Create datasets
    from sklearn.model_selection import train_test_split

    texts_train, texts_val, targets_train, targets_val = train_test_split(
        texts, targets, test_size=0.2, random_state=42
    )

    train_dataset = CardTextDataset(texts_train, targets_train, tokenizer, config.max_length)
    val_dataset = CardTextDataset(texts_val, targets_val, tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Train
    model = train_text_regressor(model, train_loader, val_loader, config, epochs=epochs)

    # Save
    encoder_path = os.path.join(output_dir, "text_encoder.pt")
    torch.save(model.encoder.state_dict(), encoder_path)
    logger.info(f"Text encoder saved to {encoder_path}")

    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    logger.info(f"Tokenizer saved to {output_dir}/tokenizer")

    return model.encoder, tokenizer


if __name__ == "__main__":
    # Example usage
    # prepare_text_encoder("data/processed/cards.parquet", "models", epochs=3)
    pass
