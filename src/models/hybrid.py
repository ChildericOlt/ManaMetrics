"""
ManaMetrics - Hybrid Multi-Modal Fusion Model
==============================================
This script implements the multi-modal fusion for MTG card price prediction.
It combines:
- Tabular features via MLP (from ml.py approach)
- Text embeddings via DistilBERT (from deep.py)

Architecture:
- Late Fusion: Concatenate text [CLS] embedding with tabular MLP output
- Loss: MSLE (Mean Squared Logarithmic Error) for price variance handling
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Import components from deep.py
from .deep import (
    NLPConfig,
    get_tokenizer,
    CardTextEncoder,
    CardTextDataset,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ManaMetrics.Hybrid")


# --- Configuration ---
@dataclass
class HybridConfig:
    """Configuration for the hybrid model."""
    # Tabular
    tabular_features: List[str] = field(default_factory=lambda: [
        "cmc", "rarity", "power_num", "toughness_num",
        "days_since_release", "is_creature",
        "devotion_W", "devotion_U", "devotion_B", "devotion_R", "devotion_G",
    ])
    tabular_hidden_dim: int = 128
    
    # NLP
    nlp_config: NLPConfig = field(default_factory=NLPConfig)
    
    # Fusion
    fusion_hidden_dim: int = 256
    dropout: float = 0.3
    
    # Training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --- Tabular MLP Component ---
class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular features.
    
    Processes numeric and encoded categorical features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.mlp(x)


# --- Hybrid Fusion Model ---
class HybridPricePredictor(nn.Module):
    """
    Hybrid Multi-Modal Price Predictor.
    
    Architecture:
    1. Text Encoder (DistilBERT) -> [CLS] embedding (768d)
    2. Tabular MLP -> Feature vector (64d)
    3. Late Fusion: Concatenate both
    4. Regression Head -> Log-price prediction
    """

    def __init__(
        self,
        config: HybridConfig,
        tokenizer,
        num_tabular_features: int,
    ):
        super().__init__()
        self.config = config

        # Text Encoder
        self.text_encoder = CardTextEncoder(config.nlp_config, tokenizer)
        text_dim = config.nlp_config.embedding_dim  # 768

        # Tabular MLP
        tabular_output_dim = 64
        self.tabular_mlp = TabularMLP(
            input_dim=num_tabular_features,
            hidden_dim=config.tabular_hidden_dim,
            output_dim=tabular_output_dim,
        )

        # Fusion Layer
        fusion_input_dim = text_dim + tabular_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.fusion_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Regression Head
        self.regressor = nn.Linear(config.fusion_hidden_dim // 2, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized text (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            tabular_features: Numeric features (batch_size, num_features)
        
        Returns:
            Log-price predictions (batch_size,)
        """
        # Text encoding
        text_emb = self.text_encoder(input_ids, attention_mask)

        # Tabular encoding
        tabular_emb = self.tabular_mlp(tabular_features)

        # Late Fusion
        fused = torch.cat([text_emb, tabular_emb], dim=1)
        fused = self.fusion(fused)

        # Regression
        output = self.regressor(fused).squeeze(-1)

        return output


# --- Hybrid Dataset ---
class HybridDataset(Dataset):
    """PyTorch Dataset combining text and tabular features."""

    def __init__(
        self,
        texts: List[str],
        tabular: np.ndarray,
        targets: np.ndarray,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.tabular = tabular
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx] if self.texts[idx] else ""

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tabular": torch.tensor(self.tabular[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


# --- MSLE Loss ---
class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error Loss."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Since we're already working with log-transformed prices (log1p),
        this effectively becomes MSE on the log scale.
        """
        return nn.functional.mse_loss(predictions, targets)


# --- Training ---
def train_hybrid_model(
    input_path: str,
    output_dir: str = "models",
    epochs: int = 10,
) -> HybridPricePredictor:
    """
    Train the hybrid multi-modal model.
    
    Args:
        input_path: Path to processed parquet file.
        output_dir: Directory to save the model.
        epochs: Number of training epochs.
    
    Returns:
        Trained model.
    """
    config = HybridConfig(epochs=epochs)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)

    # Prepare tabular features
    label_encoder = LabelEncoder()
    df["rarity_encoded"] = label_encoder.fit_transform(df["rarity"].astype(str))

    # Replace rarity with encoded version in features list
    tabular_cols = [
        c if c != "rarity" else "rarity_encoded" for c in config.tabular_features
    ]
    
    # Handle missing columns gracefully
    available_cols = [c for c in tabular_cols if c in df.columns]
    
    scaler = StandardScaler()
    tabular_data = scaler.fit_transform(df[available_cols].values)

    # Prepare text and targets
    texts = df["oracle_text"].fillna("").tolist()
    targets = np.log1p(df["price_usd"].values)

    # Split data
    (
        texts_train, texts_val,
        tabular_train, tabular_val,
        targets_train, targets_val,
    ) = train_test_split(
        texts, tabular_data, targets,
        test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = get_tokenizer(config.nlp_config)
    model = HybridPricePredictor(config, tokenizer, num_tabular_features=len(available_cols))

    # Create datasets
    train_dataset = HybridDataset(
        texts_train, tabular_train, targets_train, tokenizer, config.nlp_config.max_length
    )
    val_dataset = HybridDataset(
        texts_val, tabular_val, targets_val, tokenizer, config.nlp_config.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Training setup
    device = config.device
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = MSLELoss()

    logger.info(f"Training Hybrid Model on {device} for {epochs} epochs...")
    logger.info(f"Text features: oracle_text | Tabular features: {available_cols}")

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular = batch["tabular"].to(device)
            targets_batch = batch["target"].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask, tabular)
            loss = criterion(predictions, targets_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tabular = batch["tabular"].to(device)
                targets_batch = batch["target"].to(device)

                predictions = model(input_ids, attention_mask, tabular)
                val_loss += criterion(predictions, targets_batch).item()

                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(targets_batch.cpu().numpy())

        # Calculate metrics on original scale
        preds_orig = np.expm1(np.array(all_preds))
        targets_orig = np.expm1(np.array(all_targets))
        
        rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
        mape = mean_absolute_percentage_error(targets_orig, preds_orig)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"RMSE: ${rmse:.2f} - MAPE: {mape:.1%}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "hybrid_model.pt"))
            logger.info("  -> Best model saved!")

    # Save preprocessors
    joblib.dump(label_encoder, os.path.join(output_dir, "hybrid_label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "hybrid_scaler.joblib"))
    tokenizer.save_pretrained(os.path.join(output_dir, "hybrid_tokenizer"))

    logger.info(f"\nHybrid model training complete. Best Val Loss: {best_val_loss:.4f}")

    return model


if __name__ == "__main__":
    # Example usage
    # train_hybrid_model("data/processed/cards.parquet", "models", epochs=10)
    pass
