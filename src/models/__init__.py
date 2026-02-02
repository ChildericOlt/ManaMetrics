"""
ManaMetrics - Models Package
============================
This package contains all ML and Deep Learning models for MTG card price prediction.

Modules:
    - ml.py: Comparative baseline study (XGBoost, Random Forest, Ridge)
    - deep.py: NLP component (DistilBERT text encoder)
    - hybrid.py: Multi-modal fusion model
"""

from .ml import run_comparative_study, ModelConfig
from .deep import (
    NLPConfig,
    get_tokenizer,
    CardTextEncoder,
    CardTextDataset,
    prepare_text_encoder,
)
from .hybrid import (
    HybridConfig,
    HybridPricePredictor,
    train_hybrid_model,
)

__all__ = [
    # ML Baseline
    "run_comparative_study",
    "ModelConfig",
    # Deep Learning
    "NLPConfig",
    "get_tokenizer",
    "CardTextEncoder",
    "CardTextDataset",
    "prepare_text_encoder",
    # Hybrid Fusion
    "HybridConfig",
    "HybridPricePredictor",
    "train_hybrid_model",
]
