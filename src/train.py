"""
ManaMetrics - Central Training Orchestrator
===========================================
This script provides a unified entry point to run the ManaMetrics training pipeline:
1. Baseline Study (ml.py)
2. Hybrid Model Training (hybrid.py)
3. Deep Learning Encoder (deep.py)

Usage:
    python3 src/train.py --baseline
    python3 src/train.py --deep
    python3 src/train.py --hybrid
    python3 src/train.py --all
"""

import argparse
import logging
import os
from src.models.ml import run_comparative_study
from src.models.deep import prepare_text_encoder
from src.models.hybrid import train_hybrid_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ManaMetrics.Train")

def main():
    parser = argparse.ArgumentParser(description="ManaMetrics Training Orchestrator")
    parser.add_argument("--baseline", action="store_true", help="Run the comparative baseline study")
    parser.add_argument("--deep", action="store_true", help="Fine-tune the DistilBERT text encoder standalone")
    parser.add_argument("--hybrid", action="store_true", help="Train the hybrid multi-modal model")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline (Baseline + Deep + Hybrid)")
    parser.add_argument("--data", type=str, default="data/processed/test_cards.parquet", help="Path to processed parquet data")
    parser.add_argument("--output", type=str, default="models", help="Directory to save models and results")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for hybrid/deep training")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}. Please run the ETL pipeline first.")
        return

    if args.baseline or args.all:
        logger.info("Starting Comparative Baseline Study...")
        run_comparative_study(args.data, args.output)
        logger.info(f"Baseline results saved in: {args.output}")

    if args.deep or args.all:
        logger.info("Starting Standalone Deep Learning Training (BERT Fine-tuning)...")
        prepare_text_encoder(args.data, args.output, epochs=args.epochs)
        logger.info(f"Text encoder saved in: {args.output}/text_encoder.pt")

    if args.hybrid or args.all:
        logger.info("Starting Hybrid Multi-Modal Training...")
        train_hybrid_model(args.data, args.output, epochs=args.epochs)
        logger.info(f"Hybrid model saved in: {args.output}/hybrid_model.pt")

    if not (args.baseline or args.deep or args.hybrid or args.all):
        parser.print_help()

if __name__ == "__main__":
    main()
