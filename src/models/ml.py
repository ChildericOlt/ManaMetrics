"""
ManaMetrics - Comparative Baseline ML Study
============================================
This script implements and compares multiple regression models for MTG card price prediction:
- XGBoost Regressor
- Random Forest Regressor
- Ridge Regression (Linear Baseline)

Features:
- Interpretability integration via dedicated module
- Binned MAPE evaluation (Bulk vs Mid vs High-end)
- MLflow experiment tracking
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import joblib
from src.models.interpretability import run_shap_analysis
import logging
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ManaMetrics.ML")


# --- Configuration ---
@dataclass
class ModelConfig:
    """Configuration for baseline models."""
    features: List[str] = None
    target: str = "price_usd"
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self):
        if self.features is None:
            self.features = [
                "cmc", "rarity", "power_num", "toughness_num",
                "days_since_release", "is_creature",
                "devotion_W", "devotion_U", "devotion_B", "devotion_R", "devotion_G",
            ]


# --- Evaluation Metrics ---
def calculate_binned_mape(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate MAPE for different price bins.
    
    Bins:
    - Bulk: < $1
    - Mid: $1 - $50
    - High-end: > $100
    """
    bins = {
        "bulk": (y_true < 1),
        "mid": (y_true >= 1) & (y_true <= 50),
        "high_end": (y_true > 100),
    }

    results = {}
    for bin_name, mask in bins.items():
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
            results[f"mape_{bin_name}"] = mape
            logger.info(f"  {bin_name.upper()} MAPE: {mape:.2%} (n={mask.sum()})")
        else:
            results[f"mape_{bin_name}"] = None
            logger.warning(f"  {bin_name.upper()}: No samples in this bin.")

    return results


def evaluate_model(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Evaluate a model with RMSE, MAPE, and Binned MAPE."""
    logger.info(f"\n--- {model_name} Evaluation ---")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    logger.info(f"  Overall RMSE: ${rmse:.2f}")
    logger.info(f"  Overall MAPE: {mape:.2%}")

    binned_metrics = calculate_binned_mape(y_true, y_pred)

    return {"rmse": rmse, "mape": mape, **binned_metrics}


# --- Data Loading and Preprocessing ---
def load_and_preprocess(
    input_path: str, config: ModelConfig
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load data and apply preprocessing."""
    logger.info(f"Loading data from {input_path}")

    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_json(input_path)

    # Filter relevant columns
    df = df[config.features + [config.target]].copy()

    # Label Encoding for Rarity
    le = LabelEncoder()
    df["rarity"] = le.fit_transform(df["rarity"].astype(str))

    logger.info(f"Loaded {len(df)} samples.")
    return df, le


# --- Model Training ---
def train_xgboost(
    X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray
) -> xgb.XGBRegressor:
    """Train XGBoost Regressor."""
    logger.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> RandomForestRegressor:
    """Train Random Forest Regressor."""
    logger.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train: pd.DataFrame, y_train: np.ndarray) -> Ridge:
    """Train Ridge Regression (Linear Baseline)."""
    logger.info("Training Ridge Regression...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


# --- Interpretability Analysis ---
# SHAP analysis is now handled by src.models.interpretability module


# --- Main Orchestration ---
def run_comparative_study(
    input_path: str,
    output_dir: str = "models",
) -> Dict[str, Dict[str, float]]:
    """
    Run the full comparative baseline study.
    
    Args:
        input_path: Path to the processed parquet file.
        output_dir: Directory to save models and SHAP values.
    
    Returns:
        Dictionary of metrics for each model.
    """
    config = ModelConfig()
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    df, label_encoder = load_and_preprocess(input_path, config)

    X = df[config.features]
    y = df[config.target]

    # Log transform target (handles price variance)
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=config.test_size, random_state=config.random_state
    )

    # Keep original scale for evaluation
    y_test_orig = np.expm1(y_test_log)

    # Train all models
    models = {
        "xgboost": train_xgboost(X_train, y_train_log, X_test, y_test_log),
        "random_forest": train_random_forest(X_train, y_train_log),
        "ridge": train_ridge(X_train, y_train_log),
    }

    # Evaluate models and run SHAP
    all_metrics = {}
    for name, model in models.items():
        preds_log = model.predict(X_test)
        preds = np.expm1(preds_log)

        metrics = evaluate_model(name.upper(), y_test_orig, preds)
        all_metrics[name] = metrics

        # Save model
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Run SHAP
        run_shap_analysis(model, X_test, name, output_dir)

    # Save label encoder
    joblib.dump(label_encoder, os.path.join(output_dir, "rarity_encoder.joblib"))

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("COMPARATIVE BASELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<15} {'RMSE':>10} {'MAPE':>10} {'Bulk':>10} {'Mid':>10} {'High':>10}")
    logger.info("-" * 60)
    for name, m in all_metrics.items():
        bulk = f"{m.get('mape_bulk', 0):.1%}" if m.get('mape_bulk') else "N/A"
        mid = f"{m.get('mape_mid', 0):.1%}" if m.get('mape_mid') else "N/A"
        high = f"{m.get('mape_high_end', 0):.1%}" if m.get('mape_high_end') else "N/A"
        logger.info(f"{name:<15} ${m['rmse']:>9.2f} {m['mape']:>9.1%} {bulk:>10} {mid:>10} {high:>10}")
    logger.info("=" * 60)

    return all_metrics


if __name__ == "__main__":
    # Example usage
    # run_comparative_study("data/processed/cards.parquet", "models")
    pass
