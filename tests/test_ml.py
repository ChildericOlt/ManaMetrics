import pytest
import pandas as pd
import numpy as np
import os
from src.models.ml import load_and_preprocess, ModelConfig, calculate_binned_mape, run_comparative_study

@pytest.fixture
def mock_data(tmp_path):
    """Create a mock parquet file for testing."""
    df = pd.DataFrame({
        "cmc": [1, 2, 3, 4, 5],
        "rarity": ["common", "uncommon", "rare", "mythic", "common"],
        "power_num": [1, 2, 3, 0, 1],
        "toughness_num": [1, 2, 3, 0, 1],
        "days_since_release": [100, 200, 300, 400, 500],
        "is_creature": [1, 1, 1, 0, 1],
        "devotion_W": [1, 0, 0, 0, 0],
        "devotion_U": [0, 1, 0, 0, 0],
        "devotion_B": [0, 0, 1, 0, 0],
        "devotion_R": [0, 0, 0, 1, 0],
        "devotion_G": [0, 0, 0, 0, 1],
        "price_usd": [0.5, 2.0, 15.0, 120.0, 0.2]
    })
    path = tmp_path / "test_cards.parquet"
    df.to_parquet(path)
    return str(path)

def test_load_and_preprocess(mock_data):
    config = ModelConfig()
    df_processed, le = load_and_preprocess(mock_data, config)
    
    assert len(df_processed) == 5
    assert "rarity" in df_processed.columns
    assert df_processed["rarity"].dtype == np.int64 or df_processed["rarity"].dtype == np.int32
    assert len(le.classes_) == 4 # common, uncommon, rare, mythic

def test_calculate_binned_mape():
    y_true = np.array([0.5, 2.0, 15.0, 120.0]) # Bulk, Mid, Mid, High
    y_pred = np.array([0.6, 2.2, 12.0, 100.0])
    
    metrics = calculate_binned_mape(y_true, y_pred)
    
    assert "mape_bulk" in metrics
    assert "mape_mid" in metrics
    assert "mape_high_end" in metrics
    
    # Check bulk: true 0.5, pred 0.6 -> error 0.1/0.5 = 20%
    assert pytest.approx(metrics["mape_bulk"], 0.01) == 0.20

def test_run_comparative_study_smoke(mock_data, tmp_path):
    """Smoke test to ensure the full pipeline runs without error."""
    output_dir = tmp_path / "models"
    metrics = run_comparative_study(mock_data, str(output_dir))
    
    assert "xgboost" in metrics
    assert "random_forest" in metrics
    assert "ridge" in metrics
    assert os.path.exists(output_dir / "xgboost.joblib")
    assert os.path.exists(output_dir / "rarity_encoder.joblib")
