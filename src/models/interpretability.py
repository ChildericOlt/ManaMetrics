"""
ManaMetrics - Model Interpretability Module
==========================================
Centralizes all interpretability analysis (SHAP, feature importance, etc.)

This module extracts the interpretability logic from ml.py to provide:
- SHAP analysis for tree-based and linear models
- Feature importance visualization
- Extensible framework for additional XAI methods (Captum, LIME, etc.)
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import shap
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ManaMetrics.Interpretability")


# --- SHAP Analysis ---
def run_shap_analysis(
    model: Any, X_test: pd.DataFrame, model_name: str, output_dir: str
) -> None:
    """Generate SHAP summary plot for feature importance.
    
    Args:
        model: Trained model (XGBoost, RandomForest, Ridge, etc.)
        X_test: Test features
        model_name: Name of the model for file naming
        output_dir: Directory to save SHAP values
    """
    logger.info(f"Running SHAP analysis for {model_name}...")
    
    try:
        if isinstance(model, (xgb.XGBRegressor, RandomForestRegressor)):
            explainer = shap.TreeExplainer(model)
        else:
            # For linear models like Ridge
            explainer = shap.LinearExplainer(model, X_test)

        shap_values = explainer.shap_values(X_test)

        # Save SHAP values for later visualization
        shap_output_path = os.path.join(output_dir, f"shap_values_{model_name}.npy")
        np.save(shap_output_path, shap_values)
        logger.info(f"SHAP values saved to {shap_output_path}")
        
    except Exception as e:
        logger.warning(f"SHAP analysis failed for {model_name}: {e}")


def load_shap_values(model_name: str, input_dir: str = "models") -> np.ndarray:
    """Load previously saved SHAP values.
    
    Args:
        model_name: Name of the model
        input_dir: Directory containing SHAP values
        
    Returns:
        numpy array of SHAP values
    """
    shap_path = os.path.join(input_dir, f"shap_values_{model_name}.npy")
    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"SHAP values not found: {shap_path}")
    
    logger.info(f"Loading SHAP values from {shap_path}")
    return np.load(shap_path)


def visualize_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    model_name: str,
    output_dir: str = "models",
    max_display: int = 20
) -> None:
    """Generate and save SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X_test: Test features
        model_name: Name of the model
        output_dir: Directory to save plot
        max_display: Maximum number of features to display
    """
    import matplotlib.pyplot as plt
    
    logger.info(f"Generating SHAP summary plot for {model_name}...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    
    plot_path = os.path.join(output_dir, f"shap_summary_{model_name}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logger.info(f"SHAP summary plot saved to {plot_path}")


def generate_feature_importance_report(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    output_dir: str = "models"
) -> str:
    """Generate a markdown report with feature importance analysis.
    
    Args:
        models: Dictionary of trained models {name: model}
        X_test: Test features
        output_dir: Directory to save report
        
    Returns:
        Path to the generated report
    """
    logger.info("Generating feature importance report...")
    
    report_lines = [
        "# 🔍 ManaMetrics - Feature Importance Report\n",
        "This report analyzes which features are most important for price prediction.\n",
        "## SHAP Analysis Results\n"
    ]
    
    for model_name in models.keys():
        try:
            shap_values = load_shap_values(model_name, output_dir)
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            
            report_lines.append(f"\n### {model_name.upper()}\n")
            report_lines.append("| Feature | SHAP Importance |\n")
            report_lines.append("|---------|----------------|\n")
            
            for _, row in feature_importance.head(10).iterrows():
                report_lines.append(f"| {row['feature']} | {row['importance']:.4f} |\n")
                
        except FileNotFoundError:
            report_lines.append(f"\n### {model_name.upper()}\n")
            report_lines.append("*SHAP values not found. Run training with SHAP analysis first.*\n")
    
    report_path = os.path.join(output_dir, "feature_importance_report.md")
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    
    logger.info(f"Feature importance report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    """
    Example usage:
    
    # After training models with ml.py:
    import joblib
    from src.models.interpretability import load_shap_values, visualize_shap_summary
    
    # Load a model
    model = joblib.load("models/xgboost.joblib")
    
    # Load test data
    import pandas as pd
    df = pd.read_parquet("data/processed/cards.parquet")
    X_test = df[['cmc', 'rarity', 'power_num', ...]].head(1000)
    
    # Visualize SHAP
    shap_values = load_shap_values("xgboost")
    visualize_shap_summary(shap_values, X_test, "xgboost")
    """
    logger.info("Interpretability module loaded. Use functions to analyze model predictions.")
