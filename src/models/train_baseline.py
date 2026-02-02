import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BaselineML")

def train_baseline(input_path: str, model_output_path: str):
    logger.info(f"Loading data from {input_path}")
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        # Fallback if we are using the test JSON converted to pandas
        df = pd.read_json(input_path)
    
    # Selection of features for the baseline (Tabular only as per plan)
    features = [
        "cmc", "rarity", "power_num", "toughness_num", 
        "days_since_release", "is_creature",
        "devotion_W", "devotion_U", "devotion_B", "devotion_R", "devotion_G"
    ]
    target = "price_usd"
    
    # Filter only relevant columns
    df = df[features + [target]]
    
    # Preprocessing: Label Encoding for Rarity
    le = LabelEncoder()
    df["rarity"] = le.fit_transform(df["rarity"].astype(str))
    
    # Split Data
    X = df[features]
    y = df[target]
    
    # Log transform target due to high variance (common in price prediction)
    y_log = np.log1p(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Train XGBoost
    logger.info("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predictions
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_test_orig = np.expm1(y_test)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
    mape = mean_absolute_percentage_error(y_test_orig, preds)
    
    logger.info(f"Baseline Results:")
    logger.info(f"RMSE: ${rmse:.2f}")
    logger.info(f"MAPE: {mape:.2%}")
    
    # Save Model and Encoders
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    joblib.dump(le, model_output_path.replace(".joblib", "_encoder.joblib"))
    
    logger.info(f"Model saved to {model_output_path}")
    
    return model, features

if __name__ == "__main__":
    # Example usage for verification
    # train_baseline("data/processed/cards.parquet", "models/baseline_xgboost.joblib")
    pass
