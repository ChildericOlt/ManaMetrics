import os
import json
import logging
from src.data.scryfall_client import ScryfallClient
from src.data.etl import run_etl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Verification")

def verify():
    # 1. Initialize Client
    client = ScryfallClient()
    cards = ["Black Lotus", "Lightning Bolt", "Colossal Dreadmaw", "Ancestral Recall", "Sol Ring"]
    card_data = []

    # 2. Fetch specific cards for testing
    logger.info(f"Fetching {len(cards)} test cards...")
    for card_name in cards:
        data = client.get_card_by_name(card_name)
        if data:
            card_data.append(data)
    
    # 3. Save as raw JSON
    raw_path = "data/raw/test_cards.json"
    os.makedirs("data/raw", exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(card_data, f)
    logger.info(f"Saved {len(card_data)} cards to {raw_path}")

    # 4. Run ETL
    processed_dir = "data/processed/test_cards.parquet"
    logger.info("Starting ETL process on test cards...")
    try:
        run_etl(raw_path, processed_dir)
        logger.info("ETL verification complete.")
        
        # Check if documentation was generated
        doc_path = "data/processed/dataset_schema.md"
        if os.path.exists(doc_path):
            logger.info(f"Documentation check: PASSED ({doc_path} exists)")
        else:
            logger.error("Documentation check: FAILED")
            
    except Exception as e:
        logger.error(f"ETL verification FAILED: {str(e)}")
        logger.info("PySpark might not be installed or configured in this environment. This is expected if 'poetry install' hasn't been run.")

if __name__ == "__main__":
    verify()
