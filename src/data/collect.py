import os
import logging
from src.data.scryfall_client import ScryfallClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataCollection")

def collect_bulk_data(data_type: str = "oracle_cards", target_dir: str = "data/raw"):
    """
    Downloads bulk data from Scryfall.
    
    Args:
        data_type: Type of data to download ('oracle_cards', 'unique_artwork', 'default_cards', 'all_cards')
        target_dir: Directory to save the raw JSON.
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{data_type}.json")
    
    client = ScryfallClient()
    url = client.get_bulk_data_url(data_type)
    
    if url:
        logger.info(f"Found {data_type} bulk data URL: {url}")
        client.download_bulk_data(url, target_path)
        logger.info(f"Raw data saved to {target_path}")
    else:
        logger.error(f"Could not find {data_type} bulk data URL.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ManaMetrics Data Collection")
    parser.add_argument("--type", type=str, default="oracle_cards", 
                        choices=["oracle_cards", "unique_artwork", "default_cards", "all_cards"],
                        help="Scryfall bulk data type to download")
    parser.add_argument("--output", type=str, default="data/raw", help="Directory to save raw JSON")
    
    args = parser.parse_args()
    collect_bulk_data(args.type, args.output)
