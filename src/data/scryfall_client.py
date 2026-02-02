import time
import requests
import json
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScryfallClient")

class ScryfallClient:
    """
    Client for interacting with the Scryfall API.
    Handles rate limiting as requested by Scryfall (50-100ms between requests).
    """
    BASE_URL = "https://api.scryfall.com"

    def __init__(self, delay_ms: int = 100):
        self.delay = delay_ms / 1000.0
        self.last_request_time = 0

    def _wait_for_rate_limit(self):
        """Ensures we don't hit the API too fast."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()

    def get_bulk_data_url(self, data_type: str = "oracle_cards") -> Optional[str]:
        """
        Fetches the download URL for bulk data.
        Types: 'oracle_cards', 'unique_artwork', 'default_cards', 'all_cards'
        """
        logger.info(f"Fetching bulk data URL for type: {data_type}")
        self._wait_for_rate_limit()
        response = requests.get(f"{self.BASE_URL}/bulk-data")
        if response.status_code == 200:
            data = response.json()
            for item in data.get("data", []):
                if item.get("type") == data_type:
                    return item.get("download_uri")
        logger.error(f"Failed to fetch bulk data URL. Status: {response.status_code}")
        return None

    def download_bulk_data(self, url: str, target_path: str):
        """Downloads the bulk data JSON to a file."""
        logger.info(f"Downloading bulk data from {url} to {target_path}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download completed.")
        else:
            logger.error(f"Download failed. Status: {response.status_code}")

    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """Fetches a single card by its exact name."""
        logger.info(f"Fetching card: {name}")
        self._wait_for_rate_limit()
        params = {"exact": name}
        response = requests.get(f"{self.BASE_URL}/cards/named", params=params)
        if response.status_code == 200:
            return response.json()
        return None

if __name__ == "__main__":
    # Quick test
    client = ScryfallClient()
    url = client.get_bulk_data_url("oracle_cards")
    print(f"Bulk Data URL: {url}")
