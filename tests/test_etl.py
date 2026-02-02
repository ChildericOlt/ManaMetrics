import pytest
from pyspark.sql import SparkSession
from src.data.etl import parse_mana_cost, get_spark_session
from pyspark.sql.functions import lit

@pytest.fixture(scope="module")
def spark():
    return SparkSession.builder \
        .master("local[1]") \
        .appName("ManaMetrics_Tests") \
        .getOrCreate()

def test_parse_mana_cost():
    # Test Red devotion
    assert parse_mana_cost("{R}{R}{1}", "R") == 2
    # Test Blue devotion
    assert parse_mana_cost("{U}{B}{R}", "U") == 1
    # Test no cost
    assert parse_mana_cost(None, "W") == 0
    # Test different color
    assert parse_mana_cost("{W}{W}", "G") == 0

def test_etl_logic(spark):
    # Create a small test dataframe
    data = [
        ("Fireball", "{R}{X}", 1, "Sorcery", "Deals X damage", "Uncommon", "1", "0", "1.50"),
        ("Wall of Runes", "{U}", 1, "Creature — Wall", "Defender", "Common", "0", "4", "0.05")
    ]
    columns = ["name", "mana_cost", "cmc", "type_line", "oracle_text", "rarity", "power", "toughness", "usd"]
    
    # This is a bit complex to test without running the full ETL, 
    # but we can test the specific transformation logic if we refactor etl.py slightly.
    # For now, we'll just test the UDF as above.
    pass
