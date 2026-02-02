import os
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, lit, current_date, datediff, regexp_extract
from pyspark.sql.types import IntegerType, FloatType, StringType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ETL_Pipeline")

def get_spark_session():
    return SparkSession.builder \
        .appName("ManaMetrics_ETL") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def parse_mana_cost(cost: str, color: str) -> int:
    """UDF to extract devotion for a specific color."""
    if not cost:
        return 0
    return cost.count(f'{{{color}}}')

parse_manacost_udf = udf(parse_mana_cost, IntegerType())

def run_etl(input_path: str, output_path: str):
    spark = get_spark_session()
    logger.info(f"Loading raw data from {input_path}")
    
    # Load JSON
    df = spark.read.option("multiline", "true").json(input_path)
    
    # Selection of core columns
    # Note: Scryfall uses 'prices' object, we need 'usd' or 'eur'
    df_core = df.select(
        col("name"),
        col("mana_cost"),
        col("cmc"),
        col("type_line"),
        col("oracle_text"),
        col("rarity"),
        col("released_at"),
        col("set_name"),
        col("power"),
        col("toughness"),
        col("prices.usd").alias("price_usd")
    )
    
    # 1. Handle NULLs for Power/Toughness
    df_clean = df_core.withColumn("power_num", when(col("power").rlike("^[0-9]+$"), col("power").cast(IntegerType())).otherwise(0)) \
                      .withColumn("toughness_num", when(col("toughness").rlike("^[0-9]+$"), col("toughness").cast(IntegerType())).otherwise(0))
    
    # 2. Filter out cards without prices
    df_clean = df_clean.filter(col("price_usd").isNotNull())
    df_clean = df_clean.withColumn("price_usd", col("price_usd").cast(FloatType()))
    
    # 3. Feature Engineering: Devotion
    for color in ['W', 'U', 'B', 'R', 'G']:
        df_clean = df_clean.withColumn(f"devotion_{color}", parse_manacost_udf(col("mana_cost"), lit(color)))
    
    # 4. Feature Engineering: Card Age
    df_clean = df_clean.withColumn("days_since_release", datediff(current_date(), col("released_at")))
    
    # 5. Feature Engineering: Is Creature?
    df_clean = df_clean.withColumn("is_creature", when(col("type_line").contains("Creature"), 1).otherwise(0))
    
    logger.info(f"Transformations complete. Count: {df_clean.count()}")
    
    # Save to Parquet
    logger.info(f"Saving processed data to {output_path}")
    df_clean.write.mode("overwrite").parquet(output_path)
    
    # 6. Generate Dataset Documentation
    generate_documentation(df_clean, os.path.join(os.path.dirname(output_path), "dataset_schema.md"))

def generate_documentation(df, doc_path):
    """Generates a markdown documentation of the dataset with basic profiling."""
    logger.info(f"Generating dataset documentation at {doc_path}")
    
    schema = df.schema
    rows_count = df.count()
    
    # Basic Profiling for Price
    price_stats = df.select(
        col("price_usd")
    ).summary("count", "mean", "stddev", "min", "max").collect()
    
    stats_dict = {row['summary']: row['price_usd'] for row in price_stats}
    
    doc_content = f"# 📊 Dataset Documentation - ManaMetrics\n\n"
    doc_content += f"**Processed at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    doc_content += f"**Total Records**: {rows_count}\n\n"
    
    doc_content += "## 💰 Price Statistics (Gold Layer)\n\n"
    doc_content += f"- **Count**: {stats_dict.get('count', 'N/A')}\n"
    doc_content += f"- **Mean**: ${float(stats_dict.get('mean', 0)):.2f}\n"
    doc_content += f"- **Min**: ${float(stats_dict.get('min', 0)):.2f}\n"
    doc_content += f"- **Max**: ${float(stats_dict.get('max', 0)):.2f}\n\n"

    doc_content += "## 📋 Schema Definition\n\n"
    doc_content += "| Field | Type | Description |\n"
    doc_content += "| :--- | :--- | :--- |\n"
    
    for field in schema:
        desc = "Target Variable" if field.name == "price_usd" else "ML Feature"
        doc_content += f"| {field.name} | {field.dataType.simpleString()} | {desc} |\n"
    
    with open(doc_path, "w") as f:
        f.write(doc_content)
    logger.info("Documentation generated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ManaMetrics ETL Pipeline")
    parser.add_argument("--input", type=str, default="data/raw/oracle_cards.json", help="Path to raw JSON data")
    parser.add_argument("--output", type=str, default="data/processed/cards.parquet", help="Path to save processed Parquet data")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
    else:
        run_etl(args.input, args.output)
