import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataProfiling")

class DataProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_report(self, output_path: str):
        """Generates a detailed markdown report of the dataset quality."""
        logger.info(f"Generating profiling report at {output_path}")
        
        total_rows = len(self.df)
        report = f"# 📈 Data Profiling Report - ManaMetrics\n\n"
        report += f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Records**: {total_rows}\n\n"

        # 1. Column Completion (Nulls)
        report += "## 📋 Data Completion\n\n"
        report += "| Feature | Null Count | Completion % | Type |\n"
        report += "| :--- | :--- | :--- | :--- |\n"
        
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            comp_pct = (1 - (null_count / total_rows)) * 100
            dtype = str(self.df[col].dtype)
            report += f"| {col} | {null_count} | {comp_pct:.2f}% | {dtype} |\n"
        
        # 2. Price Distribution (Target Feature)
        if 'price_usd' in self.df.columns:
            prices = self.df['price_usd'].dropna()
            report += "\n## 💰 Price Statistics\n\n"
            report += f"- **Mean**: ${prices.mean():.2f}\n"
            report += f"- **Median**: ${prices.median():.2f}\n"
            report += f"- **Std Dev**: ${prices.std():.2f}\n"
            report += f"- **Min**: ${prices.min():.2f}\n"
            report += f"- **Max**: ${prices.max():.2f}\n"

        # 3. Categorical Distributions (Top 5 for Rarity)
        if 'rarity' in self.df.columns:
            report += "\n## ✨ Rarity Distribution\n\n"
            counts = self.df['rarity'].value_counts()
            for rarity, count in counts.items():
                report += f"- **{rarity.capitalize()}**: {count} cards ({(count/total_rows)*100:.1f}%)\n"

        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Profiling report generated successfully.")

def profile_parquet(parquet_path: str, report_path: str):
    """Loads a parquet file and generates a report."""
    try:
        if os.path.isdir(parquet_path):
            df = pd.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)
        profiler = DataProfiler(df)
        profiler.generate_report(report_path)
    except Exception as e:
        logger.error(f"Failed to profile parquet: {str(e)}")

if __name__ == "__main__":
    # Example usage for testing
    # profile_parquet("data/processed/cards.parquet", "data/processed/profiling_report.md")
    pass
