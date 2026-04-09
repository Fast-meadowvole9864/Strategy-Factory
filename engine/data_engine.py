import polars as pl
import logging
import os
import glob
from tqdm import tqdm
from typing import Dict

# Configure Module Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Anchor debug logs under the repo reports tree instead of the caller's cwd.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, "reports", "_logs")
os.makedirs(log_dir, exist_ok=True)

if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(log_dir, "engine_debug.log"))
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

def load_vault_data(vault_dir: str, timeframe: str = "15m") -> Dict[str, pl.DataFrame]:
    """
    Loads all Parquet files for a specific timeframe from a given Vault directory.
    Calculates logarithmic returns and structural log arrays.
    
    Args:
        vault_dir (str): Path to the Vault directory (e.g., 'Vault_1_InSample').
        timeframe (str): Timeframe to load (e.g., '15m' or '1m').
        
    Returns:
        Dict[str, pl.DataFrame]: Dictionary mapping asset symbols to Polars DataFrames.
    """
    data_dict = {}
    
    if not os.path.exists(vault_dir):
        logger.error(f"Vault directory not found: {vault_dir}")
        return data_dict

    # Find parquet files inside Vault_X/asset/timeframe/
    # Using glob: Vault_1_InSample/*/15m/*.parquet
    search_pattern = os.path.join(vault_dir, "*", timeframe, "*.parquet")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        logger.info(f"No parquet files found in {vault_dir} for timeframe {timeframe}.")
        return data_dict

    logger.info(f"Loading {len(file_paths)} assets from {vault_dir} ({timeframe})...")
    
    # Process files
    for filepath in tqdm(file_paths, desc=f"Loading {timeframe} data", unit="file"):
        asset_dir = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        
        try:
            logger.debug(f"Processing {filepath}")
            # Load Data
            df = pl.read_parquet(filepath)
            
            # Tag the asset for structural isolation
            df = df.with_columns(pl.lit(asset_dir).alias("symbol"))
            
            # Handle missing data (forward fill then drop remaining)
            df = df.with_columns([
                pl.col("open").fill_null(strategy="forward"),
                pl.col("high").fill_null(strategy="forward"),
                pl.col("low").fill_null(strategy="forward"),
                pl.col("close").fill_null(strategy="forward"),
                pl.col("volume").fill_null(strategy="forward")
            ])
            df = df.drop_nulls(subset=["open", "high", "low", "close"])
            
            # Calculate Contract A Log Metrics
            # Polars vectorization for structural arrays in log space
            df = df.with_columns([
                (pl.col("close").log() - pl.col("close").shift(1).log()).alias("Log_Return"),
                (pl.col("open").log() - pl.col("close").shift(1).log()).alias("Gap"),
                (pl.col("high").log() - pl.col("open").log()).alias("Move_High"),
                (pl.col("low").log() - pl.col("open").log()).alias("Move_Low"),
                (pl.col("close").log() - pl.col("open").log()).alias("Move_Close")
            ])
            
            # Drop the first row which will have nulls for Log_Return and Gap due to shift(1)
            df = df.drop_nulls(subset=["Log_Return", "Gap"])
            
            data_dict[asset_dir] = df
            logger.debug(f"Successfully processed {asset_dir}. Rows: {len(df)}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}", exc_info=True)
            
    return data_dict
