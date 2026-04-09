import os
import sys
import pytest
import polars as pl

# Add the root directory to sys.path so we can import 'engine'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.data_engine import load_vault_data

def test_load_vault_data_contract_a():
    """
    Tests that the data engine loads data and fulfills API Contract A.
    It expects the returned dataframe to be a Polars DataFrame with the specific log metrics.
    """
    vault_dir = "Vault_1_InSample"
    timeframe = "15m"
    
    # Calculate absolute path to Vault from the tests directory
    # Current dir when running pytest from root is the root, but let's make it robust
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vault_path = os.path.join(base_dir, vault_dir)
    
    if not os.path.exists(vault_path):
        pytest.skip(f"Vault directory {vault_path} not found.")
        
    data = load_vault_data(vault_path, timeframe=timeframe)
    
    assert isinstance(data, dict), "Result should be a dictionary mapping assets to Polars DataFrames."
    
    if not data:
        pytest.skip("No data loaded, possibly no parquet files downloaded yet.")

    first_asset = list(data.keys())[0]
    df = data[first_asset]
    
    # Assert type
    assert isinstance(df, pl.DataFrame), "Data values must be Polars DataFrames."
    
    # Assert required columns are present (Contract A)
    required_cols = ["timestamp", "open", "high", "low", "close", "volume",
                     "Log_Return", "Gap", "Move_High", "Move_Low", "Move_Close"]
                     
    for col in required_cols:
        assert col in df.columns, f"Missing required column from Contract A: {col}"
        
    # Assert no nulls in output
    null_counts = df.select([pl.col(col).is_null().sum() for col in required_cols])
    for col in required_cols:
        assert null_counts[col][0] == 0, f"Column {col} contains null values."
        
    # Basic math sanity check on one row
    # Log_Return should equal Gap + Move_Close
    # Tolerance for floating point precision
    diff = (df["Gap"] + df["Move_Close"]) - df["Log_Return"]
    assert diff.abs().max() < 1e-9, "Log_Return does not mathematically equal Gap + Move_Close."
