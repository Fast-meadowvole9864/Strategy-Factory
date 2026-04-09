import pytest
import pandas as pd
import numpy as np
from strategies.vwap_24h import VWAP24h

def create_mock_ohlcv(rows: int = 250) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    
    # Generate oscillating price across a mean
    # VWAP will follow the mean price
    close = 20000 + 100 * np.sin(np.linspace(0, 4 * np.pi, rows)) + np.random.normal(0, 5, rows)
    volume = np.random.uniform(100, 1000, rows)
    
    df = pd.DataFrame({
        "Open": close - 2,
        "High": close + 5,
        "Low": close - 5,
        "Close": close,
        "Volume": volume
    }, index=dates)
    
    return df

def test_vwap_24h_initialization():
    """Test if VWAP24h instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = VWAP24h(df)
    
    assert strategy.name == "VWAP_24h"
    assert strategy.type == "Directional"
    assert strategy.param_space == {}

def test_vwap_24h_run():
    """Test if VWAP24h returns the correct signal structure."""
    df = create_mock_ohlcv()
    strategy = VWAP24h(df, params={"vwap_window": 96})
    
    results = strategy.run()
    
    assert isinstance(results, pd.DataFrame)
    assert "Signal_Long" in results.columns
    assert "Signal_Short" in results.columns
    assert len(results) == len(df)
    
    # Signals should be either (1, 0), (0, -1), or (0, 0)
    net_signals = results["Signal_Long"] + results["Signal_Short"]
    unique_nets = net_signals.unique()
    for n in unique_nets:
        assert n in [1, 0, -1]

def test_vwap_24h_nan_handling():
    """Test if VWAP handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(150)
    strategy = VWAP24h(df, params={"vwap_window": 96})
    results = strategy.run()
    
    # Initial rows (warmup period < 96) should be 0
    assert results["Signal_Long"].iloc[0] == 0
    assert results["Signal_Short"].iloc[0] == 0
    assert not results["Signal_Long"].isna().any()
    assert not results["Signal_Short"].isna().any()
