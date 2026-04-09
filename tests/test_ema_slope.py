import pytest
import pandas as pd
import numpy as np
from strategies.ema_slope import EMASlope

def create_mock_ohlcv(rows: int = 200) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    
    # Generate oscillating trend
    close = 20000 + 1000 * np.sin(np.linspace(0, 4 * np.pi, rows)) + np.random.normal(0, 10, rows)
    
    df = pd.DataFrame({
        "Open": close - np.random.uniform(0, 10, rows),
        "High": close + np.random.uniform(10, 20, rows),
        "Low": close - np.random.uniform(10, 20, rows),
        "Close": close,
        "Volume": np.random.uniform(100, 1000, rows)
    }, index=dates)
    
    return df

def test_ema_slope_initialization():
    """Test if EMASlope instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = EMASlope(df)
    
    assert strategy.name == "EMA_Slope"
    assert strategy.type == "Directional"
    assert "ema_length" in strategy.param_space

def test_ema_slope_run():
    """Test if EMASlope returns the correct signal structure."""
    df = create_mock_ohlcv()
    strategy = EMASlope(df, params={"ema_length": 14})
    
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

def test_ema_slope_parameter_override():
    """Test if parameter override affects calculation."""
    df = create_mock_ohlcv(200)
    
    # Run with short length
    strat1 = EMASlope(df, params={"ema_length": 14})
    res1 = strat1.run()
    
    # Run with long length
    strat2 = EMASlope(df, params={"ema_length": 50})
    res2 = strat2.run()
    
    # Results should be different
    assert not res1["Signal_Long"].equals(res2["Signal_Long"])

def test_ema_slope_nan_handling():
    """Test if EMA handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(50)
    strategy = EMASlope(df, params={"ema_length": 20})
    results = strategy.run()
    
    # Initial rows (warmup period) should be 0
    assert results["Signal_Long"].iloc[0] == 0
    assert results["Signal_Short"].iloc[0] == 0
    assert not results["Signal_Long"].isna().any()
    assert not results["Signal_Short"].isna().any()
