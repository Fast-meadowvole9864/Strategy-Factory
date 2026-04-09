import pytest
import pandas as pd
import numpy as np
from strategies.rsi_momentum import RSIMomentum

def create_mock_ohlcv(rows: int = 200) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    
    # Generate oscillating price to cross RSI 50
    # RSI > 50 on uptrend, RSI < 50 on downtrend
    close = 20000 + 100 * np.sin(np.linspace(0, 8 * np.pi, rows)) + np.random.normal(0, 5, rows)
    
    df = pd.DataFrame({
        "Open": close - 5,
        "High": close + 10,
        "Low": close - 10,
        "Close": close,
        "Volume": 1000
    }, index=dates)
    
    return df

def test_rsi_momentum_initialization():
    """Test if RSIMomentum instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = RSIMomentum(df)
    
    assert strategy.name == "RSI_Momentum"
    assert strategy.type == "Directional"
    assert "rsi_length" in strategy.param_space

def test_rsi_momentum_run():
    """Test if RSIMomentum returns the correct signal structure."""
    df = create_mock_ohlcv()
    strategy = RSIMomentum(df, params={"rsi_length": 14})
    
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

def test_rsi_momentum_parameter_override():
    """Test if parameter override affects calculation."""
    df = create_mock_ohlcv(200)
    
    # Run with short length
    strat1 = RSIMomentum(df, params={"rsi_length": 7})
    res1 = strat1.run()
    
    # Run with long length
    strat2 = RSIMomentum(df, params={"rsi_length": 21})
    res2 = strat2.run()
    
    # Results should be different
    assert not res1["Signal_Long"].equals(res2["Signal_Long"])

def test_rsi_momentum_nan_handling():
    """Test if RSI handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(50)
    strategy = RSIMomentum(df, params={"rsi_length": 14})
    results = strategy.run()
    
    # Initial rows (warmup period) should be 0
    assert results["Signal_Long"].iloc[0] == 0
    assert results["Signal_Short"].iloc[0] == 0
    assert not results["Signal_Long"].isna().any()
    assert not results["Signal_Short"].isna().any()
