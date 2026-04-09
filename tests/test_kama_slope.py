import pytest
import pandas as pd
import numpy as np
from strategies.kama_slope import KAMASlope

def create_mock_ohlcv(rows: int = 250) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    
    # Generate some regimes: range bound then trending then range bound
    close = np.zeros(rows)
    mid1 = rows // 3
    mid2 = 2 * rows // 3
    
    # First 1/3: Range bound (KAMA should flatline)
    close[:mid1] = 20000 + np.random.normal(0, 10, mid1)
    
    # Second 1/3: Strong trend (KAMA should step up)
    close[mid1:mid2] = close[mid1-1] + np.linspace(0, 500, mid2 - mid1) + np.random.normal(0, 5, mid2 - mid1)
    
    # Last 1/3: Range bound (KAMA should flatline again)
    close[mid2:] = close[mid2-1] + np.random.normal(0, 10, rows - mid2)
    
    df = pd.DataFrame({
        "Open": close - np.random.uniform(0, 5, rows),
        "High": close + np.random.uniform(5, 10, rows),
        "Low": close - np.random.uniform(5, 10, rows),
        "Close": close,
        "Volume": 1000
    }, index=dates)
    
    return df

def test_kama_slope_initialization():
    """Test if KAMASlope instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = KAMASlope(df)
    
    assert strategy.name == "KAMA_Slope"
    assert strategy.type == "Directional"
    assert "kama_length" in strategy.param_space

def test_kama_slope_run():
    """Test if KAMASlope returns the correct signal structure."""
    df = create_mock_ohlcv()
    strategy = KAMASlope(df, params={"kama_length": 10})
    
    results = strategy.run()
    
    assert isinstance(results, pd.DataFrame)
    assert "Signal_Long" in results.columns
    assert "Signal_Short" in results.columns
    assert len(results) == len(df)
    
    # Signals should be either (1, 0) or (0, -1) or (0, 0)
    net_signals = results["Signal_Long"] + results["Signal_Short"]
    unique_nets = net_signals.unique()
    for n in unique_nets:
        assert n in [1, 0, -1]

def test_kama_slope_parameter_override():
    """Test if parameter override affects calculation."""
    df = create_mock_ohlcv(200)
    
    # Run with short length
    strat1 = KAMASlope(df, params={"kama_length": 10})
    res1 = strat1.run()
    
    # Run with long length
    strat2 = KAMASlope(df, params={"kama_length": 50})
    res2 = strat2.run()
    
    # Results should be different
    assert not res1["Signal_Long"].equals(res2["Signal_Long"])

def test_kama_slope_nan_handling():
    """Test if KAMA handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(50)
    strategy = KAMASlope(df, params={"kama_length": 10})
    results = strategy.run()
    
    # Initial rows (warmup period) should be 0
    assert results["Signal_Long"].iloc[0] == 0
    assert results["Signal_Short"].iloc[0] == 0
    assert not results["Signal_Long"].isna().any()
    assert not results["Signal_Short"].isna().any()
