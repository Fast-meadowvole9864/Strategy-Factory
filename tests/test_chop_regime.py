import pytest
import pandas as pd
import numpy as np
from strategies.chop_regime import CHOPRegime

def create_mock_ohlcv(rows: int = 150) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    
    # Generate two regimes:
    # 1. First half: Tight range (High CHOP)
    # 2. Second half: Strong trend (Low CHOP)
    
    mid = rows // 2
    close = np.zeros(rows)
    # Range bound
    close[:mid] = 20000 + np.random.normal(0, 10, mid)
    # Trending
    close[mid:] = close[mid-1] + np.linspace(0, 1000, rows - mid) + np.random.normal(0, 5, rows - mid)
    
    df = pd.DataFrame({
        "Open": close - np.random.uniform(0, 10, rows),
        "High": close + np.random.uniform(5, 20, rows),
        "Low": close - np.random.uniform(5, 20, rows),
        "Close": close,
        "Volume": np.random.uniform(100, 1000, rows)
    }, index=dates)
    
    return df

def test_chop_regime_initialization():
    """Test if CHOPRegime instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = CHOPRegime(df)
    
    assert strategy.name == "CHOP_Regime"
    assert strategy.type == "Magnitude"
    assert "chop_length" in strategy.param_space

def test_chop_regime_run():
    """Test if CHOPRegime returns the correct signal structure."""
    df = create_mock_ohlcv()
    strategy = CHOPRegime(df, params={"chop_length": 14})
    
    results = strategy.run()
    
    assert isinstance(results, pd.DataFrame)
    assert "Signal_Magnitude" in results.columns
    assert len(results) == len(df)
    
    # Check if we have 1s and 0s
    unique_signals = results["Signal_Magnitude"].unique()
    for s in unique_signals:
        assert s in [0, 1]

def test_chop_regime_parameter_override():
    """Test if parameter override affects calculation."""
    df = create_mock_ohlcv(200)
    
    # Run with short length
    strat1 = CHOPRegime(df, params={"chop_length": 10})
    res1 = strat1.run()
    
    # Run with long length
    strat2 = CHOPRegime(df, params={"chop_length": 30})
    res2 = strat2.run()
    
    # Results should be different
    assert not res1["Signal_Magnitude"].equals(res2["Signal_Magnitude"])

def test_chop_regime_nan_handling():
    """Test if CHOP handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(50)
    strategy = CHOPRegime(df, params={"chop_length": 14})
    results = strategy.run()
    
    # Initial rows (warmup period) should be 0
    assert results["Signal_Magnitude"].iloc[0] == 0
    assert not results["Signal_Magnitude"].isna().any()
