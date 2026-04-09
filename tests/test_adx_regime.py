import pytest
import pandas as pd
import numpy as np
from strategies.adx_regime import ADXRegime

def create_mock_ohlcv(rows: int = 150) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    
    # Generate a much stronger trending price movement to ensure ADX hits 25
    base_price = 20000
    price_trend = np.linspace(0, 2000, rows) # Stronger trend
    noise = np.random.normal(0, 10, rows) # Lower noise
    close = base_price + price_trend + noise
    
    df = pd.DataFrame({
        "Open": close - np.random.uniform(0, 10, rows),
        "High": close + np.random.uniform(5, 20, rows),
        "Low": close - np.random.uniform(5, 20, rows),
        "Close": close,
        "Volume": np.random.uniform(100, 1000, rows)
    }, index=dates)
    
    return df

def test_adx_regime_initialization():
    """Test if ADXRegime instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = ADXRegime(df)
    
    assert strategy.name == "ADX_Regime"
    assert strategy.type == "Magnitude"
    assert strategy.params == {}

def test_adx_regime_run():
    """Test if ADXRegime returns the correct signal structure."""
    df = create_mock_ohlcv()
    strategy = ADXRegime(df, params={"adx_length": 14})
    
    results = strategy.run()
    
    assert isinstance(results, pd.DataFrame)
    assert "Signal_Magnitude" in results.columns
    assert len(results) == len(df)
    
    # Check if we have 1s and 0s
    unique_signals = results["Signal_Magnitude"].unique()
    for s in unique_signals:
        assert s in [0, 1]

def test_adx_regime_parameter_override():
    """Test if parameter override affects calculation."""
    df = create_mock_ohlcv(200)
    
    # Run with short length
    strat1 = ADXRegime(df, params={"adx_length": 10})
    res1 = strat1.run()
    
    # Run with long length
    strat2 = ADXRegime(df, params={"adx_length": 40})
    res2 = strat2.run()
    
    # Results should be different
    assert not res1["Signal_Magnitude"].equals(res2["Signal_Magnitude"])

def test_adx_regime_nan_handling():
    """Test if ADX handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(50)
    strategy = ADXRegime(df, params={"adx_length": 14})
    results = strategy.run()
    
    # Initial rows (warmup period) should be 0
    # For ADX 14, at least 14 rows + some smoothing rows will be NaN
    assert results["Signal_Magnitude"].iloc[0] == 0
    assert not results["Signal_Magnitude"].isna().any()
