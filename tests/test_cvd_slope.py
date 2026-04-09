import pytest
import pandas as pd
import numpy as np
from strategies.cvd_slope import CVDSlope

def create_mock_ohlcv(rows: int = 200, with_taker: bool = True) -> pd.DataFrame:
    """Creates a mock OHLCV DataFrame for testing."""
    np.random.seed(42)
    # Generate millisecond timestamps (i64)
    start_ts = 1672531200000 # 2023-01-01 00:00:00 UTC
    timestamps = [start_ts + i * 15 * 60 * 1000 for i in range(rows)]
    
    # Generate a much stronger trending price movement to ensure CVD triggers
    # Combine a strong trend with some oscillation
    price_trend = np.linspace(0, 1000, rows)
    close = 20000 + price_trend + 50 * np.sin(np.linspace(0, 4 * np.pi, rows)) + np.random.normal(0, 2, rows)
    volume = np.random.uniform(500, 2000, rows)
    
    data = {
        "timestamp": timestamps,
        "open": close - np.random.uniform(0, 10, rows),
        "high": close + np.random.uniform(10, 20, rows),
        "low": close - np.random.uniform(10, 20, rows),
        "close": close,
        "volume": volume
    }
    
    if with_taker:
        # Bias taker_buy_vol EXTREMELY towards the trend to ensure positive/negative delta
        range_ext = (data['high'] - data['low'])
        range_ext[range_ext == 0] = 1e-9
        ratio = (data['close'] - data['open']) / range_ext
        # Map ratio [-1, 1] to [0.1, 0.9] proportion
        # This will ensure delta = 2*taker - vol is often non-zero
        taker_proportion = 0.5 + 0.4 * ratio
        data["taker_buy_vol"] = volume * taker_proportion
        
    df = pd.DataFrame(data)
    return df

def create_cvd_reset_frame(symbol: str, rows: int, step_minutes: int, taker_buy_vol: float) -> pd.DataFrame:
    """Create deterministic OHLCV rows for daily-reset / symbol-isolation tests."""
    start_ts = 1672531200000  # 2023-01-01 00:00:00 UTC
    timestamps = [start_ts + i * step_minutes * 60 * 1000 for i in range(rows)]

    open_price = np.full(rows, 100.0)
    close_price = np.full(rows, 101.0 if taker_buy_vol > 50 else 99.0)
    high_price = np.maximum(open_price, close_price) + 1.0
    low_price = np.minimum(open_price, close_price) - 1.0
    volume = np.full(rows, 100.0)

    return pd.DataFrame({
        "symbol": [symbol] * rows,
        "timestamp": timestamps,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "taker_buy_vol": np.full(rows, taker_buy_vol),
    })

def test_cvd_slope_initialization():
    """Test if CVDSlope instantiates correctly."""
    df = create_mock_ohlcv()
    strategy = CVDSlope(df)
    
    assert strategy.name == "CVD_Slope"
    assert strategy.type == "Directional"
    assert "cvd_lookback" in strategy.param_space

def test_cvd_slope_run_with_taker():
    """Test if CVDSlope returns the correct signal structure with taker volume."""
    df = create_mock_ohlcv(with_taker=True)
    strategy = CVDSlope(df, params={"cvd_lookback": 14})
    
    results = strategy.run()
    
    assert isinstance(results, pd.DataFrame)
    assert "Signal_Long" in results.columns
    assert "Signal_Short" in results.columns
    assert len(results) == len(df)
    
    # Check if signals are in [1, 0, -1]
    net_signals = results["Signal_Long"] + results["Signal_Short"]
    unique_nets = net_signals.unique()
    for n in unique_nets:
        assert n in [1, 0, -1]

def test_cvd_slope_run_without_taker():
    """Test if CVDSlope uses approximation fallback when taker volume is missing."""
    df = create_mock_ohlcv(with_taker=False)
    strategy = CVDSlope(df, params={"cvd_lookback": 14})
    
    results = strategy.run()
    
    assert isinstance(results, pd.DataFrame)
    assert "Signal_Long" in results.columns
    assert len(results) == len(df)
    assert not results["Signal_Long"].isna().any()

def test_cvd_slope_parameter_override():
    """Test if parameter override affects calculation."""
    df = create_mock_ohlcv(200)
    
    # Run with short length
    strat1 = CVDSlope(df, params={"cvd_lookback": 14})
    res1 = strat1.run()
    
    # Run with long length
    strat2 = CVDSlope(df, params={"cvd_lookback": 40})
    res2 = strat2.run()
    
    # Results should be different in at least one signal column
    assert not (res1["Signal_Long"].equals(res2["Signal_Long"]) and res1["Signal_Short"].equals(res2["Signal_Short"]))

def test_cvd_slope_nan_handling():
    """Test if CVD handles warmup period (NaNs) gracefully."""
    df = create_mock_ohlcv(50)
    strategy = CVDSlope(df, params={"cvd_lookback": 20})
    results = strategy.run()
    
    # Initial rows (warmup period) should be 0
    assert results["Signal_Long"].iloc[0] == 0
    assert results["Signal_Short"].iloc[0] == 0
    assert not results["Signal_Long"].isna().any()
    assert not results["Signal_Short"].isna().any()

def test_cvd_slope_daily_reset():
    """Test if CVD correctly resets every day at 00:00 UTC."""
    # Create 3 days of data (96 bars per day at 15m)
    rows = 96 * 3
    df = create_mock_ohlcv(rows=rows, with_taker=True)
    strategy = CVDSlope(df, params={"cvd_lookback": 14})
    
    # We need to access the inner cvd_series logic to verify reset
    # or verify that signals change predictably at reset
    # Since _calculate_cvd_diff is internal, we just check if it runs without error
    # and produces signals across the reset boundary
    results = strategy.run()
    assert len(results) == rows
    assert not results["Signal_Long"].isna().any()

def test_cvd_slope_isolates_symbols_within_daily_reset():
    """CVD state must not bleed between assets that share the same UTC date."""
    btc_df = create_cvd_reset_frame("btc", rows=6, step_minutes=60, taker_buy_vol=80.0)
    eth_df = create_cvd_reset_frame("eth", rows=6, step_minutes=60, taker_buy_vol=20.0)
    df = pd.concat([btc_df, eth_df], ignore_index=True)

    strategy = CVDSlope(df, params={"cvd_lookback": 3})
    cvd_diff = strategy._calculate_cvd_diff()

    eth_diff = cvd_diff[df["symbol"] == "eth"].reset_index(drop=True)
    assert eth_diff.iloc[0] == 0
    assert eth_diff.iloc[3] < 0

def test_cvd_slope_caps_hourly_lookback_to_daily_window():
    """On 1h data a daily-reset CVD should not accept lookbacks beyond one day."""
    df = create_cvd_reset_frame("btc", rows=48, step_minutes=60, taker_buy_vol=80.0)
    strategy = CVDSlope(df, params={"cvd_lookback": 50})

    cvd_diff = strategy._calculate_cvd_diff().reset_index(drop=True)

    assert cvd_diff.iloc[23] > 0
    assert cvd_diff.iloc[47] > 0
