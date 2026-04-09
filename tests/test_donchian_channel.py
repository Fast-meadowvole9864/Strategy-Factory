import numpy as np
import pandas as pd

from strategies.donchian_channel import DonchianChannel


def create_mock_ohlcv(rows: int = 200) -> pd.DataFrame:
    """Create deterministic OHLCV data for Donchian tests."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=rows, freq="15min")
    close = 20000 + np.linspace(0, 500, rows) + np.random.normal(0, 5, rows)

    return pd.DataFrame({
        "Open": close - 2,
        "High": close + 5,
        "Low": close - 5,
        "Close": close,
        "Volume": 1000,
    }, index=dates)


def create_symbol_frame(symbol: str, closes: list[float]) -> pd.DataFrame:
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "symbol": [symbol] * len(close),
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(len(close), 1000.0),
    })


def test_donchian_channel_initialization():
    df = create_mock_ohlcv()
    strategy = DonchianChannel(df)

    assert strategy.name == "Donchian_Channel"
    assert strategy.type == "Directional"
    assert "dc_length" in strategy.param_space


def test_donchian_channel_run():
    df = create_mock_ohlcv()
    strategy = DonchianChannel(df, params={"dc_length": 20})

    results = strategy.run()

    assert isinstance(results, pd.DataFrame)
    assert "Signal_Long" in results.columns
    assert "Signal_Short" in results.columns
    assert len(results) == len(df)

    net_signals = results["Signal_Long"] + results["Signal_Short"]
    for signal in net_signals.unique():
        assert signal in [1, 0, -1]


def test_donchian_channel_resets_state_per_symbol():
    """A fresh symbol should not inherit prior Donchian state from another asset."""
    btc_df = create_symbol_frame("btc", [100, 101, 102, 103, 104, 110, 111, 112])
    eth_df = create_symbol_frame("eth", [50, 51, 52, 53, 54, 60])
    df = pd.concat([btc_df, eth_df], ignore_index=True)

    strategy = DonchianChannel(df, params={"dc_length": 5})
    state = strategy._get_trend_state()

    eth_state = state[df["symbol"] == "eth"].reset_index(drop=True)

    assert eth_state.iloc[:4].eq(0).all()
    assert eth_state.iloc[5] == 1
