import numpy as np
import pandas as pd

from strategies.adx_regime import ADXRegime
from strategies.ema_150_slope import EMA150Slope


def create_symbol_frame(symbol: str, close_values: np.ndarray) -> pd.DataFrame:
    """Create deterministic OHLCV rows for multi-symbol isolation tests."""
    close = np.asarray(close_values, dtype=float)
    rows = len(close)

    return pd.DataFrame({
        "symbol": [symbol] * rows,
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": np.full(rows, 1000.0),
    })


def create_isolation_df(btc_rows: int, eth_rows: int) -> pd.DataFrame:
    """Stack a long BTC history ahead of a fresh ETH history."""
    btc_close = np.linspace(100.0, 200.0, btc_rows)
    eth_close = np.linspace(50.0, 60.0, eth_rows)

    return pd.concat(
        [
            create_symbol_frame("btc", btc_close),
            create_symbol_frame("eth", eth_close),
        ],
        ignore_index=True,
    )


def test_ema_150_slope_warmup_resets_per_symbol():
    """A fresh symbol should not inherit EMA slope state from the prior asset."""
    df = create_isolation_df(btc_rows=220, eth_rows=12)
    strategy = EMA150Slope(df, params={"slope_threshold": 0.0})

    normalized_slope = strategy._calculate_slope()
    eth_slope = normalized_slope[df["symbol"] == "eth"].reset_index(drop=True)

    assert eth_slope.eq(0).all()


def test_adx_regime_warmup_resets_per_symbol():
    """ADX should warm up independently for each symbol slice."""
    df = create_isolation_df(btc_rows=40, eth_rows=10)
    strategy = ADXRegime(df, params={"adx_length": 5, "adx_threshold": 10})

    results = strategy.run()
    eth_signal = results.loc[df["symbol"] == "eth", "Signal_Magnitude"].reset_index(drop=True)

    assert eth_signal.iloc[:4].eq(0).all()
    assert eth_signal.iloc[4] == 1
