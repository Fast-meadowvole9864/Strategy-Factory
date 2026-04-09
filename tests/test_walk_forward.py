import pytest
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Any, Literal
from engine.walk_forward import WalkForwardEngine, simulate_execution_1m
from strategies.base_strategy import BaseStrategy

class MockCartridge(BaseStrategy):
    @property
    def name(self) -> str: return "Mock"
    
    @property
    def type(self) -> Literal["Directional"]: return "Directional"
    
    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {"test_param": {"type": "int", "min": 1, "max": 5}}
        
    def generate_long_signal(self) -> pd.Series:
        # Return 1 for first element, 0 rest
        s = pd.Series(0, index=self.df.index)
        if len(s) > 0: s.iloc[0] = 1
        return s
        
    def generate_short_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)
        
    def generate_magnitude_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

def test_walk_forward_rolling_loop(monkeypatch):
    """
    Assertion 1: Prove the rolling loop correctly stitches the OOS windows without data leaks.
    """
    df_15m = pl.DataFrame({
        "timestamp": np.arange(10) * 15 * 60000,
        "symbol": ["btc"] * 10,
        "open": np.ones(10), "high": np.ones(10), "low": np.ones(10), "close": np.ones(10),
        "Log_Return": np.zeros(10)
    })
    df_1m = pl.DataFrame({
        "timestamp": np.arange(150) * 60000,
        "symbol": ["btc"] * 150,
        "open": np.ones(150), "high": np.ones(150), "low": np.ones(150), "close": np.ones(150)
    })
    
    def mock_run(self):
        return {"test_param": 1}, 1.0, {}
        
    monkeypatch.setattr("engine.walk_forward.OptunaOptimizer.run", mock_run)
    
    engine = WalkForwardEngine(df_15m, df_1m, MockCartridge, train_window=4, test_window=2, optuna_trials=1)
    
    # Mock optimize_sl_tp to bypass Optuna in tests
    monkeypatch.setattr(engine, "optimize_sl_tp", lambda arrs: (0.02, 0.04))
    
    trades, oos_df, rolling_parameters = engine.run()
    
    # 3 rolls of test_window=2
    assert len(oos_df) == 6
    assert oos_df["timestamp"][0] == 4 * 15 * 60000
    assert oos_df["timestamp"][5] == 9 * 15 * 60000

def test_simulate_execution_1m_sl_before_tp():
    """
    Assertion 2: Prove the Numba loop correctly triggers a Stop-Loss before a Take-Profit if the 1m Low hits the threshold first.
    """
    signals_15m = np.array([
        [0, 1, 0],   
        [15, 0, 0]
    ], dtype=np.float64)
    
    prices_1m = np.array([
        [0, 100, 110, 95, 100], # Entry at Open=100. SL evaluated. Low 95 > SL 90.
        [1, 100, 125, 85, 100]  # Hit SL (85 <= 90). Return -0.10. Note: High 125 also hits TP, but SL should trigger first.
    ], dtype=np.float64)
    
    results = simulate_execution_1m(signals_15m, prices_1m, sl_pct=0.10, tp_pct=0.20)
    assert len(results) == 1
    assert results[0] == -0.10

def test_simulate_execution_1m_entry_candle_sl():
    """
    Prove that the entry candle itself can resolve a Stop-Loss.
    """
    signals_15m = np.array([
        [0, 1, 0]
    ], dtype=np.float64)
    
    prices_1m = np.array([
        [0, 100, 101, 89, 95]
    ], dtype=np.float64)
    
    results = simulate_execution_1m(signals_15m, prices_1m, sl_pct=0.10, tp_pct=0.20)
    assert len(results) == 1
    assert results[0] == -0.10

def test_simulate_execution_1m_both_hit_uses_closest_level():
    """
    Prove that a same-candle SL/TP collision resolves to the level closest to the open.
    """
    signals_15m = np.array([
        [0, 1, 0]
    ], dtype=np.float64)
    
    prices_1m = np.array([
        [0, 100, 111, 79, 100]
    ], dtype=np.float64)
    
    results = simulate_execution_1m(signals_15m, prices_1m, sl_pct=0.20, tp_pct=0.10)
    assert len(results) == 1
    assert results[0] == 0.10

def test_simulate_execution_1m_ignores_entry_candle_opposing_signal():
    """
    Prove conflicting entry-candle signals do not create an artificial zero-PnL exit.
    """
    signals_15m = np.array([
        [0, 1, -1],
        [15, 0, -1]
    ], dtype=np.float64)
    
    prices_1m = np.array([
        [0, 100, 101, 99, 100],
        [15, 105, 106, 104, 105]
    ], dtype=np.float64)
    
    results = simulate_execution_1m(signals_15m, prices_1m, sl_pct=0.10, tp_pct=0.20)
    assert len(results) == 1
    assert abs(results[0] - 0.05) < 1e-6

def test_simulate_execution_1m_tp_before_sl():
    """
    Additional proof for Take Profit resolution.
    """
    signals_15m = np.array([
        [0, 1, 0]
    ], dtype=np.float64)
    
    prices_1m = np.array([
        [0, 100, 110, 95, 100],
        [1, 100, 120, 95, 100] # High hits 120 (TP=120) before SL. 
    ], dtype=np.float64)
    
    results = simulate_execution_1m(signals_15m, prices_1m, sl_pct=0.10, tp_pct=0.20)
    assert len(results) == 1
    assert results[0] == 0.20

def test_simulate_execution_1m_flat_exit():
    """
    Prove that a sparse signal dropping to 0 does not exit a trade under the decoupled execution model.
    """
    signals_15m = np.array([
        [0, 1, 0],   # Timestamp 0: Enter Long
        [15, 0, 0]   # Timestamp 15: Indicator goes Flat (Long drops to 0)
    ], dtype=np.float64)
    
    prices_1m = np.array([
        [0, 100, 102, 98, 101], # Enter at 100. SL is 90, TP is 120. (Neither hit)
        [1, 101, 103, 99, 102], # Still active, state is 1. (Neither hit)
        [15, 105, 108, 104, 106] # Timestamp 15: State drops to 0! Must exit at Open=105.
    ], dtype=np.float64)
    
    results = simulate_execution_1m(signals_15m, prices_1m, sl_pct=0.10, tp_pct=0.20)
    assert len(results) == 0
