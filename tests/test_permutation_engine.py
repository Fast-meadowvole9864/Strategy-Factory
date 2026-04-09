import pytest
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, Any, Literal
from engine.permutation_engine import PermutationEngine
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
        return pd.Series(0, index=self.df.index)
        
    def generate_short_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)
        
    def generate_magnitude_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

def test_permutation_synthetic_data_generation():
    # Construct a small mock DataFrame to test exact math logic
    df = pl.DataFrame({
        "symbol": ["btc", "btc", "btc"],
        "close": [100.0, 105.0, 102.0],
        "Gap": [0.0, np.log(101.0) - np.log(100.0), np.log(104.0) - np.log(105.0)],
        "Move_High": [0.0, np.log(106.0) - np.log(101.0), np.log(105.0) - np.log(104.0)],
        "Move_Low": [0.0, np.log(100.0) - np.log(101.0), np.log(101.0) - np.log(104.0)],
        "Move_Close": [0.0, np.log(105.0) - np.log(101.0), np.log(102.0) - np.log(104.0)]
    })
    
    engine = PermutationEngine(df, MockCartridge, n_permutations=1)
    synth_df = engine.generate_synthetic_data()
    
    # Verify shape
    assert len(synth_df) == len(df)
    
    # Verify all arrays are present
    assert "open" in synth_df.columns
    assert "high" in synth_df.columns
    assert "low" in synth_df.columns
    assert "close" in synth_df.columns
    assert "Log_Return" in synth_df.columns
    
    # The first element's close log should match the initial close log + the first log return
    expected_first_close = df["close"][0] * np.exp(synth_df["Log_Return"][0])
    assert abs(synth_df["close"][0] - expected_first_close) < 1e-6

def test_permutation_engine_run(monkeypatch):
    """
    Mock OptunaOptimizer.run to return a static value and avoid actually running Optuna 3 * 10 times.
    Also tests the Benjamini-Hochberg FDR correction doesn't crash.
    """
    def mock_run(self, *args, **kwargs):
        # We just return a dummy metric. 
        # For permutation paths let's return a random uniform to simulate synthetic distribution
        metric = np.random.uniform(0.5, 1.5)
        # Mock returns (best_params, metric, dict)
        return ({"test_param": 2}, metric, {"Profit_Factor_Total": metric, "Total_Trades": 35})
        
    monkeypatch.setattr("engine.permutation_engine.OptunaOptimizer.run", mock_run)
    
    df = pl.DataFrame({
        "symbol": ["btc", "btc", "btc"],
        "close": [100.0, 105.0, 102.0],
        "Gap": [0.0, 0.01, -0.01],
        "Move_High": [0.0, 0.02, 0.02],
        "Move_Low": [0.0, -0.01, -0.02],
        "Move_Close": [0.0, 0.01, 0.01]
    })
    
    # Run only 3 permutations to prove the loop + BH logic works
    engine = PermutationEngine(df, MockCartridge, n_permutations=3, optuna_trials=2)
    real_bench, synth_metrics, p_value = engine.run()
    
    assert isinstance(real_bench, float)
    assert len(synth_metrics) == 3
    assert isinstance(p_value, float)
    # P-value should be between 0 and 1
    assert 0.0 <= p_value <= 1.0
