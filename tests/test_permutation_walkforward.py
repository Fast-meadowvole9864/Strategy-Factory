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

def test_permutation_walkforward_preservation():
    # Construct a larger mock DataFrame to test start_index=50
    n_rows = 100
    np.random.seed(42) # For reproducibility of non-matching tail
    closes = np.linspace(100.0, 200.0, n_rows)
    log_closes = np.log(closes)
    log_returns = np.concatenate([[0.0], np.diff(log_closes)])
    
    gaps = log_returns
    move_closes = np.zeros(n_rows)
    move_highs = np.zeros(n_rows)
    move_lows = np.zeros(n_rows)
    
    df = pl.DataFrame({
        "symbol": ["btc"] * n_rows,
        "close": closes,
        "Gap": gaps,
        "Move_High": move_highs,
        "Move_Low": move_lows,
        "Move_Close": move_closes
    })
    
    start_index = 50
    engine = PermutationEngine(df, MockCartridge, n_permutations=1)
    synth_df = engine.generate_synthetic_data(start_index=start_index)
    
    # Use array equality checks to verify that the first 50 rows of original and synthetic data match
    original_head = df.head(start_index)
    synth_head = synth_df.head(start_index)
    
    # Check close match
    assert np.allclose(original_head["close"].to_numpy(), synth_head["close"].to_numpy())
    
    # Check the tail does NOT match (proving shuffling happened)
    original_tail = df.tail(n_rows - start_index)
    synth_tail = synth_df.tail(n_rows - start_index)
    
    assert not np.allclose(original_tail["close"].to_numpy(), synth_tail["close"].to_numpy())
