import pytest
import sys
import os
import polars as pl
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.sandbox_engine import SandboxEngine
from strategies.base_strategy import BaseStrategy

class MockDirectionalCartridge(BaseStrategy):
    @property
    def name(self): return "MockDirectional"
    
    @property
    def type(self): return "Directional"

    @property
    def param_space(self): return {}
    
    def generate_long_signal(self):
        # 1 at index 1 and 2
        return pd.Series([0, 1, 1, 0, 0, 0], index=self.df.index)
        
    def generate_short_signal(self):
        # -1 at index 4 and 5
        return pd.Series([0, 0, 0, 0, -1, -1], index=self.df.index)
        
    def generate_magnitude_signal(self):
        return pd.Series(dtype=float)

class MockMagnitudeCartridge(BaseStrategy):
    @property
    def name(self): return "MockMagnitude"
    
    @property
    def type(self): return "Magnitude"

    @property
    def param_space(self): return {}
    
    def generate_long_signal(self):
        return pd.Series(dtype=float)
        
    def generate_short_signal(self):
        return pd.Series(dtype=float)
        
    def generate_magnitude_signal(self):
        # 1 at index 2 and 3
        return pd.Series([0, 0, 1, 1, 0, 0], index=self.df.index)

def test_sandbox_engine_directional():
    # Hallucinate a mock Polars DataFrame
    df = pl.DataFrame({
        "timestamp": [1, 2, 3, 4, 5, 6],
        "symbol": ["btc"] * 6,
        "Log_Return": [0.0, 0.1, -0.2, 0.3, -0.4, 0.5],
        "Move_Close": [0.0, 0.05, -0.1, 0.15, -0.2, 0.25]
    })
    
    engine = SandboxEngine(df, MockDirectionalCartridge)
    res = engine.run()
    
    # Expected calculations:
    # Log_Return:    [0.0, 0.1, -0.2, 0.3, -0.4, 0.5]
    # Move_Close:    [0.0, 0.05, -0.1, 0.15, -0.2, 0.25]
    
    # Signal_Long:   [0, 1, 1, 0, 0, 0]
    # S_Long_1:      [null, 0, 1, 1, 0, 0]
    # S_Long_2:      [null, null, 0, 1, 1, 0]
    # Return_Long: [0, 0, -0.2, 0.3, 0, 0]
    # PF_Long = 0.3 / 0.2 = 1.5
    
    # Signal_Short:  [0, 0, 0, 0, -1, -1]
    # S_Short_1:     [null, 0, 0, 0, 0, -1]
    # S_Short_2:     [null, null, 0, 0, 0, 0]
    # Return_Short: [0, 0, 0, 0, 0, -0.5]
    # PF_Short = 0.0 / 0.5 = 0.0
    
    # Total Returns: Return_Long + Return_Short
    # [0, 0, -0.2, 0.3, 0, -0.5]
    # PF_Total = 0.3 / (0.2 + 0.5) = 0.4285714285714286
    
    assert abs(res["Profit_Factor_Long"] - 1.5) < 1e-6
    assert abs(res["Profit_Factor_Short"] - 0.0) < 1e-6
    assert abs(res["Profit_Factor_Total"] - (0.3 / 0.7)) < 1e-6
    assert res["Total_Trades"] == 2

def test_sandbox_engine_magnitude():
    # Hallucinate a mock Polars DataFrame
    df = pl.DataFrame({
        "timestamp": [1, 2, 3, 4, 5, 6],
        "symbol": ["btc"] * 6,
        "Log_Return": [0.0, 0.1, -0.5, 0.5, -0.1, 0.1],
        "Move_Close": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    engine = SandboxEngine(df, MockMagnitudeCartridge)
    res = engine.run()
    
    # Expected calculations (incorporating .shift(1) lookahead prevention):
    # Log_Return:      [0.0, 0.1, -0.5, 0.5, -0.1, 0.1]
    
    # Signal_Mag:      [0, 0, 1, 1, 0, 0]
    # Shifted_Mag:     [null, 0, 0, 1, 1, 0]
    
    # Active Returns (where Shifted_Mag == 1):
    # Indexes: 3, 4 -> Log_Returns: 0.5, -0.1
    # Absolute Active Returns: 0.5, 0.1
    # Active Mean = (0.5 + 0.1) / 2 = 0.3
    
    # Inactive Returns (where Shifted_Mag == 0):
    # Indexes: 0, 1, 2, 5 -> Log_Returns: 0.0, 0.1, -0.5, 0.1
    # Absolute Inactive Returns: 0.0, 0.1, 0.5, 0.1
    # Inactive Mean = (0.0 + 0.1 + 0.5 + 0.1) / 4 = 0.175
    
    # Magnitude Ratio = 0.3 / 0.175 = 1.7142857142857142
    
    assert abs(res["Active_Mean"] - 0.3) < 1e-6
    assert abs(res["Inactive_Mean"] - 0.175) < 1e-6
    assert abs(res["Magnitude_Ratio"] - (0.3 / 0.175)) < 1e-6
