import pytest
import polars as pl
import pandas as pd
from typing import Dict, Any, Literal
from engine.optimizer import OptunaOptimizer, TRADE_FLOOR_PER_ASSET_YEAR
from strategies.base_strategy import BaseStrategy

# Mock Cartridge
class MockCartridge(BaseStrategy):
    @property
    def name(self) -> str:
        return "Mock"

    @property
    def type(self) -> Literal["Magnitude"]:
        return "Magnitude"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "test_param": {"type": "int", "min": 10, "max": 20}
        }

    def generate_long_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def generate_short_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def generate_magnitude_signal(self) -> pd.Series:
        return pd.Series(1, index=self.df.index)

class MockDirectionalCartridge(MockCartridge):
    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

def test_optuna_optimizer(monkeypatch):
    """
    Tests the OptunaOptimizer by mocking the SandboxEngine.run method
    to prevent needing real data or expensive calculations.
    """
    def mock_run(self):
        # Read the param passed from Optuna
        val = self.params.get("test_param", 10)
        # Return a metric that increases with the param, and trades > 30 to avoid penalty
        return {
            "Active_Mean": float(val), 
            "Total_Trades": 50,
            "Profit_Factor_Long": float(val),
            "Profit_Factor_Short": 1.2,
            "Profit_Factor_Total": float(val)
        }
        
    monkeypatch.setattr("engine.optimizer.SandboxEngine.run", mock_run)
    
    # Create empty polars df
    df = pl.DataFrame()
    
    optimizer = OptunaOptimizer(data=df, cartridge_class=MockDirectionalCartridge, n_trials=5)
    best_params, best_metric, best_results = optimizer.run()
    
    assert "test_param" in best_params
    assert best_metric > 0
    # The highest possible value in the search space is 20, so the max metric returned should be close to 20
    assert best_params["test_param"] <= 20
    
    # Check that best_results dictionary is successfully returned
    assert best_results["Total_Trades"] == 50
    assert best_results["Profit_Factor_Long"] > 0.0

def test_optuna_trade_penalty(monkeypatch):
    """
    Tests that the optimizer correctly zeroes out the metric if trades are below the floor.
    """
    def mock_run_penalty(self):
        val = self.params.get("test_param", 10)
        return {
            "Profit_Factor_Total": 100.0,
            "Total_Trades": int(TRADE_FLOOR_PER_ASSET_YEAR) - 1
        }
        
    monkeypatch.setattr("engine.optimizer.SandboxEngine.run", mock_run_penalty)
    
    df = pl.DataFrame()
    optimizer = OptunaOptimizer(data=df, cartridge_class=MockDirectionalCartridge, n_trials=2)
    _, best_metric, _ = optimizer.run()
    
    assert best_metric == 0.0

def test_optuna_trade_floor_uses_30_per_combined_asset_year():
    timeframe_ms = 60 * 60 * 1000
    hours_per_year = int(365.25 * 24)
    rows = 2 * hours_per_year
    df = pl.DataFrame({"timestamp": [i * timeframe_ms for i in range(rows)]})

    optimizer = OptunaOptimizer(data=df, cartridge_class=MockDirectionalCartridge, n_trials=1)

    assert optimizer.min_trades == 60
