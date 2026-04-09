import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, Literal
from strategies.base_strategy import BaseStrategy

# --- Mock Implementations for Testing ---

class MockDirectionalStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MockDirectional"

    @property
    def type(self) -> Literal["Directional", "Magnitude"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def generate_long_signal(self) -> pd.Series:
        # Long when close > open
        return (self.df['Close'] > self.df['Open']).astype(int)

    def generate_short_signal(self) -> pd.Series:
        # Short when close < open
        return (self.df['Close'] < self.df['Open']).astype(int) * -1

    def generate_magnitude_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

class MockMagnitudeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "MockMagnitude"

    @property
    def type(self) -> Literal["Directional", "Magnitude"]:
        return "Magnitude"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {"threshold": {"type": "int", "min": 1, "max": 5000}}

    def generate_long_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def generate_short_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def generate_magnitude_signal(self) -> pd.Series:
        # Active when high-low > threshold
        threshold = self.params.get("threshold", 10)
        return ((self.df['High'] - self.df['Low']) > threshold).astype(int)

# --- Test Fixtures ---

@pytest.fixture
def sample_data():
    """Generates 100 rows of fake OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="15min")
    df = pd.DataFrame({
        "Open": np.random.uniform(20000, 21000, 100),
        "High": np.random.uniform(21000, 22000, 100),
        "Low": np.random.uniform(19000, 20000, 100),
        "Close": np.random.uniform(20000, 21000, 100),
        "Volume": np.random.uniform(1, 100, 100)
    }, index=dates)
    return df

# --- Unit Tests ---

def test_base_strategy_is_abstract():
    """Ensure BaseStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseStrategy(pd.DataFrame())

def test_directional_strategy_initialization(sample_data):
    """Test directional strategy instantiation and properties."""
    strat = MockDirectionalStrategy(sample_data)
    assert strat.name == "MockDirectional"
    assert strat.type == "Directional"
    assert isinstance(strat.df, pd.DataFrame)

def test_directional_strategy_run(sample_data):
    """Test directional strategy execution output."""
    strat = MockDirectionalStrategy(sample_data)
    signals = strat.run()
    
    assert isinstance(signals, pd.DataFrame)
    assert "Signal_Long" in signals.columns
    assert "Signal_Short" in signals.columns
    assert "Signal_Magnitude" not in signals.columns
    assert len(signals) == len(sample_data)
    
    # Check signal values
    assert all(signals["Signal_Long"].isin([0, 1]))
    assert all(signals["Signal_Short"].isin([0, -1]))

def test_magnitude_strategy_run(sample_data):
    """Test magnitude strategy execution with params."""
    params = {"threshold": 1500} # Set high to filter some
    strat = MockMagnitudeStrategy(sample_data, params=params)
    signals = strat.run()
    
    assert "Signal_Magnitude" in signals.columns
    assert "Signal_Long" not in signals.columns
    assert "Signal_Short" not in signals.columns
    assert all(signals["Signal_Magnitude"].isin([0, 1]))

def test_invalid_input_type():
    """Test error when input is not a DataFrame."""
    with pytest.raises(TypeError):
        MockDirectionalStrategy("not a dataframe")

def test_invalid_strategy_type(sample_data):
    """Test error for unknown strategy type."""
    class BrokenStrategy(MockDirectionalStrategy):
        @property
        def type(self):
            return "Broken"
            
    strat = BrokenStrategy(sample_data)
    with pytest.raises(ValueError, match="Invalid strategy type"):
        strat.run()
