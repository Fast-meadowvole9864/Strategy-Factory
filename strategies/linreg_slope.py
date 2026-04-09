import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class LinRegSlope(BaseStrategy):
    """
    LinRegSlope Cartridge: A Directional strategy based on the 
    Normalized Slope of the Linear Regression line (Slope / ATR).
    
    Physics:
    Defines the mathematical trajectory of price, normalized by volatility. 
    Normalized Slope > Threshold indicates an actionable upward trajectory (Long).
    Normalized Slope < -Threshold indicates an actionable downward trajectory (Short).
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns.
    - Output: Pandas DataFrame with 'Signal_Long' (1/0) and 'Signal_Short' (-1/0).
    """

    @property
    def name(self) -> str:
        return "LinReg_Slope"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "linreg_length": {"type": "int", "min": 14, "max": 100},
            "slope_threshold": {"type": "float", "min": 0.001, "max": 0.05}
        }

    def generate_long_signal(self) -> pd.Series:
        """
        Calculates the LONG entry signal: Normalized Slope > Threshold.
        Returns a Series of 1 (Long) or 0 (Flat).
        """
        normalized_slope = self._calculate_slope()
        threshold = self.params.get("slope_threshold", 0.0)
        
        long_signal = (normalized_slope > threshold).astype(int).fillna(0)
        return long_signal

    def generate_short_signal(self) -> pd.Series:
        """
        Calculates the SHORT entry signal: Normalized Slope < -Threshold.
        Returns a Series of -1 (Short) or 0 (Flat).
        """
        normalized_slope = self._calculate_slope()
        threshold = self.params.get("slope_threshold", 0.0)
        
        short_signal = (normalized_slope < -threshold).astype(int) * -1
        short_signal = short_signal.fillna(0)
        return short_signal

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Directional cartridges do not generate magnitude signals by default.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def _calculate_slope(self) -> pd.Series:
        """
        Internal helper to calculate the ATR-Normalized LinReg Slope.
        """
        linreg_length = self.params.get("linreg_length", 14)
        atr_length = 14  # Fixed, not tuned

        def calc(group: pd.DataFrame) -> pd.Series:
            slope_series = ta.slope(close=group['Close'], length=linreg_length)
            atr_series = ta.atr(high=group['High'], low=group['Low'], close=group['Close'], length=atr_length)

            if slope_series is None or slope_series.empty or atr_series is None or atr_series.empty:
                self.logger.warning(f"{self.name}: Slope or ATR calculation returned None or empty.")
                return pd.Series(0, index=group.index)

            normalized_slope = slope_series / atr_series
            return normalized_slope.replace([np.inf, -np.inf], 0).fillna(0)

        return self._run_indicator_per_symbol(calc, "_cached_slope")
