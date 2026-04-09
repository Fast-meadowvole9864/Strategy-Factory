import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class KAMASlope(BaseStrategy):
    """
    KAMASlope Cartridge: A Directional strategy based on the 
    Normalized Slope of Kaufman's Adaptive Moving Average (KAMA Slope / ATR).
    """

    @property
    def name(self) -> str:
        return "KAMA_Slope"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "kama_length": {"type": "int", "min": 10, "max": 100},
            "slope_threshold": {"type": "float", "min": 0.001, "max": 0.05}
        }

    def generate_long_signal(self) -> pd.Series:
        normalized_slope = self._calculate_slope()
        threshold = self.params.get("slope_threshold", 0.0)
        
        long_signal = (normalized_slope > threshold).astype(int).fillna(0)
        return long_signal

    def generate_short_signal(self) -> pd.Series:
        normalized_slope = self._calculate_slope()
        threshold = self.params.get("slope_threshold", 0.0)
        
        short_signal = (normalized_slope < -threshold).astype(int) * -1
        short_signal = short_signal.fillna(0)
        return short_signal

    def generate_magnitude_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def _calculate_slope(self) -> pd.Series:
        kama_length = self.params.get("kama_length", 10)
        slope_length = self.params.get("slope_length", 1)
        atr_length = 14  # Fixed, not tuned

        def calc(group: pd.DataFrame) -> pd.Series:
            kama_series = ta.kama(close=group['Close'], length=kama_length)
            if kama_series is None or kama_series.empty:
                return pd.Series(0, index=group.index)

            slope_series = ta.slope(close=kama_series, length=slope_length)
            atr_series = ta.atr(high=group['High'], low=group['Low'], close=group['Close'], length=atr_length)

            if slope_series is None or slope_series.empty or atr_series is None or atr_series.empty:
                self.logger.warning(f"{self.name}: Slope or ATR calculation returned None or empty.")
                return pd.Series(0, index=group.index)

            normalized_slope = slope_series / atr_series
            return normalized_slope.replace([np.inf, -np.inf], 0).fillna(0)

        return self._run_indicator_per_symbol(calc, "_cached_slope")
