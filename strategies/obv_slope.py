import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class OBVSlope(BaseStrategy):
    """
    OBVSlope Cartridge: A Directional strategy based on the 
    On-Balance Volume (OBV) trend.
    
    Physics:
    Cumulative volume sum — adds full bar volume on up-close, 
    subtracts on down-close. Signal derived from the net change 
    (slope) of the cumulative line over a lookback.
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns.
    - Output: Pandas DataFrame with 'Signal_Long' (1/0) and 'Signal_Short' (-1/0).
    """

    @property
    def name(self) -> str:
        return "OBV_Slope"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "obv_lookback": {"type": "int", "min": 14, "max": 50}
        }

    def generate_long_signal(self) -> pd.Series:
        """
        Calculates the LONG entry signal: OBV[t] - OBV[t-lookback] > 0.
        Returns a Series of 1 (Long) or 0 (Flat).
        """
        obv_diff = self._calculate_obv_diff()
        long_signal = (obv_diff > 0).astype(int).fillna(0)
        return long_signal

    def generate_short_signal(self) -> pd.Series:
        """
        Calculates the SHORT entry signal: OBV[t] - OBV[t-lookback] < 0.
        Returns a Series of -1 (Short) or 0 (Flat).
        """
        obv_diff = self._calculate_obv_diff()
        short_signal = (obv_diff < 0).astype(int) * -1
        short_signal = short_signal.fillna(0)
        return short_signal

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Directional cartridges do not generate magnitude signals by default.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def _calculate_obv_diff(self) -> pd.Series:
        """
        Internal helper to calculate the OBV net change over the lookback.
        """
        obv_lookback = self.params.get("obv_lookback", 14)

        def calc(group: pd.DataFrame) -> pd.Series:
            obv_series = ta.obv(close=group['Close'], volume=group['Volume'])
            if obv_series is None or obv_series.empty:
                self.logger.warning(f"{self.name}: OBV calculation returned None or empty.")
                return pd.Series(0, index=group.index)
            return obv_series.diff(obv_lookback)

        return self._run_indicator_per_symbol(calc, "_cached_obv")
