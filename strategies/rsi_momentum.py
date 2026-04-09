import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class RSIMomentum(BaseStrategy):
    """
    RSIMomentum Cartridge: A Directional strategy based on the 
    Relative Strength Index (RSI) momentum vector.
    
    Physics:
    Used here as a pure momentum continuation vector.
    RSI > 50 indicates bullish momentum (Long).
    RSI < 50 indicates bearish momentum (Short).
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns.
    - Output: Pandas DataFrame with 'Signal_Long' (1/0) and 'Signal_Short' (-1/0).
    """

    @property
    def name(self) -> str:
        return "RSI_Momentum"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "rsi_length": {"type": "int", "min": 7, "max": 21}
        }

    def generate_long_signal(self) -> pd.Series:
        """
        Calculates the LONG entry signal: RSI > 50.
        Returns a Series of 1 (Long) or 0 (Flat).
        """
        rsi_series = self._calculate_rsi()
        long_signal = (rsi_series > 50).astype(int).fillna(0)
        return long_signal

    def generate_short_signal(self) -> pd.Series:
        """
        Calculates the SHORT entry signal: RSI < 50.
        Returns a Series of -1 (Short) or 0 (Flat).
        """
        rsi_series = self._calculate_rsi()
        short_signal = (rsi_series < 50).astype(int) * -1
        short_signal = short_signal.fillna(0)
        return short_signal

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Directional cartridges do not generate magnitude signals by default.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def _calculate_rsi(self) -> pd.Series:
        """
        Internal helper to calculate RSI.
        """
        rsi_length = self.params.get("rsi_length", 14)

        def calc(group: pd.DataFrame) -> pd.Series:
            rsi_series = ta.rsi(close=group['Close'], length=rsi_length)
            if rsi_series is None or rsi_series.empty:
                self.logger.warning(f"{self.name}: RSI calculation returned None or empty.")
                return pd.Series(0, index=group.index)
            return rsi_series

        return self._run_indicator_per_symbol(calc, "_cached_rsi")
