import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class VWAP24h(BaseStrategy):
    """
    VWAP24h Cartridge: A Directional strategy based on the Rolling 24h VWAP.
    
    Physics:
    The institutional baseline for execution quality. 
    Close > VWAP indicates a bullish state (Long).
    Close < VWAP indicates a bearish state (Short).
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns.
    - Output: Pandas DataFrame with 'Signal_Long' (1/0) and 'Signal_Short' (-1/0).
    """

    @property
    def name(self) -> str:
        return "VWAP_24h"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Rolling 24h VWAP has no free variables according to blueprint.
        However, the lookback is implicitly fixed to 24h.
        """
        return {}

    def generate_long_signal(self) -> pd.Series:
        """
        Calculates the LONG entry signal: Close > VWAP.
        Returns a Series of 1 (Long) or 0 (Flat).
        """
        vwap_series = self._calculate_vwap()
        long_signal = (self.df['Close'] > vwap_series).astype(int).fillna(0)
        return long_signal

    def generate_short_signal(self) -> pd.Series:
        """
        Calculates the SHORT entry signal: Close < VWAP.
        Returns a Series of -1 (Short) or 0 (Flat).
        """
        vwap_series = self._calculate_vwap()
        short_signal = (self.df['Close'] < vwap_series).astype(int) * -1
        short_signal = short_signal.fillna(0)
        return short_signal

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Directional cartridges do not generate magnitude signals by default.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def _calculate_vwap(self) -> pd.Series:
        """
        Internal helper to calculate rolling 24h VWAP.
        """
        window = self.params.get("vwap_window", 96)

        def calc(group: pd.DataFrame) -> pd.Series:
            pv = group['Close'] * group['Volume']
            rolling_pv = pv.rolling(window=window).sum()
            rolling_vol = group['Volume'].rolling(window=window).sum()
            vwap_series = rolling_pv / rolling_vol
            if vwap_series is None or vwap_series.empty:
                self.logger.warning(f"{self.name}: VWAP calculation returned None or empty.")
                return pd.Series(0, index=group.index)
            return vwap_series

        return self._run_indicator_per_symbol(calc, "_cached_vwap")
