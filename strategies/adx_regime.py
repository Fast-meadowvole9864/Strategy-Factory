import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class ADXRegime(BaseStrategy):
    """
    ADXRegime Cartridge: A Magnitude/Regime filter based on the 
    Average Directional Index (ADX).
    
    Physics:
    Measures the absolute kinetic strength of a trend. 
    ADX > 25 is a mathematical baseline for momentum.
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns.
    - Output: Pandas DataFrame with 'Signal_Magnitude' (1 for active, 0 for inactive).
    """

    @property
    def name(self) -> str:
        return "ADX_Regime"

    @property
    def type(self) -> Literal["Magnitude"]:
        return "Magnitude"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "adx_length": {"type": "int", "min": 3, "max": 70},
            "adx_threshold": {"type": "int", "min": 8, "max": 80}
        }

    def generate_long_signal(self) -> pd.Series:
        """
        Magnitude cartridges do not generate directional signals.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def generate_short_signal(self) -> pd.Series:
        """
        Magnitude cartridges do not generate directional signals.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Calculates the ADX and returns 1 if ADX > 25, else 0.
        
        Free Variable: 'adx_length' (default: 14)
        Threshold: 25
        """
        adx_length = self.params.get("adx_length", 14)
        
        adx_threshold = self.params.get("adx_threshold", 25)

        def calc(group: pd.DataFrame) -> pd.Series:
            adx_df = ta.adx(
                high=group['High'],
                low=group['Low'],
                close=group['Close'],
                length=adx_length
            )
            if adx_df is None or adx_df.empty:
                self.logger.warning(f"{self.name}: ADX calculation returned None or empty.")
                return pd.Series(0, index=group.index)
            return adx_df.iloc[:, 0]

        adx_series = self._run_indicator_per_symbol(calc, "_cached_adx")
        
        # Generate signal: 1 if ADX > threshold, else 0
        # Note: ADX usually needs some bars to 'warm up', resulting in NaN initially.
        # We fill NaNs with 0 to ensure index alignment and no crashes.
        magnitude_signal = (adx_series > adx_threshold).astype(int).fillna(0)
        
        return magnitude_signal
