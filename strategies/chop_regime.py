import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class CHOPRegime(BaseStrategy):
    """
    CHOPRegime Cartridge: A Magnitude/Regime filter based on the 
    Choppiness Index (CHOP).
    
    Physics:
    Calculates fractal dimension to identify price compression.
    CHOP < 38.2 is a Fibonacci threshold confirming a directional breakout (Regime Active).
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns.
    - Output: Pandas DataFrame with 'Signal_Magnitude' (1 for active, 0 for inactive).
    """

    @property
    def name(self) -> str:
        return "CHOP_Regime"

    @property
    def type(self) -> Literal["Magnitude"]:
        return "Magnitude"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "chop_length": {"type": "int", "min": 10, "max": 50}
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
        Calculates the CHOP Index and returns 1 if CHOP < 38.2, else 0.
        
        Free Variable: 'chop_length' (default: 14)
        Threshold: 38.2
        """
        chop_length = self.params.get("chop_length", 14)

        def calc(group: pd.DataFrame) -> pd.Series:
            chop_series = ta.chop(
                high=group['High'],
                low=group['Low'],
                close=group['Close'],
                length=chop_length
            )
            if chop_series is None or chop_series.empty:
                self.logger.warning(f"{self.name}: CHOP calculation returned None or empty.")
                return pd.Series(0, index=group.index)
            return chop_series

        chop_series = self._run_indicator_per_symbol(calc, "_cached_chop")
            
        # Generate signal: 1 if CHOP < 38.2, else 0
        # Fill NaNs with 0 (warmup period)
        magnitude_signal = (chop_series < 38.2).astype(int).fillna(0)
        
        return magnitude_signal
