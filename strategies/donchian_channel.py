import pandas as pd
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class DonchianChannel(BaseStrategy):
    """
    DonchianChannel Cartridge: A famous Trend-Following continuous indicator.
    
    Physics (The Turtle Strategy):
    Tracks the Highest High and Lowest Low over a trailing N-bar window.
    - Long State: Established when Close breaks the Upper Band. Holds until Lower Band is broken.
    - Short State: Established when Close breaks the Lower Band. Holds until Upper Band is broken.
    
    This provides a massive, continuous "Trend State" vector that pairs 
    beautifully with momentum oscillators natively via the Dynamic Fusion framework.
    """

    @property
    def name(self) -> str:
        return "Donchian_Channel"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dc_length": {"type": "int", "min": 20, "max": 200}
        }

    def _get_trend_state(self) -> pd.Series:
        """
        Calculates the continuous trend state strictly without lookahead bias.
        .shift(1) is applied to the rolling channels so the current bar's High/Low 
        doesn't instantly expand the band it's being compared against.
        """
        dc_length = self.params.get("dc_length", 20)
        
        # Neurotrader precise replication: use 'Close' and (length - 1)
        lookback = max(1, dc_length - 1)

        def calc(group: pd.DataFrame) -> pd.Series:
            # Shift(1) guarantees zero lookahead bias within each symbol slice.
            upper_band = group['Close'].rolling(window=lookback).max().shift(1)
            lower_band = group['Close'].rolling(window=lookback).min().shift(1)

            state = pd.Series(float("nan"), index=group.index, dtype="float64")
            state.loc[group['Close'] > upper_band] = 1.0
            state.loc[group['Close'] < lower_band] = -1.0

            # Hold the last breakout state, but restart cleanly for each symbol.
            return state.ffill().fillna(0.0)

        return self._run_indicator_per_symbol(calc, "_cached_state")

    def generate_long_signal(self) -> pd.Series:
        """
        Returns exactly 1 when Donchian Trend is Long, else 0.
        """
        state = self._get_trend_state()
        return (state == 1).astype(int)

    def generate_short_signal(self) -> pd.Series:
        """
        Returns exactly -1 when Donchian Trend is Short, else 0.
        """
        state = self._get_trend_state()
        # Ensure it returns -1 exactly
        return (state == -1).astype(int) * -1

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Directional cartridges do not generate magnitude shields.
        """
        return pd.Series(0, index=self.df.index)
