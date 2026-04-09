import pandas as pd
import numpy as np
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class CoinFlipRegime(BaseStrategy):
    """
    CoinFlipRegime Cartridge: A "Garbage" baseline for Red Team testing.
    
    Physics:
    Generates a purely random 1/0 signal based on a 'luck' parameter.
    Used to verify that the Permutation Engine correctly rejects noise.
    """

    @property
    def name(self) -> str:
        return "CoinFlip_Regime"

    @property
    def type(self) -> Literal["Magnitude"]:
        return "Magnitude"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "luck_factor": {"type": "int", "min": 1, "max": 100}
        }

    def generate_long_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def generate_short_signal(self) -> pd.Series:
        return pd.Series(0, index=self.df.index)

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Generates a random signal. 
        The 'luck_factor' is a dummy parameter for Optuna to chase.
        """
        np.random.seed(self.params.get("luck_factor", 42))
        random_values = np.random.choice([0, 1], size=len(self.df))
        return pd.Series(random_values, index=self.df.index)
