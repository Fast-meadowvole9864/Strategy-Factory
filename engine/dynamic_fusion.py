import pandas as pd
import logging
from typing import List, Type, Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

def create_dynamic_cartridge(cartridge_classes: List[Type[BaseStrategy]]) -> Type[BaseStrategy]:
    """
    Class Factory: Dynamically generates a Super-Cartridge that combines multiple
    BaseStrategy cartridges using pure boolean confluence (AND logic).
    """
    
    # 1. Inspect properties of all incoming classes using a dummy dataframe
    dummy_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Taker_Buy_Vol"])
    dummies = [cls(dummy_df) for cls in cartridge_classes]
    
    _combined_name = "_".join([d.name for d in dummies])
    
    # A combined system is Directional if AT LEAST ONE component is Directional.
    # Otherwise, it remains purely Magnitude.
    if any(d.type == "Directional" for d in dummies):
        _combined_type = "Directional"
    else:
        _combined_type = "Magnitude"
        
    _combined_param_space = {}
    for d in dummies:
        if hasattr(d, "param_space"):
            for key, constraints in d.param_space.items():
                # Namespace prefix: e.g., "EMA_Slope::ema_length"
                prefixed_key = f"{d.name}::{key}"
                _combined_param_space[prefixed_key] = constraints

    class DynamicFusionCartridge(BaseStrategy):
        @property
        def name(self) -> str:
            return _combined_name

        @property
        def type(self) -> Literal["Directional", "Magnitude"]:
            return _combined_type

        @property
        def param_space(self) -> Dict[str, Dict[str, Any]]:
            return _combined_param_space

        def __init__(self, df: pd.DataFrame, params: Dict[str, Any] = None):
            super().__init__(df, params)
            self.instances = []
            
            # Un-prefix params and instantiate sub-cartridges
            safe_params = params or {}
            for cls in cartridge_classes:
                dummy_name = cls(dummy_df).name
                prefix = f"{dummy_name}::"
                sub_params = {}
                for k, v in safe_params.items():
                    if k.startswith(prefix):
                        sub_params[k.replace(prefix, "")] = v
                        
                self.instances.append(cls(self.df, sub_params))

        def generate_long_signal(self) -> pd.Series:
            signal = pd.Series(1, index=self.df.index)
            for instance in self.instances:
                if instance.type == "Directional":
                    # For directional, must be 1 (Long)
                    signal = signal & (instance.generate_long_signal() == 1)
                elif instance.type == "Magnitude":
                    # For magnitude, must be 1 (Active)
                    signal = signal & (instance.generate_magnitude_signal() == 1)
            return signal.astype(int).fillna(0)

        def generate_short_signal(self) -> pd.Series:
            signal = pd.Series(1, index=self.df.index)
            for instance in self.instances:
                if instance.type == "Directional":
                    # For directional, must be -1 (Short)
                    signal = signal & (instance.generate_short_signal() == -1)
                elif instance.type == "Magnitude":
                    # For magnitude, must be 1 (Active)
                    signal = signal & (instance.generate_magnitude_signal() == 1)
            # Short signal output contract must be -1/0
            return (signal.astype(int) * -1).fillna(0)

        def generate_magnitude_signal(self) -> pd.Series:
            # If the fusion is purely Magnitude filters
            if self.type == "Magnitude":
                signal = pd.Series(1, index=self.df.index)
                for instance in self.instances:
                    signal = signal & (instance.generate_magnitude_signal() == 1)
                return signal.astype(int).fillna(0)
            else:
                # If it's a Directional system, Magnitude returns 0s
                return pd.Series(0, index=self.df.index)

    return DynamicFusionCartridge
