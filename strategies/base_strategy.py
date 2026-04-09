import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Literal, Callable, Optional

class BaseStrategy(ABC):
    """
    BaseStrategy: The abstract foundation for all quantitative 'Cartridges'.
    
    All strategy cartridges must inherit from this class and implement its 
    abstract properties and methods. This ensures consistency for the 
    Sandbox Engine and Optimizer.
    
    API Contract B Alignment:
    - Input: Pandas DataFrame (from Polars/PyArrow).
    - Output: Pandas DataFrame with Signal columns aligned to the input index.
    """

    def __init__(self, df: pd.DataFrame, params: Dict[str, Any] = None):
        """
        Initializes the base strategy.
        
        Args:
            df (pd.DataFrame): The OHLCV data. 
            params (Dict[str, Any], optional): Hyperparameters for the strategy. 
                                              Defaults to empty dict.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
            
        self.df = df
        
        # Canonical Adapter: Force lowercase OHLCV to uppercase for Cartridge compatibility
        self.df.rename(columns={
            "open": "Open", 
            "high": "High", 
            "low": "Low", 
            "close": "Close", 
            "volume": "Volume", 
            "taker_buy_vol": "Taker_Buy_Vol"
        }, inplace=True)
        
        self.params = params or {}
        
        # Initialize logger for the specific subclass
        self.logger = logging.getLogger(self.__class__.__name__)

    def _run_indicator_per_symbol(
        self,
        calculator: Callable[[pd.DataFrame], pd.Series],
        cache_attr: Optional[str] = None,
    ) -> pd.Series:
        """
        Run an indicator calculation independently for each symbol slice, then
        stitch the results back together in the original row order.
        """
        if cache_attr is not None and hasattr(self, cache_attr):
            return getattr(self, cache_attr)

        def compute(group: pd.DataFrame) -> pd.Series:
            result = calculator(group)
            if result is None:
                return pd.Series(0.0, index=group.index)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=group.index)
            return result.reindex(group.index)

        if "symbol" not in self.df.columns:
            series = compute(self.df)
        else:
            work_df = self.df.copy()
            work_df["_row_pos"] = range(len(work_df))
            output = pd.Series(0.0, index=range(len(work_df)), dtype="float64")

            for _, group in work_df.groupby("symbol", sort=False):
                row_positions = group["_row_pos"].to_list()
                group_df = group.drop(columns="_row_pos")
                group_result = compute(group_df)
                output.iloc[row_positions] = group_result.to_numpy()

            output.index = self.df.index
            series = output

        series = series.replace([float("inf"), float("-inf")], 0)
        if cache_attr is not None:
            setattr(self, cache_attr, series)
        return series

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the strategy (e.g., 'ADX_Regime'). 
        Used for reporting and dynamic loading.
        """
        pass

    @property
    @abstractmethod
    def type(self) -> Literal["Directional", "Magnitude"]:
        """
        Defines the mathematical nature of the cartridge.
        - 'Directional': Generates Signal_Long (1/0) and Signal_Short (-1/0).
        - 'Magnitude': Generates Signal_Magnitude (1/0).
        """
        pass

    @property
    @abstractmethod
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Defines the hyperparameter space for the optimizer.
        Format: {'param_name': {'type': 'int', 'min': 10, 'max': 50}}
        """
        pass

    @abstractmethod
    def generate_long_signal(self) -> pd.Series:
        """
        Calculates the LONG entry signal.
        Must return a Series of 1 (Long) or 0 (Flat).
        """
        pass

    @abstractmethod
    def generate_short_signal(self) -> pd.Series:
        """
        Calculates the SHORT entry signal.
        Must return a Series of -1 (Short) or 0 (Flat).
        """
        pass

    @abstractmethod
    def generate_magnitude_signal(self) -> pd.Series:
        """
        Calculates the VOLATILITY/MAGNITUDE expansion signal.
        Must return a Series of 1 (Active) or 0 (Inactive).
        """
        pass

    def run(self) -> pd.DataFrame:
        """
        Executes the signal generation logic and returns the signals in 
        a DataFrame structure aligned with Contract B.
        
        Returns:
            pd.DataFrame: A DataFrame containing the appropriate signal columns.
        """
        self.logger.debug(f"Running strategy: {self.name} | Type: {self.type}")

        # Verify indices are aligned
        signals_df = pd.DataFrame(index=self.df.index)
        
        try:
            if self.type == "Directional":
                long_signal = self.generate_long_signal()
                short_signal = self.generate_short_signal()
                
                # Validation: Ensure they are Series and match the length
                if not isinstance(long_signal, pd.Series) or not isinstance(short_signal, pd.Series):
                    raise TypeError(f"{self.name}: Signals must be pandas Series.")
                
                signals_df["Signal_Long"] = long_signal
                signals_df["Signal_Short"] = short_signal
                
            elif self.type == "Magnitude":
                magnitude_signal = self.generate_magnitude_signal()
                
                if not isinstance(magnitude_signal, pd.Series):
                    raise TypeError(f"{self.name}: Signal must be a pandas Series.")
                    
                signals_df["Signal_Magnitude"] = magnitude_signal
            else:
                raise ValueError(f"Invalid strategy type: {self.type}")
                
        except Exception as e:
            self.logger.error(f"Error during signal generation in {self.name}: {e}")
            raise

        return signals_df
