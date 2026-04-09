import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Literal
from strategies.base_strategy import BaseStrategy

class CVDSlope(BaseStrategy):
    """
    CVDSlope Cartridge: A Directional strategy based on Cumulative Volume Delta (CVD).
    
    Physics:
    Cumulative delta approximated from candle body proportion OR 
    calculated from Taker Buy Volume if available.
    Signal derived from the net change (slope) of the cumulative line over a lookback.
    
    API Contract B Alignment:
    - Input: Pandas DataFrame with OHLCV columns (and optional 'taker_buy_vol').
    - Output: Pandas DataFrame with 'Signal_Long' (1/0) and 'Signal_Short' (-1/0).
    """

    @property
    def name(self) -> str:
        return "CVD_Slope"

    @property
    def type(self) -> Literal["Directional"]:
        return "Directional"

    @property
    def param_space(self) -> Dict[str, Dict[str, Any]]:
        return {
            "cvd_lookback": {"type": "int", "min": 3, "max": 50}
        }

    def generate_long_signal(self) -> pd.Series:
        """
        Calculates the LONG entry signal: CVD[t] - CVD[t-lookback] > 0.
        Returns a Series of 1 (Long) or 0 (Flat).
        """
        cvd_diff = self._calculate_cvd_diff()
        long_signal = (cvd_diff > 0).astype(int).fillna(0)
        return long_signal

    def generate_short_signal(self) -> pd.Series:
        """
        Calculates the SHORT entry signal: CVD[t] - CVD[t-lookback] < 0.
        Returns a Series of -1 (Short) or 0 (Flat).
        """
        cvd_diff = self._calculate_cvd_diff()
        short_signal = (cvd_diff < 0).astype(int) * -1
        short_signal = short_signal.fillna(0)
        return short_signal

    def generate_magnitude_signal(self) -> pd.Series:
        """
        Directional cartridges do not generate magnitude signals by default.
        Returns a series of zeros.
        """
        return pd.Series(0, index=self.df.index)

    def _calculate_cvd_diff(self) -> pd.Series:
        """
        Internal helper to calculate CVD and its net change over the lookback.
        Implements Daily Reset at 00:00 UTC while maintaining index alignment.
        """
        if hasattr(self, '_cached_cvd'):
            return self._cached_cvd

        cvd_lookback = self.params.get("cvd_lookback", 14)
        bars_per_day = self._infer_bars_per_day()
        if bars_per_day and bars_per_day > 1:
            max_lookback = bars_per_day - 1
            if cvd_lookback > max_lookback:
                self.logger.warning(
                    f"{self.name}: Capping cvd_lookback from {cvd_lookback} to {max_lookback} "
                    f"to respect the daily reset on the inferred timeframe."
                )
                cvd_lookback = max_lookback
        
        # 1. Use temporary dates for grouping to avoid destroying index alignment
        if 'timestamp' in self.df.columns:
            temp_dates = pd.to_datetime(self.df['timestamp'], unit='ms', utc=True).dt.date
        elif isinstance(self.df.index, pd.DatetimeIndex):
            temp_dates = self.df.index.date
        else:
            self.logger.warning(f"{self.name}: No timestamp column found and index is not DatetimeIndex.")
            return pd.Series(0, index=self.df.index)

        # 2. Delta Calculation
        taker_col = next((c for c in self.df.columns if c.lower() in ['taker_buy_vol', 'takerbuyvolume', 'taker_buy_volume']), None)
                
        if taker_col:
            # Precise calculation: Delta = 2 * Taker Buy - Volume
            delta = 2 * self.df[taker_col] - self.df['Volume']
        else:
            # Fallback approximation: volume * (close - open) / (high - low)
            range_ext = (self.df['High'] - self.df['Low']).replace(0, 1e-9)
            delta = self.df['Volume'] * (self.df['Close'] - self.df['Open']) / range_ext
            
        # 3. The Daily Reset Protocol: Vectorized GroupBy + CumSum
        # Group by both symbol and UTC date so multi-asset portfolios do not bleed
        # same-day CVD state from one asset into another.
        symbol_keys = (
            self.df["symbol"]
            if "symbol" in self.df.columns
            else pd.Series("__single_asset__", index=self.df.index)
        )
        group_keys = [symbol_keys, temp_dates]
        cvd_series = delta.groupby(group_keys).cumsum()
        
        # 4. Slope Calculation: Diff WITHIN the group to avoid the "Midnight Cliff"
        # This ensures we don't compare today's reset CVD against yesterday's un-reset CVD
        # and also keeps per-asset state fully isolated.
        cvd_diff = cvd_series.groupby(group_keys).diff(cvd_lookback)
        
        self._cached_cvd = cvd_diff.fillna(0)
        return self._cached_cvd

    def _infer_bars_per_day(self) -> int | None:
        """
        Infer the active timeframe from timestamps so the daily-reset lookback
        cannot exceed the number of bars available inside a UTC day.
        """
        if 'timestamp' in self.df.columns:
            ts_series = pd.Series(self.df['timestamp'], index=self.df.index)
            if 'symbol' in self.df.columns and not self.df.empty:
                first_symbol = self.df['symbol'].iloc[0]
                ts_series = ts_series[self.df['symbol'] == first_symbol]

            diffs = ts_series.diff().dropna()
            diffs = diffs[diffs > 0]
            if diffs.empty:
                return None

            tf_ms = int(diffs.median())
            if tf_ms <= 0:
                return None

            return max(1, int(round((24 * 60 * 60 * 1000) / tf_ms)))

        if isinstance(self.df.index, pd.DatetimeIndex):
            idx_series = pd.Series(self.df.index, index=self.df.index)
            if 'symbol' in self.df.columns and not self.df.empty:
                first_symbol = self.df['symbol'].iloc[0]
                idx_series = idx_series[self.df['symbol'] == first_symbol]

            diffs = idx_series.diff().dropna()
            diffs = diffs[diffs > pd.Timedelta(0)]
            if diffs.empty:
                return None

            tf_seconds = diffs.median().total_seconds()
            if tf_seconds <= 0:
                return None

            return max(1, int(round((24 * 60 * 60) / tf_seconds)))

        return None
