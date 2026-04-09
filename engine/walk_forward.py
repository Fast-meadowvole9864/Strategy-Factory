import optuna
import polars as pl
import pandas as pd
import numpy as np
from typing import Type, Tuple, List, Dict, Any
from tqdm import tqdm
import logging
from numba import njit
from engine.optimizer import OptunaOptimizer
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

@njit
def simulate_execution_1m(signals_15m: np.ndarray, prices_1m: np.ndarray, sl_pct: float, tp_pct: float, fees_active: bool = False) -> np.ndarray:
    n_prices = prices_1m.shape[0]
    n_signals = signals_15m.shape[0]
    
    sig_idx = 0
    position = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    
    trade_results = np.zeros(n_prices)
    t_count = 0
    
    for p_idx in range(n_prices):
        current_ts = prices_1m[p_idx, 0]
        
        # Asset Boundary Detection: If time flows backwards, we switched to the next coin
        if p_idx > 0 and current_ts < prices_1m[p_idx - 1, 0]:
            position = 0
            
        c_open = prices_1m[p_idx, 1]
        c_high = prices_1m[p_idx, 2]
        c_low = prices_1m[p_idx, 3]
        
        while sig_idx + 1 < n_signals and signals_15m[sig_idx + 1, 0] <= current_ts:
            sig_idx += 1
            
        active_long = signals_15m[sig_idx, 1]
        active_short = signals_15m[sig_idx, 2]
        entered_this_candle = False

        # 1. Evaluate Entries
        if position == 0:
            if active_long == 1:
                position = 1
                entry_price = c_open
                sl_price = entry_price * (1.0 - sl_pct)
                tp_price = entry_price * (1.0 + tp_pct)
                entered_this_candle = True
            elif active_short == -1:
                position = -1
                entry_price = c_open
                sl_price = entry_price * (1.0 + sl_pct)
                tp_price = entry_price * (1.0 - tp_pct)
                entered_this_candle = True
                
        # 2. Evaluate Exits
        if position != 0:
            resolved = False
            pnl = 0.0
            
            if position == 1:
                hit_sl = c_low <= sl_price
                hit_tp = c_high >= tp_price
                if hit_sl and hit_tp:
                    if abs(c_open - sl_price) <= abs(tp_price - c_open):
                        pnl = -sl_pct
                    else:
                        pnl = tp_pct
                    resolved = True
                elif hit_sl:
                    pnl = -sl_pct
                    resolved = True
                elif hit_tp:
                    pnl = tp_pct
                    resolved = True
                elif active_short == -1 and not entered_this_candle:
                    pnl = (c_open - entry_price) / entry_price
                    resolved = True
            elif position == -1:
                hit_sl = c_high >= sl_price
                hit_tp = c_low <= tp_price
                if hit_sl and hit_tp:
                    if abs(sl_price - c_open) <= abs(c_open - tp_price):
                        pnl = -sl_pct
                    else:
                        pnl = tp_pct
                    resolved = True
                elif hit_sl:
                    pnl = -sl_pct
                    resolved = True
                elif hit_tp:
                    pnl = tp_pct
                    resolved = True
                elif active_long == 1 and not entered_this_candle:
                    pnl = (entry_price - c_open) / entry_price
                    resolved = True
                    
            if resolved:
                if fees_active:
                    pnl -= 0.001  # 0.1% round-trip friction (0.05% entry + 0.05% exit slippage/fee)
                trade_results[t_count] = pnl
                t_count += 1
                position = 0
                
    return trade_results[:t_count]

class WalkForwardEngine:
    """
    Module 6: The Walk-Forward Loop.
    Rolling WFO using 15m data for training and 1m data for precise SL/TP execution.
    Implements Sequential SL/TP Optimization.
    """
    def __init__(self, data_15m: pl.DataFrame, data_1m: pl.DataFrame, cartridge_class: Type[BaseStrategy], 
                 train_window: int, test_window: int, optuna_trials: int = 50, sl_tp_trials: int = 20,
                 warmup_bars: int = 500, mode: str = "bar", matrix: list = None, n_jobs: int = -1, fees_active: bool = False):
        self.data_15m = data_15m
        self.data_1m = data_1m
        self.cartridge_class = cartridge_class
        self.train_window = train_window
        self.test_window = test_window
        self.optuna_trials = optuna_trials
        self.sl_tp_trials = sl_tp_trials
        self.warmup_bars = warmup_bars
        self.mode = mode
        self.matrix = matrix if matrix is not None else [4, 8, 12]
        self.n_jobs = n_jobs
        self.fees_active = fees_active

    def optimize_sl_tp(self, arrs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial: optuna.Trial) -> float:
            # Expanded boundaries to support 1m scalping through 1H macro swings
            sl_pct = trial.suggest_float("sl_pct", 0.005, 0.07)
            tp_pct = trial.suggest_float("tp_pct", 0.005, 0.2)
            total_pnl = 0.0
            for s_arr, p_arr in arrs:
                if len(s_arr) > 0 and len(p_arr) > 0:
                    trade_results = simulate_execution_1m(s_arr, p_arr, sl_pct, tp_pct, self.fees_active)
                    total_pnl += float(trade_results.sum())
            return total_pnl
            
        study = optuna.create_study(direction="maximize")
        
        with tqdm(total=self.sl_tp_trials, desc="SL/TP Trials", leave=False) as pbar:
            def callback(study, trial):
                pbar.update(1)
            study.optimize(objective, n_trials=self.sl_tp_trials, callbacks=[callback])
        
        best_sl = study.best_params["sl_pct"]
        best_tp = study.best_params["tp_pct"]
        return best_sl, best_tp

    def _prepare_signals(self, df: pl.DataFrame) -> np.ndarray:
        if "Signal_Long" in df.columns:
            df = df.with_columns([
                pl.col("Signal_Long").shift(1).over("symbol").fill_null(0).alias("Shifted_Long"),
                pl.col("Signal_Short").shift(1).over("symbol").fill_null(0).alias("Shifted_Short")
            ])
            return df.select([
                pl.col("timestamp").cast(pl.Int64), 
                pl.col("Shifted_Long").cast(pl.Int32), 
                pl.col("Shifted_Short").cast(pl.Int32)
            ]).to_numpy()
        elif "Signal_Magnitude" in df.columns:
            df = df.with_columns([
                pl.col("Signal_Magnitude").shift(1).over("symbol").fill_null(0).alias("Shifted_Long")
            ])
            return df.select([
                pl.col("timestamp").cast(pl.Int64), 
                pl.col("Shifted_Long").cast(pl.Int32), 
                pl.lit(0).cast(pl.Int32).alias("Shifted_Short")
            ]).to_numpy()
        else:
            raise ValueError("No Signal columns found in output.")

    def _prepare_prices(self, df_15m: pl.DataFrame, symbol: str) -> np.ndarray:
        min_ts = df_15m["timestamp"].min()
        max_ts = df_15m["timestamp"].max()
        
        # Determine timeframe in ms
        tf_ms = df_15m["timestamp"].diff().drop_nulls().median()
        if tf_ms is None or tf_ms <= 0:
            tf_ms = 15 * 60 * 1000  # fallback to 15m
            
        prices_1m_filtered = self.data_1m.filter(
            (pl.col("symbol") == symbol) & (pl.col("timestamp") >= min_ts) & (pl.col("timestamp") < max_ts + tf_ms)
        )
        return prices_1m_filtered.select([
            pl.col("timestamp").cast(pl.Int64),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close")
        ]).to_numpy()

    def run(self, seed_params: dict = None) -> Tuple[np.ndarray, pl.DataFrame, List[Dict[str, Any]]]:
        # Determine timeframe in milliseconds
        tf_ms = self.data_15m["timestamp"].diff().drop_nulls().median()
        if tf_ms is None or tf_ms <= 0:
            tf_ms = 15 * 60 * 1000  # fallback to 15m
            
        train_ms = self.train_window * tf_ms
        test_ms = self.test_window * tf_ms
        warmup_ms = self.warmup_bars * tf_ms
        
        # We need the absolute min and max timestamps across the entire concatenated dataset
        min_ts = self.data_15m["timestamp"].min()
        max_ts = self.data_15m["timestamp"].max()
        
        all_signals_dfs = []
        all_trade_results = []
        rolling_parameters = []
        
        logger.info("Starting Sequential Time-Based Rolling WFO...")
        
        total_rolls = int(np.ceil((max_ts - min_ts - train_ms) / test_ms))
        if total_rolls <= 0:
            raise ValueError("Data too short for train_window")
            
        current_start_ts = min_ts
        
        with tqdm(total=total_rolls, desc="WFO Rolls") as pbar:
            while current_start_ts + train_ms < max_ts:
                train_end_ts = current_start_ts + train_ms
                test_end_ts = min(train_end_ts + test_ms, max_ts + tf_ms) # Add tf_ms to ensure we capture the last bar
                
                train_slice = self.data_15m.filter(
                    (pl.col("timestamp") >= current_start_ts) & 
                    (pl.col("timestamp") < train_end_ts)
                )
                
                test_slice = self.data_15m.filter(
                    (pl.col("timestamp") >= train_end_ts) & 
                    (pl.col("timestamp") < test_end_ts)
                )
                
                if len(test_slice) == 0:
                    break
                    
                # Step A: OptunaOptimizer on train_slice
                current_seed = seed_params if current_start_ts == min_ts else None
                optimizer = OptunaOptimizer(train_slice, self.cartridge_class, n_trials=self.optuna_trials, seed_params=current_seed, mode=self.mode, matrix=self.matrix, n_jobs=self.n_jobs)
                best_params, _, _ = optimizer.run()
                
                # Step B: Optimize SL/TP on train_slice
                train_cartridge = self.cartridge_class(train_slice.to_pandas(), best_params)
                train_signals_pandas = train_cartridge.run()
                train_signals_polars = pl.from_pandas(train_signals_pandas)
                train_with_signals = pl.concat([train_slice, train_signals_polars], how="horizontal")
                
                arrs_train = []
                for sym in train_with_signals["symbol"].unique():
                    s_df = train_with_signals.filter(pl.col("symbol") == sym)
                    s_arr = self._prepare_signals(s_df)
                    p_arr = self._prepare_prices(train_slice, sym)
                    arrs_train.append((s_arr, p_arr))
                
                optimal_sl, optimal_tp = self.optimize_sl_tp(arrs_train)
                
                # Save rolling parameter set
                roll_params = best_params.copy()
                roll_params["sl_pct"] = optimal_sl
                roll_params["tp_pct"] = optimal_tp
                
                # Record the start of the test window for this parameter set
                roll_params["_test_start_ts"] = float(train_end_ts)
                
                # Step C: OOS Execution on test_slice (with warmup)
                pad_start_ts = max(min_ts, train_end_ts - warmup_ms)
                padded_test_slice = self.data_15m.filter(
                    (pl.col("timestamp") >= pad_start_ts) & 
                    (pl.col("timestamp") < test_end_ts)
                )
                
                test_cartridge = self.cartridge_class(padded_test_slice.to_pandas(), best_params)
                test_signals_pandas = test_cartridge.run()
                
                test_signals_polars = pl.from_pandas(test_signals_pandas)
                padded_test_with_signals = pl.concat([padded_test_slice, test_signals_polars], how="horizontal")
                
                test_with_signals = padded_test_with_signals.filter(
                    pl.col("timestamp") >= train_end_ts
                )
                
                all_signals_dfs.append(test_with_signals)
                
                test_trade_results_list = []
                for sym in test_with_signals["symbol"].unique():
                    s_padded_df = padded_test_with_signals.filter(pl.col("symbol") == sym)
                    p_arr = self._prepare_prices(test_slice, sym)
                    s_padded_arr = self._prepare_signals(s_padded_df)
                    
                    # Filter the numpy array down to the actual test window boundary for execution
                    s_arr = s_padded_arr[s_padded_arr[:, 0] >= train_end_ts]
                    
                    if len(s_arr) > 0 and len(p_arr) > 0:
                        res = simulate_execution_1m(s_arr, p_arr, optimal_sl, optimal_tp, self.fees_active)
                        test_trade_results_list.append(res)
                        
                if test_trade_results_list:
                    test_trade_results = np.concatenate(test_trade_results_list)
                    all_trade_results.append(test_trade_results)
                    roll_params["roll_pnl"] = float(test_trade_results.sum())
                    roll_params["roll_trades"] = len(test_trade_results)
                else:
                    roll_params["roll_pnl"] = 0.0
                    roll_params["roll_trades"] = 0
                    
                rolling_parameters.append(roll_params)
                
                current_start_ts += test_ms
                pbar.update(1)
                
        if not all_signals_dfs:
            raise ValueError("No OOS windows generated. Check window sizes.")
            
        oos_df = pl.concat(all_signals_dfs)
        final_trade_results = np.concatenate(all_trade_results) if all_trade_results else np.array([])
        
        return final_trade_results, oos_df, rolling_parameters
