import os

# EXPLICIT WINDOWS MULTIPROCESSING SHIELD
# This must execute BEFORE any C-based data libraries are imported.
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import polars as pl
import numpy as np
from joblib import Parallel, delayed
import logging
from tqdm import tqdm
from typing import Type, Tuple, List
from statsmodels.stats.multitest import multipletests
from engine.optimizer import OptunaOptimizer
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class PermutationEngine:
    """
    Module 5: The Permutation Matrix.
    Implements 2D Matrix Vectorization for Monte Carlo Data Permutation
    using the 'Logarithmic Neurotrader' method.
    """
    def __init__(self, data: pl.DataFrame, cartridge_class: Type[BaseStrategy], n_permutations: int = 1000, optuna_trials: int = 10, target_direction: str = "Total", mode: str = "bar", matrix: list = None, n_jobs: int = -1):
        self.data = data
        self.cartridge_class = cartridge_class
        self.n_permutations = n_permutations
        self.optuna_trials = optuna_trials
        self.target_direction = target_direction
        self.mode = mode
        self.matrix = matrix if matrix is not None else [4, 8, 12]
        self.n_jobs = n_jobs
        self.dummy_cartridge = self.cartridge_class(pl.DataFrame().to_pandas())
        self.cartridge_type = self.dummy_cartridge.type

    def generate_synthetic_data(self, start_index: int = 0) -> pl.DataFrame:
        """
        Implements Logarithmic Neurotrader Monte Carlo Permutation.
        Preserves data up to start_index, shuffles the tail.
        Iterates over unique symbols to strictly prevent cross-asset scrambling.
        """
        close_col = "Close" if "Close" in self.data.columns else "close"
        if close_col not in self.data.columns:
            raise ValueError("Data must contain a close/Close column.")
            
        has_volume = "volume" in self.data.columns
        has_taker = "taker_buy_vol" in self.data.columns
        
        open_col = "Open" if "Open" in self.data.columns else "open"
        high_col = "High" if "High" in self.data.columns else "high"
        low_col = "Low" if "Low" in self.data.columns else "low"
        
        synth_dfs = []
        
        # Iterate over unique assets to ensure independent, safe shuffling without cross-asset bleeding
        for symbol in self.data["symbol"].unique():
            asset_df = self.data.filter(pl.col("symbol") == symbol)
            n_rows = len(asset_df)
            
            if start_index >= n_rows:
                synth_dfs.append(asset_df)
                continue
                
            # 1. Log Transformation initial baseline
            initial_close_log = np.log(asset_df[close_col][0])
            
            # 2. Extraction: Fetch structural arrays.
            gap = asset_df["Gap"].to_numpy()
            move_high = asset_df["Move_High"].to_numpy()
            move_low = asset_df["Move_Low"].to_numpy()
            move_close = asset_df["Move_Close"].to_numpy()
            
            if has_volume: volume_arr = asset_df["volume"].to_numpy()
            if has_taker: taker_arr = asset_df["taker_buy_vol"].to_numpy()
            
            # 3. Shuffle: Scramble Gap and Intra-bar arrays independently (Tail only)
            rng = np.random.default_rng()
            tail_idx_gap = rng.permutation(n_rows - start_index) + start_index
            tail_idx_intra = rng.permutation(n_rows - start_index) + start_index
            
            shuffled_gap = np.concatenate([gap[:start_index], gap[tail_idx_gap]])
            shuffled_move_high = np.concatenate([move_high[:start_index], move_high[tail_idx_intra]])
            shuffled_move_low = np.concatenate([move_low[:start_index], move_low[tail_idx_intra]])
            shuffled_move_close = np.concatenate([move_close[:start_index], move_close[tail_idx_intra]])
            
            if has_volume:
                shuffled_volume = np.concatenate([volume_arr[:start_index], volume_arr[tail_idx_intra]])
            if has_taker:
                shuffled_taker = np.concatenate([taker_arr[:start_index], taker_arr[tail_idx_intra]])
            
            # 4. Reconstruction (Log Addition)
            shuffled_log_return = shuffled_gap + shuffled_move_close
            new_close_log = initial_close_log + np.cumsum(shuffled_log_return)
            
            new_prev_close_log = np.empty(n_rows)
            new_prev_close_log[0] = initial_close_log
            new_prev_close_log[1:] = new_close_log[:-1]
            
            new_open_log = new_prev_close_log + shuffled_gap
            new_high_log = new_open_log + shuffled_move_high
            new_low_log = new_open_log + shuffled_move_low
            
            # 5. Exponentiation
            new_open = np.exp(new_open_log)
            new_high = np.exp(new_high_log)
            new_low = np.exp(new_low_log)
            new_close = np.exp(new_close_log)
            
            synth_cols = [
                pl.Series(open_col, new_open),
                pl.Series(high_col, new_high),
                pl.Series(low_col, new_low),
                pl.Series(close_col, new_close),
                pl.Series("Log_Return", shuffled_log_return)
            ]
            
            if has_volume: synth_cols.append(pl.Series("volume", shuffled_volume))
            if has_taker: synth_cols.append(pl.Series("taker_buy_vol", shuffled_taker))
                
            synth_asset_df = asset_df.with_columns(synth_cols)
            synth_dfs.append(synth_asset_df)
            
        return pl.concat(synth_dfs)

    def run(self, start_index: int = 0) -> Tuple[float, List[float], float]:
        """
        Runs the permutation matrix and calculates the P-Value.
        Returns: (real_benchmark_metric, synthetic_metrics_list, p_value)
        """
        logger.info(f"Running Real Data Benchmark via OptunaOptimizer (Target: {self.target_direction})...")
        real_optimizer = OptunaOptimizer(self.data, self.cartridge_class, n_trials=self.optuna_trials, target_direction=self.target_direction, mode=self.mode, matrix=self.matrix, n_jobs=self.n_jobs)
        _, real_benchmark, _ = real_optimizer.run()
        logger.info(f"Real Data Benchmark Metric: {real_benchmark}")
        
        def worker(i):
            os.environ["POLARS_MAX_THREADS"] = "1"
            synth_df = self.generate_synthetic_data(start_index=start_index)
            synth_optimizer = OptunaOptimizer(synth_df, self.cartridge_class, n_trials=self.optuna_trials, target_direction=self.target_direction, mode=self.mode, matrix=self.matrix, n_jobs=1)
            _, synth_metric, _ = synth_optimizer.run(show_progress_bar=False)
            return synth_metric

        tasks = (delayed(worker)(i) for i in range(self.n_permutations))
        
        synthetic_metrics = []
        for res in tqdm(Parallel(n_jobs=self.n_jobs, backend="loky", return_as="generator")(tasks), total=self.n_permutations, desc="Permutations"):
            synthetic_metrics.append(res)
            
        synthetic_metrics_arr = np.array(synthetic_metrics)
        
        # Calculate raw P-Value: proportion of synthetic paths that equal or beat the real data benchmark
        raw_p_value = np.sum(synthetic_metrics_arr >= real_benchmark) / self.n_permutations
        
        # Apply Benjamini-Hochberg FDR correction
        # Note: If testing only 1 hypothesis (1 strategy), BH doesn't penalize, but it's applied per spec.
        reject, pvals_corrected, _, _ = multipletests([raw_p_value], alpha=0.05, method='fdr_bh')
        final_p_value = pvals_corrected[0]
        
        logger.info(f"Final BH-Corrected P-Value: {final_p_value}")
        
        return float(real_benchmark), synthetic_metrics, float(final_p_value)

    def run_transfer(self, locked_params: dict) -> Tuple[float, List[float], float]:
        """
        Runs the permutation matrix using locked parameters without re-optimization.
        Used for Stage 1.5 Transfer Validation.
        """
        from engine.sandbox_engine import SandboxEngine
        
        logger.info(f"Running Real Data Benchmark via SandboxEngine (Locked Params)...")
        real_engine = SandboxEngine(self.data, self.cartridge_class, locked_params, mode=self.mode, matrix=self.matrix)
        real_results = real_engine.run()
        
        if self.cartridge_type == "Directional":
            if self.mode == "forwardr":
                real_benchmark = real_results.get("Profit_Factor_Forward", 0.0)
            elif self.mode == "eratio":
                real_benchmark = real_results.get("E_Ratio_Mean", 0.0)
            else:
                if self.target_direction == "Total":
                    real_benchmark = real_results.get("Profit_Factor_Total", 0.0)
                elif self.target_direction == "Long":
                    real_benchmark = real_results.get("Profit_Factor_Long", 0.0)
                elif self.target_direction == "Short":
                    real_benchmark = real_results.get("Profit_Factor_Short", 0.0)
        elif self.cartridge_type == "Magnitude":
            real_benchmark = real_results.get("Magnitude_Ratio", 0.0)
        else:
            real_benchmark = 0.0
            
        logger.info(f"Real Data Benchmark Metric: {real_benchmark}")
        
        def worker_transfer(i):
            os.environ["POLARS_MAX_THREADS"] = "1"
            synth_df = self.generate_synthetic_data(start_index=0)
            synth_engine = SandboxEngine(synth_df, self.cartridge_class, locked_params, mode=self.mode, matrix=self.matrix)
            synth_results = synth_engine.run()
            
            if self.cartridge_type == "Directional":
                if self.mode == "forwardr":
                    synth_metric = synth_results.get("Profit_Factor_Forward", 0.0)
                elif self.mode == "eratio":
                    synth_metric = synth_results.get("E_Ratio_Mean", 0.0)
                else:
                    if self.target_direction == "Total":
                        synth_metric = synth_results.get("Profit_Factor_Total", 0.0)
                    elif self.target_direction == "Long":
                        synth_metric = synth_results.get("Profit_Factor_Long", 0.0)
                    elif self.target_direction == "Short":
                        synth_metric = synth_results.get("Profit_Factor_Short", 0.0)
            elif self.cartridge_type == "Magnitude":
                synth_metric = synth_results.get("Magnitude_Ratio", 0.0)
            else:
                synth_metric = 0.0
                
            return synth_metric
            
        tasks = (delayed(worker_transfer)(i) for i in range(self.n_permutations))
        
        synthetic_metrics = []
        for res in tqdm(Parallel(n_jobs=self.n_jobs, backend="loky", return_as="generator")(tasks), total=self.n_permutations, desc="Permutations"):
            synthetic_metrics.append(res)
            
        synthetic_metrics_arr = np.array(synthetic_metrics)
        
        raw_p_value = np.sum(synthetic_metrics_arr >= real_benchmark) / self.n_permutations
        
        reject, pvals_corrected, _, _ = multipletests([raw_p_value], alpha=0.05, method='fdr_bh')
        final_p_value = pvals_corrected[0]
        
        logger.info(f"Final BH-Corrected P-Value: {final_p_value}")
        
        return float(real_benchmark), synthetic_metrics, float(final_p_value)

    def run_wfo(self, start_index: int, train_window: int, test_window: int, data_1m: pl.DataFrame, seed_params: dict = None, sl_tp_trials: int = 20) -> Tuple[float, List[float], float]:
        """
        Runs the WFO loop on permuted data.
        """
        from engine.walk_forward import WalkForwardEngine
        
        logger.info(f"Running Real WFO Benchmark...")
        real_engine = WalkForwardEngine(self.data, data_1m, self.cartridge_class, train_window=train_window, test_window=test_window, optuna_trials=self.optuna_trials, sl_tp_trials=sl_tp_trials, mode=self.mode, matrix=self.matrix, n_jobs=self.n_jobs)
        real_trade_results, _, _ = real_engine.run(seed_params=seed_params)
        
        wins = real_trade_results[real_trade_results > 0]
        losses = real_trade_results[real_trade_results < 0]
        gross_win = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        real_benchmark = float(gross_win / gross_loss) if gross_loss > 0 else 0.0
        
        logger.info(f"Real WFO Benchmark Metric: {real_benchmark}")
        
        def worker_wfo(i):
            synth_df = self.generate_synthetic_data(start_index=start_index)
            synth_engine = WalkForwardEngine(synth_df, data_1m, self.cartridge_class, train_window=train_window, test_window=test_window, optuna_trials=self.optuna_trials, sl_tp_trials=sl_tp_trials, mode=self.mode, matrix=self.matrix, n_jobs=1)
            synth_trade_results, _, _ = synth_engine.run(seed_params=seed_params)
            
            s_wins = synth_trade_results[synth_trade_results > 0]
            s_losses = synth_trade_results[synth_trade_results < 0]
            s_gross_win = s_wins.sum() if len(s_wins) > 0 else 0.0
            s_gross_loss = abs(s_losses.sum()) if len(s_losses) > 0 else 0.0
            synth_metric = float(s_gross_win / s_gross_loss) if s_gross_loss > 0 else 0.0
            
            return synth_metric

        tasks = (delayed(worker_wfo)(i) for i in range(self.n_permutations))
        
        synthetic_metrics = []
        for res in tqdm(Parallel(n_jobs=self.n_jobs, backend="loky", return_as="generator")(tasks), total=self.n_permutations, desc="WFO Permutations"):
            synthetic_metrics.append(res)
            
        synthetic_metrics_arr = np.array(synthetic_metrics)
        
        raw_p_value = np.sum(synthetic_metrics_arr >= real_benchmark) / self.n_permutations
        
        reject, pvals_corrected, _, _ = multipletests([raw_p_value], alpha=0.05, method='fdr_bh')
        final_p_value = pvals_corrected[0]
        
        logger.info(f"Final WFO BH-Corrected P-Value: {final_p_value}")
        
        return float(real_benchmark), synthetic_metrics, float(final_p_value)
