import optuna
import polars as pl
import pandas as pd
import logging
import math
from typing import Type, Dict, Any, Tuple
from tqdm import tqdm
from engine.sandbox_engine import SandboxEngine
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)
TRADE_FLOOR_PER_ASSET_YEAR = 30.0

class OptunaOptimizer:
    """
    Module 4.5: The Bayesian Optimizer
    Wraps the Sandbox Engine to find optimal parameters using Optuna.
    """
    def __init__(self, data: pl.DataFrame, cartridge_class: Type[BaseStrategy], n_trials: int = 50, target_direction: str = "Total", seed_params: Dict[str, Any] = None, mode: str = "bar", matrix: list = None, n_jobs: int = 1):
        self.mode = mode
        self.matrix = matrix if matrix is not None else [4, 8, 12]
        self.n_jobs = n_jobs
        self.data = data
        self.cartridge_class = cartridge_class
        self.n_trials = n_trials
        self.target_direction = target_direction
        self.seed_params = seed_params
        
        # Instantiate dummy to get properties
        self.dummy_cartridge = self.cartridge_class(pd.DataFrame())
        
        # We handle cases where param_space might not be implemented yet on all cartridges
        self.param_space = getattr(self.dummy_cartridge, "param_space", {})
        self.cartridge_type = self.dummy_cartridge.type
        
        # --- DYNAMIC TRADE FLOOR SCALING ---
        # Portfolio-wide floor: 30 trades per combined asset-year.
        # Example: BTC+ETH over one year requires at least 60 total trades.
        if "timestamp" in self.data.columns:
            # Polars diff to find the timeframe safely
            timeframe_ms = self.data["timestamp"].diff().drop_nulls().median()
            if timeframe_ms is not None and timeframe_ms > 0:
                total_ms = len(self.data) * timeframe_ms
                combined_years = total_ms / (1000.0 * 60.0 * 60.0 * 24.0 * 365.25)
                self.min_trades = max(1, int(combined_years * TRADE_FLOOR_PER_ASSET_YEAR))
            else:
                self.min_trades = int(TRADE_FLOOR_PER_ASSET_YEAR)
        else:
            self.min_trades = int(TRADE_FLOOR_PER_ASSET_YEAR)
            
        logger.debug(f"Dynamic Minimum Trade Floor calculated as: {self.min_trades}")

    def _objective(self, trial: optuna.Trial) -> float:
        # Dynamically build parameters based on param_space
        params = {}
        if self.param_space:
            for param_name, constraints in self.param_space.items():
                if constraints.get("type") == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        constraints["min"], 
                        constraints["max"]
                    )
                elif constraints.get("type") == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        constraints["min"], 
                        constraints["max"]
                    )
        
        # Run Sandbox
        engine = SandboxEngine(self.data, self.cartridge_class, params, mode=self.mode, matrix=self.matrix)
        results = engine.run()
        
        # Store full results dict in the trial for later extraction
        trial.set_user_attr("results", results)
        
        # Target Metric Base
        if self.cartridge_type == "Directional":
            if self.mode == "forwardr":
                base_metric = results.get("Profit_Factor_Forward", 0.0)
            elif self.mode == "eratio":
                base_metric = results.get("E_Ratio_Mean", 0.0)
            else:
                if self.target_direction == "Long":
                    base_metric = results.get("Profit_Factor_Long", 0.0)
                elif self.target_direction == "Short":
                    base_metric = results.get("Profit_Factor_Short", 0.0)
                else:
                    base_metric = results.get("Profit_Factor_Total", 0.0)
                
            # The Dynamic Trade Floor (No scaler, pure geometric edge)
            trades = results.get("Total_Trades", 0)
            if trades < self.min_trades:
                target = 0.0  # Statistical significance quarantine
            else:
                target = base_metric
                
        elif self.cartridge_type == "Magnitude":
            # Magnitude requires no floor and no frequency target. Pure Ratio.
            target = results.get("Magnitude_Ratio", 0.0)
            
        else:
            target = 0.0

        return target

    def run(self, show_progress_bar: bool = True, leave_bar: bool = False) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        """
        Runs the optimization and returns (best_params, best_metric, best_results).
        """
        # Suppress Optuna's verbose printing by default unless specifically needed
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        
        if self.seed_params:
            # Filter seed_params to only include variables in current param_space
            filtered_seed = {k: v for k, v in self.seed_params.items() if k in self.param_space}
            if filtered_seed:
                logger.info(f"Enqueuing Stage 1 optimal params as seed trial: {filtered_seed}")
                study.enqueue_trial(filtered_seed)
                
        if show_progress_bar:
            with tqdm(total=self.n_trials, desc="Indicator Trials", leave=leave_bar) as pbar:
                def callback(study, trial):
                    pbar.update(1)
                study.optimize(self._objective, n_trials=self.n_trials, callbacks=[callback], n_jobs=1)
        else:
            study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        
        self.study = study
        best_results = study.best_trial.user_attrs.get("results", {})
        
        return study.best_params, study.best_value, best_results
