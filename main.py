import os
import sys

# CRITICAL FIX: Intercept the jobs argument BEFORE importing Polars or Engine tools.
# If we are running parallel jobs, we MUST lock Polars to 1 thread globally to prevent thread explosion.
if "--jobs" in sys.argv:
    jobs_idx = sys.argv.index("--jobs")
    if jobs_idx + 1 < len(sys.argv) and sys.argv[jobs_idx + 1] != "1":
        os.environ["POLARS_MAX_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
elif "-1" in sys.argv: 
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import importlib
import json
import polars as pl
import logging
import re
from engine.data_engine import load_vault_data
from engine.optimizer import OptunaOptimizer
from engine.permutation_engine import PermutationEngine
from engine.walk_forward import WalkForwardEngine, simulate_execution_1m
from engine.tearsheet import TearsheetGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

VAULT_1 = 'Vault_1_InSample'
VAULT_2 = 'Vault_2_OOS'
VAULT_3 = 'Vault_3_Holdout'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports")

COINS_FILTER = None
DEFAULT_WFO_WINDOWS = {
    "1h": (4320, 720),
    "15m": (17280, 2880),
    "1m": (259200, 43200),
}
EXECUTION_PARAM_KEYS = {"sl_pct", "tp_pct", "_test_start_ts", "roll_pnl", "roll_trades"}

def load_and_concat(vault: str, tf: str) -> pl.DataFrame:
    data_dict = load_vault_data(vault, tf)
    if not data_dict:
        raise ValueError(f"No data found in {vault} for {tf}")
        
    if COINS_FILTER:
        filtered_dict = {k: v for k, v in data_dict.items() if k in COINS_FILTER}
        if not filtered_dict:
            raise ValueError(f"None of the specified coins {COINS_FILTER} were found in {vault}. Available: {list(data_dict.keys())}")
        data_dict = filtered_dict
        
    # Sort data per asset by timestamp, then concat
    dfs = []
    for asset, df in data_dict.items():
        dfs.append(df.sort("timestamp"))
    return pl.concat(dfs)

def calculate_per_asset_start_index(df: pl.DataFrame) -> int:
    if len(df) == 0:
        raise ValueError("Cannot calculate WFO permutation start_index from an empty dataframe.")
    if "symbol" not in df.columns:
        return len(df)

    first_symbol = df["symbol"][0]
    return len(df.filter(pl.col("symbol") == first_symbol))

def strip_execution_params(params: dict) -> dict:
    return {key: value for key, value in params.items() if key not in EXECUTION_PARAM_KEYS}

def load_params_from_json(path: str, prefer_last_roll: bool = True) -> dict:
    with open(path, "r") as f:
        data = json.load(f)

    params = data
    if isinstance(data, dict) and "optimal_params" in data:
        params = data["optimal_params"]
    elif isinstance(data, dict) and prefer_last_roll and "rolling_parameters" in data:
        rolling_params = data["rolling_parameters"]
        if not isinstance(rolling_params, list):
            raise ValueError(f"Invalid rolling_parameters in {path}: expected a list.")
        valid_rolls = [roll for roll in rolling_params if isinstance(roll, dict)]
        params = valid_rolls[-1] if valid_rolls else {}

    if not isinstance(params, dict):
        raise ValueError(f"Could not extract a parameter dictionary from {path}.")

    return strip_execution_params(dict(params))

def load_optional_seed_params(path: str, missing_message: str, required: bool = False) -> dict | None:
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(missing_message)
        logger.warning(missing_message)
        return None

    params = load_params_from_json(path)
    if not params:
        if required:
            raise ValueError(f"No parameters found in {path}.")
        logger.warning(f"No parameters found in {path}. Starting without a seed.")
        return None

    logger.info(f"Loaded seed parameters from {path}")
    return params

def resolve_wfo_windows(timeframe: str, train_window: int | None = None, test_window: int | None = None) -> tuple[int, int]:
    if train_window is not None:
        resolved_train = train_window
        resolved_test = test_window if test_window is not None else int(resolved_train / 6)
    else:
        resolved_train, resolved_test = DEFAULT_WFO_WINDOWS.get(timeframe, DEFAULT_WFO_WINDOWS["15m"])
        if test_window is not None:
            resolved_test = test_window

    if resolved_train <= 0 or resolved_test <= 0:
        raise ValueError("WFO train and test windows must be positive integers.")

    return int(resolved_train), int(resolved_test)

def make_wfo_results_filename(stage: str, mode: str, timeframe: str, train_window: int, test_window: int) -> str:
    if stage == "stage2":
        return f"stage2_{mode}_wfo_results_tf-{timeframe}_train-{train_window}b_test-{test_window}b.json"
    if stage == "stage3":
        return f"stage3_{mode}_holdout_results_tf-{timeframe}_train-{train_window}b_test-{test_window}b.json"
    raise ValueError(f"Unsupported WFO results stage: {stage}")

def resolve_stage2_results_file(report_dir: str, mode: str, timeframe: str, train_window: int, test_window: int) -> str:
    canonical_file = os.path.join(report_dir, make_wfo_results_filename("stage2", mode, timeframe, train_window, test_window))
    legacy_file = os.path.join(report_dir, f"stage2_{mode}_wfo_results.json")
    if os.path.exists(canonical_file) or not os.path.exists(legacy_file):
        return canonical_file
    return legacy_file

def main():
    parser = argparse.ArgumentParser(description="Quant Engine CLI Master Orchestrator")
    parser.add_argument("--cartridge", required=True, nargs='+', help="Name of the cartridge(s) to run (e.g., adx_regime ema_slope)")
    parser.add_argument("--trials", type=int, default=250, help="Number of Optuna trials for Stage 1 indicator parameters (default: 250)")
    parser.add_argument("--sltp-trials", type=int, default=20, help="Number of Optuna trials for Stage 2/3 SL/TP friction optimization (default: 20)")
    parser.add_argument("--permutations", type=int, default=1000, help="Number of synthetic Monte Carlo permutations (default: 1000)")
    parser.add_argument("--target", type=str, choices=["Long", "Short", "Total"], default="Total", help="Direction parameter target for optimization (Directional only)")
    parser.add_argument("--coins", type=str, nargs='+', help="Optional list of specific coins to load (e.g., btc eth)")
    parser.add_argument("--timeframe", type=str, choices=["1h", "15m", "1m"], default="15m", help="Timeframe macro resolution to process (default: 15m)")
    parser.add_argument("--mode", type=str, choices=["bar", "forwardr", "eratio"], default="bar", help="Metric target mode (default: bar)")
    parser.add_argument("--matrix", type=int, nargs="+", default=[4, 8, 12], help="Horizon matrix for forwardr and eratio modes (default: 4 8 12)")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs for Optuna/Joblib (default: -1)")
    parser.add_argument("--train-window", type=int, help="Optional: Override the WFO training window (in bars)")
    parser.add_argument("--test-window", type=int, help="Optional: Override the WFO testing window (in bars)")
    parser.add_argument("--params-file", type=str, help="Optional: Load downstream seed parameters from a specific JSON file")
    parser.add_argument("--fees", action="store_true", help="Apply 0.1% round-trip execution fees (0.05% maker/taker + slippage penalty)")
    
    # Exclusive routing flags
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage1-optuna", action="store_true", help="Run Stage 1 Optuna Optimizer")
    group.add_argument("--stage1-permute", action="store_true", help="Run Stage 1 Permutation Matrix")
    group.add_argument("--stage1.5-transfer", action="store_true", help="Run Stage 1.5 Signal Transfer Validation")
    group.add_argument("--stage2-wfo", action="store_true", help="Run Stage 2 Adaptive WFO Simulator")
    group.add_argument("--stage2-permute", action="store_true", help="Run Stage 2 WFO Permutation Test")
    group.add_argument("--stage3-holdout", action="store_true", help="Run Stage 3 Holdout Verification")

    args = parser.parse_args()

    global COINS_FILTER
    COINS_FILTER = args.coins

    # Dynamic Loading
    from strategies.base_strategy import BaseStrategy
    from engine.dynamic_fusion import create_dynamic_cartridge
    
    cartridge_classes = []
    for cart_name in args.cartridge:
        # Security Patch: Whitelist imports to prevent directory traversal
        if not re.match(r"^[a-zA-Z0-9_]+$", cart_name):
            logger.error(f"Invalid cartridge name: {cart_name}. Only alphanumeric characters and underscores are allowed.")
            return
            
        try:
            module = importlib.import_module(f"strategies.{cart_name}")
            cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseStrategy) and attr is not BaseStrategy:
                    cls = attr
                    break
            if cls is None:
                logger.error(f"Could not find a valid Cartridge class in strategies.{cart_name}")
                return
            cartridge_classes.append(cls)
        except ImportError as e:
            logger.error(f"Failed to import cartridge: strategies.{cart_name} - {e}")
            return
            
    if len(cartridge_classes) == 1:
        cartridge_class = cartridge_classes[0]
    else:
        logger.info(f"Dynamically fusing {len(cartridge_classes)} cartridges...")
        cartridge_class = create_dynamic_cartridge(cartridge_classes)

    # Instantiate dummy to get the name for the report directory
    dummy_cart = cartridge_class(pl.DataFrame().to_pandas())
    report_dir = os.path.join(REPORTS_ROOT, dummy_cart.name, args.timeframe)
    
    if dummy_cart.type == "Directional":
        params_file = os.path.join(report_dir, f'stage1_{args.mode}_optimal_params_{args.target.lower()}.json')
    else:
        params_file = os.path.join(report_dir, f'stage1_{args.mode}_optimal_params.json')

    # Routing
    if args.stage1_optuna:
        logger.info(f"Executing Stage 1: Optuna Optimization ({args.trials} trials, Target: {args.target})")
        vault_1 = load_and_concat(VAULT_1, args.timeframe)
        
        # Override jobs for Optuna to prevent deadlock
        logger.info(f"Forcing Optuna to n_jobs=1 to prevent Polars threading deadlock.")
        optimizer = OptunaOptimizer(vault_1, cartridge_class, n_trials=args.trials, target_direction=args.target, mode=args.mode, matrix=args.matrix, n_jobs=1)
        best_params, best_metric, best_results = optimizer.run(show_progress_bar=True, leave_bar=True)
        
        os.makedirs(report_dir, exist_ok=True)
        
        summary_file = os.path.join(report_dir, f'stage1_{args.mode}_optuna_results.json')

        if dummy_cart.type == "Directional":
            params_file = os.path.join(report_dir, f'stage1_{args.mode}_optimal_params_{args.target.lower()}.json')
            if args.mode == "forwardr":
                metric_name = f"Forward Return Profit Factor"
            elif args.mode == "eratio":
                metric_name = f"Mean E-Ratio"
            else:
                metric_name = f"Profit Factor ({args.target})"
            
            pf_key = f"Profit_Factor_{args.target}"
            
            metrics_dict = {
                pf_key: best_results.get(pf_key, 0.0),
                "Trade_PF_Total": best_results.get("Trade_PF_Total", 0.0),
                "Total_Trades": best_results.get("Total_Trades", 0),
                "Active_Bars": best_results.get("Active_Bars", 0),
                "Active_Bars_Pct": best_results.get("Active_Bars_Pct", 0.0),
                "Avg_Win": best_results.get("Avg_Win", 0.0),
                "Avg_Loss": best_results.get("Avg_Loss", 0.0),
                "Win_Rate": best_results.get("Win_Rate", 0.0)
            }
            
            if args.mode == "forwardr":
                metrics_dict["Forward_Return_Mean"] = best_results.get("Forward_Return_Mean", 0.0)
                metrics_dict["Forward_Return_By_Horizon"] = best_results.get("Forward_Return_By_Horizon", {})
                metrics_dict["Profit_Factor_Forward"] = best_results.get("Profit_Factor_Forward", 0.0)
            elif args.mode == "eratio":
                metrics_dict["E_Ratio_Mean"] = best_results.get("E_Ratio_Mean", 0.0)
                metrics_dict["E_Ratio_By_Horizon"] = best_results.get("E_Ratio_By_Horizon", {})

            summary_data = {
                "optimal_params": best_params,
                "metrics": metrics_dict
            }
        else:
            params_file = os.path.join(report_dir, f'stage1_{args.mode}_optimal_params.json')
            metric_name = "Active Mean Return"
            summary_data = {
                "optimal_params": best_params,
                "metrics": {
                    "Active_Mean": best_results.get("Active_Mean", 0.0),
                    "Inactive_Mean": best_results.get("Inactive_Mean", 0.0),
                    "Magnitude_Ratio": best_results.get("Magnitude_Ratio", 0.0),
                    "Total_Trades": best_results.get("Total_Trades", 0),
                    "Active_Bars": best_results.get("Active_Bars", 0),
                    "Active_Bars_Pct": best_results.get("Active_Bars_Pct", 0.0),
                    "Avg_Win": 0.0,
                    "Avg_Loss": 0.0,
                    "Win_Rate": 0.0
                }
            }
            
        with open(params_file, "w") as f:
            json.dump(best_params, f, indent=4)
            
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)
            
        reporter = TearsheetGenerator(dummy_cart.name, timeframe=args.timeframe)
        file_suffix = args.target.lower() if dummy_cart.type == "Directional" else ""
        reporter.generate_optimization_chart(optimizer.study, metric_name=metric_name, file_suffix=file_suffix, stage_prefix=f"stage1_{args.mode}_")
        
        logger.info(f"Optimization complete. Metric: {best_metric}. Params saved to {params_file}")

    elif args.stage1_permute:
        logger.info(f"Executing Stage 1: Permutation Matrix (Target: {args.target}, Permutations: {args.permutations})")
        vault_1 = load_and_concat(VAULT_1, args.timeframe)
        engine = PermutationEngine(vault_1, cartridge_class, n_permutations=args.permutations, optuna_trials=args.trials, target_direction=args.target, mode=args.mode, matrix=args.matrix, n_jobs=args.jobs)
        real_bench, synth_metrics, p_value = engine.run(start_index=0)
        
        reporter = TearsheetGenerator(dummy_cart.name, timeframe=args.timeframe)
        metric_name = f"Profit Factor ({args.target})" if dummy_cart.type == "Directional" else "Active Mean Return"
        file_suffix = args.target.lower() if dummy_cart.type == "Directional" else ""
        reporter.generate_permutation_chart(synth_metrics, real_bench, p_value, metric_name=metric_name, file_suffix=file_suffix, stage_prefix=f"stage1_{args.mode}_")
        
        stats_file = os.path.join(report_dir, f"stage1_{args.mode}_permutation_stats.json")
        stats_data = {
            "strategy": dummy_cart.name,
            "target": args.target,
            "real_benchmark": real_bench,
            "p_value_bh_corrected": p_value,
            "permutations_run": args.permutations,
            "trials_per_permutation": args.trials
        }
        with open(stats_file, "w") as f:
            json.dump(stats_data, f, indent=4)
            
        logger.info(f"Permutation complete. Real Benchmark: {real_bench}. Final P-Value: {p_value}")
        logger.info(f"Saved stats to {stats_file}")

    elif getattr(args, "stage1.5_transfer", False):
        logger.info(f"Executing Stage 1.5: Signal Transfer Validation (Target: {args.target}, Permutations: {args.permutations})")
        params_source = args.params_file or params_file
        locked_params = load_optional_seed_params(
            params_source,
            f"Missing parameter file: {params_source}. Run --stage1-optuna first or pass --params-file.",
            required=True
        )
            
        vault_2 = load_and_concat(VAULT_2, args.timeframe)
        
        engine = PermutationEngine(vault_2, cartridge_class, n_permutations=args.permutations, optuna_trials=args.trials, target_direction=args.target, mode=args.mode, matrix=args.matrix, n_jobs=args.jobs)
        real_bench, synth_metrics, p_value = engine.run_transfer(locked_params)
        
        reporter = TearsheetGenerator(dummy_cart.name, timeframe=args.timeframe)
        metric_name = f"Transfer Profit Factor ({args.target})" if dummy_cart.type == "Directional" else "Transfer Active Mean Return"
        file_suffix = args.target.lower() if dummy_cart.type == "Directional" else ""
        reporter.generate_permutation_chart(synth_metrics, real_bench, p_value, metric_name=metric_name, file_suffix=file_suffix, stage_prefix=f"stage1.5_{args.mode}_transfer_")
        
        stats_file = os.path.join(report_dir, f"stage1.5_{args.mode}_transfer_stats.json")
        stats_data = {
            "strategy": dummy_cart.name,
            "target": args.target,
            "params_source": params_source,
            "real_benchmark": real_bench,
            "p_value_bh_corrected": p_value,
            "permutations_run": args.permutations
        }
        with open(stats_file, "w") as f:
            json.dump(stats_data, f, indent=4)
            
        logger.info(f"Signal Transfer complete. Real Benchmark: {real_bench}. Final P-Value: {p_value}")
        logger.info(f"Saved stats to {stats_file}")

    elif getattr(args, "stage2_wfo", False):
        logger.info("Executing Stage 2: Adaptive WFO Simulator")
        vault_1 = load_and_concat(VAULT_1, args.timeframe)
        vault_2 = load_and_concat(VAULT_2, args.timeframe)
        
        vault_1_1m = load_and_concat(VAULT_1, "1m")
        vault_2_1m = load_and_concat(VAULT_2, "1m")
        
        train_window, test_window = resolve_wfo_windows(args.timeframe, args.train_window, args.test_window)
            
        # Determine timeframe in milliseconds to slice the last 6 months chronologically across all assets
        tf_ms = vault_1["timestamp"].diff().drop_nulls().median()
        if tf_ms is None or tf_ms <= 0:
            tf_ms = 15 * 60 * 1000  # fallback to 15m
            
        train_ms = train_window * tf_ms
        v1_max_ts = vault_1["timestamp"].max()
        seed_start_ts = v1_max_ts - train_ms + tf_ms
        
        # Get the last 6 months of Vault 1 chronologically to seed the training window
        v1_tail_15m = vault_1.filter(pl.col("timestamp") >= seed_start_ts)
        master_15m = pl.concat([v1_tail_15m, vault_2])
        
        v1_tail_1m = vault_1_1m.filter(pl.col("timestamp") >= seed_start_ts)
        master_1m = pl.concat([v1_tail_1m, vault_2_1m])
        
        params_source = args.params_file or params_file
        locked_params = load_optional_seed_params(
            params_source,
            f"Missing Stage 1 parameter file: {params_source}. Starting WFO without a seed.",
            required=args.params_file is not None
        )
        
        # Override jobs for WFO Optuna to prevent deadlock
        logger.info(f"Forcing WFO Optuna to n_jobs=1 to prevent Polars threading deadlock.")
        engine = WalkForwardEngine(
            master_15m, 
            master_1m, 
            cartridge_class, 
            train_window=train_window, 
            test_window=test_window,
            optuna_trials=args.trials,
            sl_tp_trials=args.sltp_trials,
            mode=args.mode,
            matrix=args.matrix,
            n_jobs=1,
            fees_active=args.fees
        )
        trade_results, oos_df, rolling_parameters = engine.run(seed_params=locked_params)
        
        os.makedirs(report_dir, exist_ok=True)
        results_file = os.path.join(report_dir, make_wfo_results_filename("stage2", args.mode, args.timeframe, train_window, test_window))
        
        # Calculate overall Profit Factor from trade_results
        wins = trade_results[trade_results > 0]
        losses = trade_results[trade_results < 0]
        gross_win = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        pf_total = float(gross_win / gross_loss) if gross_loss > 0 else 0.0
        
        results_data = {
            "strategy": dummy_cart.name,
            "target": args.target,
            "timeframe": args.timeframe,
            "train_window": train_window,
            "test_window": test_window,
            "params_source": params_source if locked_params is not None else None,
            "total_trades": len(trade_results),
            "profit_factor": pf_total,
            "total_pnl": float(trade_results.sum()),
            "rolling_parameters": rolling_parameters
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
            
        logger.info(f"Stage 2 Execution complete. Total Trades: {len(trade_results)}. OOS Profit Factor: {pf_total:.4f}. Final PnL sum: {trade_results.sum():.4f}")
        logger.info(f"Saved results to {results_file}")

    elif getattr(args, "stage2_permute", False):
        logger.info("Executing Stage 2: WFO Permutation Test")
        vault_1 = load_and_concat(VAULT_1, args.timeframe)
        vault_2 = load_and_concat(VAULT_2, args.timeframe)
        
        vault_1_1m = load_and_concat(VAULT_1, "1m")
        vault_2_1m = load_and_concat(VAULT_2, "1m")
        
        train_window, test_window = resolve_wfo_windows(args.timeframe, args.train_window, args.test_window)
            
        # Determine timeframe in milliseconds to slice the last 6 months chronologically across all assets
        tf_ms = vault_1["timestamp"].diff().drop_nulls().median()
        if tf_ms is None or tf_ms <= 0:
            tf_ms = 15 * 60 * 1000  # fallback to 15m
            
        train_ms = train_window * tf_ms
        v1_max_ts = vault_1["timestamp"].max()
        seed_start_ts = v1_max_ts - train_ms + tf_ms
        
        # Get the last 6 months of Vault 1 chronologically to seed the training window
        v1_tail_15m = vault_1.filter(pl.col("timestamp") >= seed_start_ts)
        master_15m = pl.concat([v1_tail_15m, vault_2])
        
        v1_tail_1m = vault_1_1m.filter(pl.col("timestamp") >= seed_start_ts)
        master_1m = pl.concat([v1_tail_1m, vault_2_1m])
            
        params_source = args.params_file or params_file
        locked_params = load_optional_seed_params(
            params_source,
            f"Missing Stage 1 parameter file: {params_source}. Starting WFO Permute without a seed.",
            required=args.params_file is not None
        )
                
        # The permutation engine shuffles per symbol, so preserve one asset's train window length.
        start_index = calculate_per_asset_start_index(v1_tail_15m)
        
        engine = PermutationEngine(master_15m, cartridge_class, n_permutations=args.permutations, optuna_trials=args.trials, target_direction=args.target, mode=args.mode, matrix=args.matrix, n_jobs=args.jobs)
        real_bench, synth_metrics, p_value = engine.run_wfo(start_index=start_index, train_window=train_window, test_window=test_window, data_1m=master_1m, seed_params=locked_params, sl_tp_trials=args.sltp_trials)
        
        reporter = TearsheetGenerator(dummy_cart.name, timeframe=args.timeframe)
        if dummy_cart.type == "Directional":
            if args.mode == "forwardr":
                metric_name = "WFO Forward Return Profit Factor"
            elif args.mode == "eratio":
                metric_name = "WFO Mean E-Ratio"
            else:
                metric_name = f"WFO Profit Factor ({args.target})"
        else:
            metric_name = "WFO Active Mean Return"
        file_suffix = args.target.lower() if dummy_cart.type == "Directional" else ""
        reporter.generate_permutation_chart(synth_metrics, real_bench, p_value, metric_name=metric_name, file_suffix=file_suffix, stage_prefix=f"stage2_{args.mode}_wfo_")
        
        stats_file = os.path.join(report_dir, f"stage2_{args.mode}_wfo_permutation_stats.json")
        stats_data = {
            "strategy": dummy_cart.name,
            "target": args.target,
            "timeframe": args.timeframe,
            "train_window": train_window,
            "test_window": test_window,
            "params_source": params_source if locked_params is not None else None,
            "real_benchmark": real_bench,
            "p_value_bh_corrected": p_value,
            "permutations_run": args.permutations,
            "trials_per_permutation": args.trials,
            "sl_tp_trials": args.sltp_trials
        }
        with open(stats_file, "w") as f:
            json.dump(stats_data, f, indent=4)
            
        logger.info(f"WFO Permutation complete. Real Benchmark: {real_bench}. Final P-Value: {p_value}")
        logger.info(f"Saved stats to {stats_file}")

    elif args.stage3_holdout:
        logger.info("Executing Stage 3: Final Verification (Holdout)")
        vault_2_15m = load_and_concat(VAULT_2, args.timeframe)
        vault_3_15m = load_and_concat(VAULT_3, args.timeframe)
        
        vault_2_1m = load_and_concat(VAULT_2, "1m")
        vault_3_1m = load_and_concat(VAULT_3, "1m")
        
        train_window, test_window = resolve_wfo_windows(args.timeframe, args.train_window, args.test_window)
            
        # Determine timeframe in milliseconds to slice the last 6 months chronologically across all assets
        tf_ms = vault_2_15m["timestamp"].diff().drop_nulls().median()
        if tf_ms is None or tf_ms <= 0:
            tf_ms = 15 * 60 * 1000  # fallback to 15m
            
        train_ms = train_window * tf_ms
        v2_max_ts = vault_2_15m["timestamp"].max()
        seed_start_ts = v2_max_ts - train_ms + tf_ms
        
        # Get the last 6 months of Vault 2 chronologically to seed the training window
        v2_tail_15m = vault_2_15m.filter(pl.col("timestamp") >= seed_start_ts)
        master_15m = pl.concat([v2_tail_15m, vault_3_15m])
        
        v2_tail_1m = vault_2_1m.filter(pl.col("timestamp") >= seed_start_ts)
        master_1m = pl.concat([v2_tail_1m, vault_3_1m])
        
        # Load the last set of parameters from Stage 2 WFO unless an explicit source is supplied.
        params_source = args.params_file or resolve_stage2_results_file(report_dir, args.mode, args.timeframe, train_window, test_window)
        locked_params = load_optional_seed_params(
            params_source,
            f"Missing Stage 2 WFO results: {params_source}. Starting Stage 3 Holdout without a seed.",
            required=args.params_file is not None
        )
                    
        # Override jobs for WFO Optuna to prevent deadlock
        logger.info(f"Forcing WFO Optuna to n_jobs=1 to prevent Polars threading deadlock.")
        engine = WalkForwardEngine(
            master_15m, 
            master_1m, 
            cartridge_class, 
            train_window=train_window, 
            test_window=test_window,
            optuna_trials=args.trials,
            sl_tp_trials=args.sltp_trials,
            mode=args.mode,
            matrix=args.matrix,
            n_jobs=1,
            fees_active=args.fees
        )
        trade_results, oos_df, rolling_parameters = engine.run(seed_params=locked_params)
        
        os.makedirs(report_dir, exist_ok=True)
        results_file = os.path.join(report_dir, make_wfo_results_filename("stage3", args.mode, args.timeframe, train_window, test_window))
        
        wins = trade_results[trade_results > 0]
        losses = trade_results[trade_results < 0]
        gross_win = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        pf_total = float(gross_win / gross_loss) if gross_loss > 0 else 0.0
        
        results_data = {
            "strategy": dummy_cart.name,
            "target": args.target,
            "timeframe": args.timeframe,
            "train_window": train_window,
            "test_window": test_window,
            "params_source": params_source if locked_params is not None else None,
            "total_trades": len(trade_results),
            "profit_factor": pf_total,
            "total_pnl": float(trade_results.sum()),
            "rolling_parameters": rolling_parameters
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
            
        logger.info(f"Stage 3 Holdout complete. Total Trades: {len(trade_results)}. OOS Profit Factor: {pf_total:.4f}. Final PnL sum: {trade_results.sum():.4f}")
        logger.info(f"Saved results to {results_file}")

if __name__ == "__main__":
    main()
