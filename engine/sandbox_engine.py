import polars as pl
import logging
from typing import Type, Dict, Any
from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class SandboxEngine:
    """
    SandboxEngine: The Vectorized Sandbox.
    
    API Contract C:
    Takes a Polars DataFrame, bridges to Pandas for Cartridge signal generation,
    bridges back to Polars, and calculates lookahead-bias-free metrics.
    """
    def __init__(self, data: pl.DataFrame, cartridge_class: Type[BaseStrategy], params: Dict[str, Any] = None, mode: str = "bar", matrix: list = None):
        self.mode = mode
        self.matrix = matrix if matrix is not None else [4, 8, 12]
        self.data = data
        self.cartridge_class = cartridge_class
        self.params = params or {}

    def run(self) -> Dict[str, float]:
        """
        Executes the sandbox backtest loop.
        """
        # Contract B: Polars -> Pandas via PyArrow
        # use_pyarrow_extension_array is True by default for Polars to Pandas where applicable,
        # but we use the simple to_pandas() which uses pyarrow under the hood if available.
        pandas_df = self.data.to_pandas()

        # Instantiate Cartridge and run to get signals
        cartridge = self.cartridge_class(pandas_df, self.params)
        signals_pandas = cartridge.run()

        # Bridge back to Polars
        # Reset index is not strictly required if we are just creating Polars series,
        # but to ensure horizontal concatenation aligns exactly without index issues:
        signals_polars = pl.from_pandas(signals_pandas)

        # Concatenate horizontally
        df = pl.concat([self.data, signals_polars], how="horizontal")

        if cartridge.type == "Directional":
            return self._calculate_directional(df)
        elif cartridge.type == "Magnitude":
            return self._calculate_magnitude(df)
        else:
            raise ValueError(f"Unknown cartridge type: {cartridge.type}")

    def _calculate_directional(self, df: pl.DataFrame) -> Dict[str, float]:
        df = df.with_columns([
            pl.col("Signal_Long").shift(1).over("symbol").fill_null(0).alias("Pos_Long"),
            pl.col("Signal_Short").shift(1).over("symbol").fill_null(0).alias("Pos_Short"),
        ])
        
        df = df.with_columns([
            ((pl.col("Pos_Long") == 1) & (pl.col("Pos_Long").shift(1).over("symbol").fill_null(0) == 0)).alias("Entry_Long"),
            ((pl.col("Pos_Short") == -1) & (pl.col("Pos_Short").shift(1).over("symbol").fill_null(0) == 0)).alias("Entry_Short")
        ])

        if self.mode in ["forwardr", "eratio"]:
            horizon_cols = []
            for H in self.matrix:
                horizon_cols.extend([
                    (pl.col("close").shift(-H).over("symbol") / pl.col("close") - 1.0).alias(f"Fwd_Ret_L_{H}"),
                    (1.0 - pl.col("close").shift(-H).over("symbol") / pl.col("close")).alias(f"Fwd_Ret_S_{H}"),
                    ((pl.col("high").rolling_max(window_size=H).shift(-H).over("symbol") / pl.col("close")) - 1.0).clip(lower_bound=0.0).alias(f"MFE_L_{H}"),
                    (1.0 - (pl.col("low").rolling_min(window_size=H).shift(-H).over("symbol") / pl.col("close"))).clip(lower_bound=0.0).alias(f"MAE_L_{H}"),
                    (1.0 - (pl.col("low").rolling_min(window_size=H).shift(-H).over("symbol") / pl.col("close"))).clip(lower_bound=0.0).alias(f"MFE_S_{H}"),
                    ((pl.col("high").rolling_max(window_size=H).shift(-H).over("symbol") / pl.col("close")) - 1.0).clip(lower_bound=0.0).alias(f"MAE_S_{H}")
                ])
            df = df.with_columns(horizon_cols)
        
        # 2. Base Bar-by-Bar Return (Neurotrader Method)
        df = df.with_columns([
            (pl.col("Pos_Long") * pl.col("Log_Return")).alias("Raw_Long"),
            (pl.col("Pos_Short") * pl.col("Log_Return")).alias("Raw_Short")
        ])
        
        # 3. Fee Allocation (TEMPORARILY DISABLED)
        df = df.with_columns([
            (pl.col("Pos_Long").diff().over("symbol").abs() * 0.0).fill_null(0.0).alias("Fee_Long"),
            (pl.col("Pos_Short").diff().over("symbol").abs() * 0.0).fill_null(0.0).alias("Fee_Short"),
        ])
        
        # 4. Net Vector
        df = df.with_columns([
            (pl.col("Raw_Long") - pl.col("Fee_Long")).alias("Return_Long"),
            (pl.col("Raw_Short") - pl.col("Fee_Short")).alias("Return_Short"),
        ])

        # Helper to calculate Profit Factor
        def calc_pf(series: pl.Series) -> float:
            series = series.drop_nulls()
            pos_sum = series.filter(series > 0).sum()
            neg_sum = series.filter(series < 0).sum()

            if pos_sum is None: pos_sum = 0.0
            if neg_sum is None: neg_sum = 0.0

            abs_neg = abs(neg_sum)
            if abs_neg == 0.0:
                if pos_sum > 0.0:
                    return 999.0  # Perfect win rate (Synthetic Infinity)
                return 0.0  # Empty or no valid trades
            return float(pos_sum / abs_neg)

        pf_long = calc_pf(df["Return_Long"])
        pf_short = calc_pf(df["Return_Short"])

        total_returns = df["Return_Long"].fill_null(0.0) + df["Return_Short"].fill_null(0.0)
        pf_total = calc_pf(total_returns)

        # Vectorized metrics
        pos_total = df["Pos_Long"].fill_null(0.0) + df["Pos_Short"].fill_null(0.0)
        active_bars = pos_total.abs().sum()
        total_bars = len(df)
        active_bars_pct = float(active_bars / total_bars) if total_bars > 0 else 0.0

        total_trades_val = df.select(
            pl.col("Entry_Long").cast(pl.Int32).sum() + pl.col("Entry_Short").cast(pl.Int32).sum()
        ).item()
        total_trades = int(total_trades_val) if total_trades_val is not None else 0
        
        # Trade ID isolating (Make sure Trade IDs don't bleed across symbols)
        df = df.with_columns([
            pos_total.alias("Pos_Total"),
            total_returns.alias("Return_Total")
        ]).with_columns([
            (pl.col("Pos_Total").diff().over("symbol").fill_null(0) != 0).cast(pl.Int32).cum_sum().over("symbol").alias("Trade_Id")
        ])
        
        # We also need to group by symbol + Trade_Id to ensure absolute uniqueness
        active_df = df.filter(pl.col("Pos_Total") != 0.0)
        if len(active_df) > 0:
            trade_pnls = active_df.group_by(["symbol", "Trade_Id"]).agg(pl.col("Return_Total").sum().alias("PnL"))["PnL"]
            wins = trade_pnls.filter(trade_pnls > 0)
            losses = trade_pnls.filter(trade_pnls < 0)
            
            gross_win = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
            pf_total_trade = float(gross_win / gross_loss) if gross_loss > 0 else 0.0
            
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            total_actual_trades = len(trade_pnls)
            win_rate = float(len(wins) / total_actual_trades) if total_actual_trades > 0 else 0.0
        else:
            avg_win = avg_loss = win_rate = pf_total_trade = 0.0

        metrics = {
            "Profit_Factor_Long": pf_long,
            "Profit_Factor_Short": pf_short,
            "Profit_Factor_Total": pf_total,     # This is the Bar-by-Bar PF used natively in Stage 1!
            "Trade_PF_Total": pf_total_trade,    # This is the Trade-by-Trade PF cleanly isolated!
            "Total_Trades": total_trades,
            "Active_Bars": int(active_bars),
            "Active_Bars_Pct": active_bars_pct,
            "Avg_Win": avg_win,
            "Avg_Loss": avg_loss,
            "Win_Rate": win_rate
        }

        if self.mode == "forwardr":
            fwd_rets_long = []
            fwd_rets_short = []
            ret_by_horizon = {}
            for H in self.matrix:
                l_ret = df.filter(pl.col("Entry_Long"))[f"Fwd_Ret_L_{H}"].drop_nulls()
                s_ret = df.filter(pl.col("Entry_Short"))[f"Fwd_Ret_S_{H}"].drop_nulls()
                combined_ret = pl.concat([l_ret, s_ret])
                ret_by_horizon[f"H{H}"] = float(combined_ret.mean()) if len(combined_ret) > 0 else 0.0
                fwd_rets_long.append(l_ret)
                fwd_rets_short.append(s_ret)
            
            all_rets = pl.concat(fwd_rets_long + fwd_rets_short) if fwd_rets_long or fwd_rets_short else pl.Series(dtype=pl.Float64)
            pos_sum = all_rets.filter(all_rets > 0).sum() if len(all_rets) > 0 else 0.0
            neg_sum = all_rets.filter(all_rets < 0).sum() if len(all_rets) > 0 else 0.0
            abs_neg = abs(neg_sum)
            pf_forward = float(pos_sum / abs_neg) if abs_neg > 0 else (999.0 if pos_sum > 0 else 0.0)
            
            metrics["Forward_Return_Mean"] = float(all_rets.mean()) if len(all_rets) > 0 else 0.0
            metrics["Forward_Return_By_Horizon"] = ret_by_horizon
            metrics["Profit_Factor_Forward"] = pf_forward

        elif self.mode == "eratio":
            eratio_by_horizon = {}
            all_eratios = []
            for H in self.matrix:
                l_mfe = df.filter(pl.col("Entry_Long"))[f"MFE_L_{H}"].drop_nulls().sum()
                l_mae = df.filter(pl.col("Entry_Long"))[f"MAE_L_{H}"].drop_nulls().sum()
                s_mfe = df.filter(pl.col("Entry_Short"))[f"MFE_S_{H}"].drop_nulls().sum()
                s_mae = df.filter(pl.col("Entry_Short"))[f"MAE_S_{H}"].drop_nulls().sum()
                
                tot_mfe = l_mfe + s_mfe
                tot_mae = l_mae + s_mae
                eratio = float(tot_mfe / tot_mae) if tot_mae > 0 else (999.0 if tot_mfe > 0 else 0.0)
                eratio_by_horizon[f"H{H}"] = eratio
                all_eratios.append(eratio)
            
            metrics["E_Ratio_Mean"] = sum(all_eratios) / len(all_eratios) if all_eratios else 0.0
            metrics["E_Ratio_By_Horizon"] = eratio_by_horizon

        return metrics

    def _calculate_magnitude(self, df: pl.DataFrame) -> Dict[str, float]:
        # CRITICAL: .shift(1) to prevent Lookahead Bias
        df = df.with_columns([
            pl.col("Signal_Magnitude").shift(1).over("symbol").fill_null(0).alias("Shifted_Signal"),
            (pl.col("Signal_Magnitude").diff().over("symbol") == 1).alias("Mag_Trigger")
        ])

        # Calculate Active_Mean (average abs(Log_Return) when Shifted_Signal == 1)
        # Calculate Inactive_Mean (when 0)

        active_returns = df.filter(pl.col("Shifted_Signal") == 1)["Log_Return"].abs()
        inactive_returns = df.filter(pl.col("Shifted_Signal") == 0)["Log_Return"].abs()

        active_mean = active_returns.mean()
        inactive_mean = inactive_returns.mean()

        if active_mean is None: active_mean = 0.0
        if inactive_mean is None: inactive_mean = 0.0

        if inactive_mean == 0.0:
            ratio = 0.0
        else:
            ratio = active_mean / inactive_mean

        total_trades_val = df["Mag_Trigger"].sum()
        total_trades = int(total_trades_val) if total_trades_val is not None else 0
        
        active_bars = df["Shifted_Signal"].sum()
        total_bars = len(df)
        active_bars_pct = float(active_bars / total_bars) if total_bars > 0 else 0.0

        return {
            "Active_Mean": float(active_mean),
            "Inactive_Mean": float(inactive_mean),
            "Magnitude_Ratio": float(ratio),
            "Total_Trades": total_trades,
            "Active_Bars": int(active_bars),
            "Active_Bars_Pct": active_bars_pct,
            "Avg_Win": 0.0,
            "Avg_Loss": 0.0,
            "Win_Rate": 0.0
        }
