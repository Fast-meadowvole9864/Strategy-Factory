import os
import matplotlib
matplotlib.use('Agg') # For headless execution without a live X-server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
import logging
from typing import List
from optuna.visualization.matplotlib import plot_parallel_coordinate

logger = logging.getLogger(__name__)

class TearsheetGenerator:
    """
    Module 5.5: Visualization & Reporting.
    Generates visual tear-sheets for backtest results.
    """
    def __init__(self, cartridge_name: str, timeframe: str = "15m", base_dir: str = None):
        self.cartridge_name = cartridge_name
        
        if base_dir is None:
            # Assuming this script is in engine/, root is one level up
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.report_dir = os.path.join(base_dir, 'reports', self.cartridge_name, timeframe)
        os.makedirs(self.report_dir, exist_ok=True)
        
    def generate_optimization_chart(self, study: optuna.Study, metric_name: str = "Profit Factor", file_suffix: str = "", stage_prefix: str = "") -> str:
        """
        Chart A: Optimization Surface.
        Plots a scatter plot of the Bayesian optimization trials.
        X/Y axes are tested parameters, color gradient is the target metric.
        """
        df = study.trials_dataframe(attrs=('params', 'value', 'state'))
        # Filter only completed trials
        if df.empty:
            logger.warning("No trials to plot for optimization chart.")
            return ""
            
        if 'state' in df.columns:
            df = df[df['state'] == 'COMPLETE']
            
        if df.empty:
            logger.warning("No completed trials to plot.")
            return ""
            
        # Extract params
        param_cols = [c for c in df.columns if c.startswith('params_')]
        
        plt.figure(figsize=(10, 8))
        
        if len(param_cols) == 0:
            logger.warning("No parameters to plot for optimization chart.")
            plt.close()
            return ""
        elif len(param_cols) == 1:
            # 1D Scatter plot
            x_col = param_cols[0]
            param_name = x_col.replace('params_', '')
            scatter = plt.scatter(df[x_col], df['value'], c=df['value'], cmap='viridis', s=100, alpha=0.8, edgecolor='k')
            plt.xlabel(param_name)
            plt.ylabel(metric_name)
            plt.title(f"Optimization Surface: {param_name} vs {metric_name}")
            plt.colorbar(scatter, label=metric_name)
        elif len(param_cols) == 2:
            # 2D Scatter plot
            x_col = param_cols[0]
            y_col = param_cols[1]
            x_name = x_col.replace('params_', '')
            y_name = y_col.replace('params_', '')
            
            scatter = plt.scatter(df[x_col], df[y_col], c=df['value'], cmap='viridis', s=100, alpha=0.8, edgecolor='k')
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title(f"Optimization Surface: {x_name} vs {y_name}")
            plt.colorbar(scatter, label=metric_name)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
        else:
            # Multi-Dimensional Parallel Coordinates Plot
            plt.close() # Close the earlier created figure since plot_parallel_coordinate creates its own
            ax = plot_parallel_coordinate(study, target_name=metric_name)
            # plot_parallel_coordinate returns an AxesSubplot object. 
            # we need to get the figure to adjust it, and set the title on the axis directly to overwrite default
            ax.figure.set_size_inches(14, 8)
            ax.set_title(f"Parallel Coordinates Optimization Surface (Target: {metric_name})")
            plt.tight_layout()
            
        filename = f'{stage_prefix}optimization_surface_{file_suffix}.png' if file_suffix else f'{stage_prefix}optimization_surface.png'
        save_path = os.path.join(self.report_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close('all')
        logger.info(f"Saved Optimization Chart to {save_path}")
        return save_path

    def generate_permutation_chart(self, synthetic_metrics: List[float], real_benchmark: float, p_value: float, metric_name: str = "Profit Factor", file_suffix: str = "", stage_prefix: str = "") -> str:
        """
        Chart B: Permutation Distribution.
        Plots a histogram of the synthetic metrics.
        Draws an axvline for the Real Data Benchmark.
        Annotates with Benjamini-Hochberg P-Value.
        """
        plt.figure(figsize=(10, 6))
        
        # Seaborn histogram
        sns.histplot(synthetic_metrics, bins=50, kde=True, color='skyblue', edgecolor='black')
        
        # Real benchmark line
        plt.axvline(x=real_benchmark, color='red', linestyle='--', linewidth=2, label=f'Real Benchmark: {real_benchmark:.6f}')
        
        # Annotation for P-Value
        plt.annotate(
            f'BH P-Value: {p_value:.4f}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            fontsize=12,
            fontweight='bold',
            verticalalignment='top'
        )
        
        plt.xlabel(f"Synthetic {metric_name}s")
        plt.ylabel("Frequency")
        plt.title(f"Monte Carlo Permutation Distribution ({len(synthetic_metrics)} Paths)")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        filename = f'{stage_prefix}permutation_distribution_{file_suffix}.png' if file_suffix else f'{stage_prefix}permutation_distribution.png'
        save_path = os.path.join(self.report_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Saved Permutation Chart to {save_path}")
        return save_path
