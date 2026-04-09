import os
import json
import pandas as pd
from pathlib import Path

def generate_permutation_leaderboard():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    
    if not os.path.exists(reports_dir):
        print("No reports directory found.")
        return

    results = []
    
    for path in Path(reports_dir).rglob('*.json'):
        filename = path.name
        
        mode = "bar" # Default fallback
        stage_name = "Unknown"
        
        if filename.startswith('stage1_') and filename.endswith('_permutation_stats.json'):
            mode = filename.replace('stage1_', '').replace('_permutation_stats.json', '')
            stage_name = 'Stage 1 (Permute)'
        elif filename.startswith('stage1.5_') and filename.endswith('_transfer_stats.json'):
            mode = filename.replace('stage1.5_', '').replace('_transfer_stats.json', '')
            stage_name = 'Stage 1.5 (Transfer Validation)'
        elif filename.startswith('stage2_') and filename.endswith('_wfo_permutation_stats.json'):
            mode = filename.replace('stage2_', '').replace('_wfo_permutation_stats.json', '')
            stage_name = 'Stage 2 (WFO Permute)'
        else:
            continue
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            timeframe = path.parts[-2]
            strategy_name = path.parts[-3]
            
            if strategy_name.endswith("_OLD"):
                continue # Skip quarantined V1 folders
                
            p_value = data.get("p_value_bh_corrected", 1.0)
            real_benchmark = data.get("real_benchmark", 0.0)
            permutations_run = data.get("permutations_run", 0)
            
            # Formatting verdict based on p-value
            if p_value < 0.01:
                verdict = "✅ Excellent (<1%)"
            elif p_value < 0.05:
                verdict = "✅ Passed (<5%)"
            elif p_value < 0.10:
                verdict = "⚠️ Borderline (<10%)"
            else:
                verdict = "❌ Failed (Noise)"

            results.append({
                "Stage": stage_name,
                "Timeframe": timeframe,
                "Mode": mode,
                "Strategy": strategy_name,
                "Real Benchmark": f"{real_benchmark:.4f}",
                "P-Value": f"{p_value:.4f}",
                "Verdict": verdict,
                "Permutations": permutations_run
            })
        except Exception as e:
            print(f"Failed to parse {path}: {e}")
            
    if not results:
        print("No valid permutation results found to generate leaderboard.")
        return
        
    df = pd.DataFrame(results)
    
    markdown_output = "# 🛡️ Quant Engine: Statistical Robustness Leaderboard\n\n"
    markdown_output += "This leaderboard tracks the Monte Carlo permutation tests across all Walk-Forward stages. A lower P-Value indicates a higher probability of genuine structural edge.\n\n"
    
    # Group by Stage and Timeframe
    for stage in sorted(df['Stage'].unique()):
        markdown_output += f"## {stage}\n"
        stage_df = df[df['Stage'] == stage]
        
        for tf in sorted(stage_df['Timeframe'].unique(), key=lambda x: (len(x), x)):
            markdown_output += f"### Timeframe: {tf}\n"
            tf_df = stage_df[stage_df['Timeframe'] == tf].copy()
            # Sort by Mode, then P-Value ascending (lower is better)
            tf_df.sort_values(by=["Mode", "P-Value"], ascending=[True, True], inplace=True)
            tf_df.drop(columns=['Stage', 'Timeframe'], inplace=True)
            tf_df.reset_index(drop=True, inplace=True)
            
            markdown_output += tf_df.to_markdown(index=False) + "\n\n"
    
    leaderboard_file = os.path.join(base_dir, "PERMUTATION_LEADERBOARD.md")
    with open(leaderboard_file, "w", encoding="utf-8") as f:
        f.write(markdown_output)
        
    print(f"Successfully generated PERMUTATION_LEADERBOARD.md ranking {len(df)} tests!")

if __name__ == "__main__":
    generate_permutation_leaderboard()