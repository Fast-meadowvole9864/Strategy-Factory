import os
import sys
import glob
import subprocess
import argparse
import itertools

def get_available_cartridges():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    strategies_dir = os.path.join(base_dir, "strategies")
    cartridges = []
    for f in os.listdir(strategies_dir):
        if f.endswith(".py") and not f.startswith("__") and f != "base_strategy.py":
            cartridges.append(f[:-3])
    return sorted(cartridges)

def main():
    parser = argparse.ArgumentParser(description="Mass Optuna Runner")
    parser.add_argument("--base", nargs='+', required=True, help="Base cartridge(s) to always include (e.g., ema_slope)")
    parser.add_argument("--combine", type=int, default=1, choices=[1, 2], help="Number of additional cartridges to combine with the base (default: 1)")
    parser.add_argument("--pool", nargs='+', help="Specific cartridges to draw from for combinations. If omitted, uses all available in strategies folder.")
    parser.add_argument("--trials", type=int, default=300, help="Number of Optuna trials per run (default: 300)")
    
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(base_dir, "main.py")

    pool = args.pool if args.pool else get_available_cartridges()
    
    # Remove base cartridges from the pool so we don't test a cartridge against itself
    pool = [c for c in pool if c not in args.base]
    
    modes = ["forwardr", "eratio"]
    timeframes = ["1h", "15m"]
    
    # Generate combinations from the pool
    combos = list(itertools.combinations(pool, args.combine))
    
    total_runs = len(modes) * len(timeframes) * len(combos)
    
    print(f"=========================================")
    print(f"MASS OPTUNA ORCHESTRATOR")
    print(f"=========================================")
    print(f"Base Cartridge(s): {args.base}")
    print(f"Combination Size : {args.combine}")
    print(f"Pool Size        : {len(pool)} cartridges")
    print(f"Generated Combos : {len(combos)} unique pairs")
    print(f"Modes            : {modes}")
    print(f"Timeframes       : {timeframes}")
    print(f"Trials per Run   : {args.trials}")
    print(f"-----------------------------------------")
    print(f"Total Runs Scheduled: {total_runs}")
    print(f"=========================================\n")
    
    if total_runs > 0:
        proceed = input("Do you want to proceed with this batch execution? (y/n): ")
        if proceed.lower() != 'y':
            print("Execution aborted.")
            return

    run_count = 0
    for mode in modes:
        for tf in timeframes:
            for combo in combos:
                run_count += 1
                cartridges = args.base + list(combo)
                
                cmd = [
                    sys.executable, main_path,
                    "--cartridge"
                ] + cartridges + [
                    "--stage1-optuna",
                    "--trials", str(args.trials),
                    "--mode", mode,
                    "--matrix", "4", "8", "12",
                    "--timeframe", tf,
                    "--jobs", "-1"
                ]
                
                cmd_str = " ".join(cmd)
                print(f"\n>>> [{run_count}/{total_runs}] EXECUTING: {cmd_str}")
                
                try:
                    subprocess.run(cmd, check=True, cwd=base_dir)
                except subprocess.CalledProcessError as e:
                    print(f"!!! Error executing run {run_count}. Moving to next combination. Error: {e}")
                except KeyboardInterrupt:
                    print("\n!!! Mass execution physically cancelled by operator.")
                    sys.exit(1)
                    
    print("\n=========================================")
    print(f"MASS OPTUNA BATCH COMPLETE ({total_runs} Runs)")
    print(f"Run 'python scripts/generate_leaderboard.py' to see the results!")
    print("=========================================")

if __name__ == "__main__":
    main()
