# Quant Engine Public Core Manual

This manual is the operator-facing command reference for the public-core repo.

## 1. Environment Setup

Use Python `3.12.x`.

Verified interpreter:

- Python `3.12.10`

Install with the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 2. Data Acquisition

The downloader currently supports Binance Futures only and routes data into the fixed three-vault layout.

Default-style download with explicit assets and macro timeframe:

```powershell
.\.venv\Scripts\python.exe scripts\data_downloader.py --assets BTC ETH --timeframes 1h --start-date 2020-02-01
```

Patch a bounded date range:

```powershell
.\.venv\Scripts\python.exe scripts\data_downloader.py --assets BTC ETH --timeframes 15m 1m --start-date 2024-01-01 --end-date 2024-12-31
```

Single-timeframe legacy flag:

```powershell
.\.venv\Scripts\python.exe scripts\data_downloader.py --assets BTC --timeframe 15m
```

Important:

- If you request only `1h` or `15m`, the downloader warns because Stage 2 and Stage 3 also require `1m`
- If you accept the prompt, `1m` is added automatically
- In non-interactive runs, `1m` is auto-added

## 3. CLI Overview

[`main.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/main.py) is the primary switchboard.

Required argument:

- `--cartridge <name> [<name> ...]`

You can pass one strategy or multiple strategies. Multiple strategies are dynamically fused using strict intersection logic.

Common optional arguments:

- `--trials <int>`: Optuna trials for indicator-parameter search
- `--sltp-trials <int>`: SL/TP search depth for execution modeling
- `--permutations <int>`: Monte Carlo permutation count
- `--target <Long|Short|Total>`: directional optimization target
- `--coins <coin> [<coin> ...]`: limit runs to selected loaded assets
- `--timeframe <1h|15m|1m>`: macro timeframe to process
- `--mode <bar|forwardr|eratio>`: optimization metric mode
- `--matrix <int> [<int> ...]`: horizon matrix for `forwardr` and `eratio`
- `--jobs <int>`: permutation job parallelism
- `--train-window <int>`: override WFO training window in bars
- `--test-window <int>`: override WFO testing window in bars
- `--params-file <path>`: seed parameters from supported JSON outputs
- `--fees`: apply round-trip execution friction during walk-forward stages

## 4. Stage Commands

### Stage 1: Optuna Baseline

Optimize a public strategy on the in-sample vault:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1-optuna --trials 50 --timeframe 1h
```

Directional target example:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1-optuna --trials 100 --target Long --timeframe 1h
```

Two-strategy fusion example:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage1-optuna --trials 100 --timeframe 1h
```

### Stage 1: Permutation Matrix

Test the Stage 1 benchmark against randomized structure:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1-permute --permutations 250 --trials 20 --timeframe 1h
```

### Stage 1.5: Transfer Validation

Apply frozen Stage 1 parameters to unseen Vault 2 data:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1.5-transfer --permutations 250 --timeframe 1h
```

Custom params file example:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1.5-transfer --params-file reports\ema_slope\1h\stage1_forwardr_optuna_results.json
```

### Stage 2: Walk-Forward Optimization

Run the adaptive WFO loop using macro timeframe data plus `1m` execution data:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage2-wfo --timeframe 1h --mode forwardr
```

Explicit windows and seed example:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage2-wfo --timeframe 1h --mode forwardr --train-window 21600 --test-window 2160 --params-file reports\ema_slope\1h\stage1_forwardr_optuna_results.json
```

### Stage 2: WFO Permutation

Test whether the WFO loop survives degraded paths:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage2-permute --permutations 250 --trials 50 --timeframe 1h --mode forwardr
```

### Stage 3: Holdout Verification

Run the same WFO process on the locked holdout vault:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage3-holdout --timeframe 1h --mode forwardr
```

## 5. `--params-file`

Supported JSON inputs:

- raw parameter dictionary
- Stage 1 result JSON containing `optimal_params`
- Stage 2 or Stage 3 WFO JSON containing `rolling_parameters`

If a WFO result JSON is used, the engine takes the last rolling parameter set and strips execution-only keys before reuse.

## 6. `mass_optuna.py`

[`scripts/mass_optuna.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/scripts/mass_optuna.py) batches Stage 1 Optuna scans across strategy combinations.

Base-plus-one search:

```powershell
.\.venv\Scripts\python.exe scripts\mass_optuna.py --base ema_slope --combine 1
```

Restricted pool:

```powershell
.\.venv\Scripts\python.exe scripts\mass_optuna.py --base ema_slope --combine 1 --pool adx_regime vwap_24h cvd_slope
```

Two-base search:

```powershell
.\.venv\Scripts\python.exe scripts\mass_optuna.py --base ema_slope adx_regime --combine 2 --trials 300
```

## 7. Reports, Leaderboards, And Viewer

Outputs are stored under [`reports/`](C:/Users/Ian/Documents/Quant%20Factory%20Public/reports).

- Results JSON and charts: `reports/<cartridge_name>/<timeframe>/`
- Debug logs: `reports/_logs/engine_debug.log`

Generate leaderboards:

```powershell
.\.venv\Scripts\python.exe scripts\generate_leaderboard.py
.\.venv\Scripts\python.exe scripts\generate_permutation_leaderboard.py
.\.venv\Scripts\python.exe scripts\generate_wfo_leaderboard.py
```

Open [`viewer.html`](C:/Users/Ian/Documents/Quant%20Factory%20Public/viewer.html) in a browser and drop in one of:

- `LEADERBOARD.md`
- `PERMUTATION_LEADERBOARD.md`
- `WFO_LEADERBOARD.md`

## 8. Rename Helper For WFO Reports

Preview only:

```powershell
.\.venv\Scripts\python.exe scripts\rename_wfo_reports.py
```

Apply changes:

```powershell
.\.venv\Scripts\python.exe scripts\rename_wfo_reports.py --apply
```

## 9. Notes

- This repo ships no vault data or reports
- Binance Futures is the only supported downloader target in this release
- The repo is script-first; `pyproject.toml` supplies build metadata, dependency metadata, and Python-version constraints
