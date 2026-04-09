# Strategy Factory Public Core

Strategy Factory Public Core is a modular quantitative research engine focused on robust validation rather than naive backtest maximization. It is built around walk-forward optimization, transfer validation, holdout discipline, and permutation testing.

This public repo is meant to teach and demonstrate the methodology and architecture. It does not include private research outputs, private data, or private strategy families.

## What This Project Does

The engine pushes a strategy through a staged research pipeline:

1. Stage 1: Optimize a strategy on the in-sample vault and establish a baseline
2. Stage 1 Permute: Test whether the discovered edge survives randomized market structure
3. Stage 1.5 Transfer: Apply frozen Stage 1 parameters to unseen data without WFO assistance
4. Stage 2 WFO: Run rolling walk-forward adaptation using macro bars plus `1m` execution data
5. Stage 2 Permute: Test the WFO loop itself against degraded paths
6. Stage 3 Holdout: Run the same WFO procedure on the locked holdout vault

The main goal is not "find the biggest backtest." The main goal is to reject fragile ideas early and keep only strategies that survive multiple forms of out-of-sample stress.

## What Is Included

- Engine modules for loading data, optimization, walk-forward simulation, and permutation testing
- Example public strategies in [`strategies/`](C:/Users/Ian/Documents/Quant%20Factory%20Public/strategies)
- Batch tooling such as [`scripts/mass_optuna.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/scripts/mass_optuna.py)
- Report utilities and leaderboard generators
- A standalone viewer in [`viewer.html`](C:/Users/Ian/Documents/Quant%20Factory%20Public/viewer.html) for leaderboard markdown files
- A test suite covering the public-core surface

## What Is Intentionally Not Included

- Private vault data
- Private reports
- Private leaderboard artifacts
- Private/internal strategy families
- Private research notes or operator-only conclusions

## Data Model

The project uses a fixed three-vault layout:

- `Vault_1_InSample`
- `Vault_2_OOS`
- `Vault_3_Holdout`

The downloader routes rows into those vaults by year. That fixed layout is part of the methodology, not an accident.

Important: Stage 2 and Stage 3 require both:

- your selected macro timeframe such as `1h` or `15m`
- `1m` execution data

The `1m` data is used to resolve stop-loss/take-profit execution without making intrabar assumptions. The downloader now warns and adds `1m` if you request only macro timeframes.

## Python Version

Use Python `3.12.x`.

This repo was verified with Python `3.12.10`, and [`pyproject.toml`](C:/Users/Ian/Documents/Quant%20Factory%20Public/pyproject.toml) declares:

- `requires-python = ">=3.12,<3.13"`

If you use a different interpreter family, some pinned scientific packages may fail to install or behave differently.

## Setup

PowerShell examples below use the virtual environment's executables directly instead of `activate`, which avoids execution-policy issues on some Windows setups.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quickstart

Download data:

```powershell
.\.venv\Scripts\python.exe scripts\data_downloader.py --assets BTC ETH --timeframes 1h
```

That command will also prompt to include `1m` data, because later walk-forward stages require it.

Run Stage 1 optimization:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1-optuna --trials 50 --timeframe 1h
```

Run Stage 1 permutation testing:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1-permute --permutations 250 --trials 20 --timeframe 1h
```

Run Stage 1.5 transfer validation:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope --stage1.5-transfer --permutations 250 --timeframe 1h
```

Run Stage 2 walk-forward:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage2-wfo --timeframe 1h --mode forwardr
```

Run Stage 3 holdout:

```powershell
.\.venv\Scripts\python.exe main.py --cartridge ema_slope adx_regime --stage3-holdout --timeframe 1h --mode forwardr
```

## Batch Research

[`scripts/mass_optuna.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/scripts/mass_optuna.py) can batch Stage 1 scans across many public strategies.

Example:

```powershell
.\.venv\Scripts\python.exe scripts\mass_optuna.py --base ema_slope --combine 1 --trials 150
```

This is useful when you want to churn through many strategy combinations without writing custom orchestration code.

## Reports And Viewer

Outputs are written under [`reports/`](C:/Users/Ian/Documents/Quant%20Factory%20Public/reports).

- JSON reports and charts go under `reports/<cartridge_name>/<timeframe>/`
- Engine debug logs go under `reports/_logs/`

Leaderboard tools:

- [`scripts/generate_leaderboard.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/scripts/generate_leaderboard.py)
- [`scripts/generate_permutation_leaderboard.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/scripts/generate_permutation_leaderboard.py)
- [`scripts/generate_wfo_leaderboard.py`](C:/Users/Ian/Documents/Quant%20Factory%20Public/scripts/generate_wfo_leaderboard.py)

[`viewer.html`](C:/Users/Ian/Documents/Quant%20Factory%20Public/viewer.html) can parse dropped-in `LEADERBOARD.md`, `PERMUTATION_LEADERBOARD.md`, and `WFO_LEADERBOARD.md` files.

## More Detail

For the operator-facing command reference, see [`MANUAL.md`](C:/Users/Ian/Documents/Quant%20Factory%20Public/MANUAL.md).

## License

This repo is released under the GNU General Public License v3.0. See [`LICENSE`](C:/Users/Ian/Documents/Quant%20Factory%20Public/LICENSE).
