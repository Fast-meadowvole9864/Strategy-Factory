import json
from pathlib import Path

import polars as pl

from main import calculate_per_asset_start_index, load_params_from_json, make_wfo_results_filename, resolve_wfo_windows


def load_params_payload(payload):
    path = Path(__file__).with_name("_tmp_params_main.json")
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_params_from_json(str(path))
    finally:
        path.unlink(missing_ok=True)


def test_calculate_per_asset_start_index_uses_single_symbol_slice():
    df = pl.DataFrame({
        "timestamp": [1, 2, 3, 1, 2, 3],
        "symbol": ["btc", "btc", "btc", "eth", "eth", "eth"],
    })

    assert calculate_per_asset_start_index(df) == 3


def test_load_params_from_json_accepts_raw_params():
    assert load_params_payload({"ema_length": 150}) == {"ema_length": 150}


def test_load_params_from_json_accepts_stage1_result():
    assert load_params_payload({"optimal_params": {"ema_length": 150}}) == {"ema_length": 150}


def test_load_params_from_json_uses_last_wfo_roll_and_strips_execution_keys():
    params = load_params_payload({
        "rolling_parameters": [
            {"ema_length": 100, "sl_pct": 0.02},
            {"ema_length": 150, "tp_pct": 0.1, "_test_start_ts": 123, "roll_pnl": 1.0, "roll_trades": 4},
        ]
    })

    assert params == {"ema_length": 150}


def test_resolve_wfo_windows_defaults_and_custom_test_window():
    assert resolve_wfo_windows("1h") == (4320, 720)
    assert resolve_wfo_windows("1h", train_window=21600) == (21600, 3600)
    assert resolve_wfo_windows("15m", test_window=1440) == (17280, 1440)


def test_make_wfo_results_filename_includes_runtime_metadata():
    assert make_wfo_results_filename("stage2", "forwardr", "1h", 21600, 2160) == (
        "stage2_forwardr_wfo_results_tf-1h_train-21600b_test-2160b.json"
    )
    assert make_wfo_results_filename("stage3", "eratio", "15m", 17280, 2880) == (
        "stage3_eratio_holdout_results_tf-15m_train-17280b_test-2880b.json"
    )
