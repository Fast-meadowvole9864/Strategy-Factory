import json
import shutil
from pathlib import Path

from scripts.generate_wfo_leaderboard import build_markdown, collect_wfo_results, parse_wfo_report
from scripts.rename_wfo_reports import collect_rename_actions, parse_legacy_window_label


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def test_parse_legacy_window_label_converts_years_and_months_to_bars():
    assert parse_legacy_window_label("2.5year_3month_post_update", "1h") == (21600, 2160, "post_update")
    assert parse_legacy_window_label("default_windows_pre_update", "15m") == (17280, 2880, "pre_update")
    assert parse_legacy_window_label("first_fee_test", "1h") == (None, None, None)


def test_rename_wfo_reports_dry_run_builds_canonical_destination():
    root = Path(__file__).with_name("_tmp_wfo_reports")
    reset_dir(root)
    try:
        report_dir = root / "example_strategy" / "1h"
        report_dir.mkdir(parents=True)
        source = report_dir / "stage2_forwardr_wfo_results_2.5year_3month_post_update.json"
        source.write_text("{}", encoding="utf-8")

        actions = collect_rename_actions(root)

        assert len(actions) == 1
        assert actions[0].status == "ready"
        assert actions[0].destination.name == "stage2_forwardr_wfo_results_tf-1h_train-21600b_test-2160b_post_update.json"
        assert source.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_parse_wfo_report_prefers_json_window_metadata():
    root = Path(__file__).with_name("_tmp_wfo_reports")
    reset_dir(root)
    try:
        report_dir = root / "example_strategy" / "1h"
        report_dir.mkdir(parents=True)
        report_file = report_dir / "stage2_forwardr_wfo_results_tf-1h_train-21600b_test-2160b_pre_update.json"
        report_file.write_text(json.dumps({
            "strategy": "Example_Strategy",
            "timeframe": "15m",
            "train_window": 17280,
            "test_window": 2880,
            "profit_factor": 1.23456789,
            "total_pnl": 0.5,
            "total_trades": 42,
        }), encoding="utf-8")

        row = parse_wfo_report(report_file, root)

        assert row["Timeframe"] == "15m"
        assert row["Train Window"] == "17280b"
        assert row["Test Window"] == "2880b"
        assert row["Profit Factor"] == "1.234568"
        assert row["Update"] == "Pre Update"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_generate_wfo_leaderboard_sorts_by_profit_factor_then_pnl():
    root = Path(__file__).with_name("_tmp_wfo_reports")
    reset_dir(root)
    try:
        report_dir = root / "example_strategy" / "1h"
        report_dir.mkdir(parents=True)
        low_pf = report_dir / "stage2_forwardr_wfo_results_tf-1h_train-21600b_test-2160b_a.json"
        high_pf = report_dir / "stage2_forwardr_wfo_results_tf-1h_train-21600b_test-2160b_b.json"
        low_pf.write_text(json.dumps({
            "strategy": "LowPF",
            "profit_factor": 1.1,
            "total_pnl": 10.0,
            "total_trades": 10,
        }), encoding="utf-8")
        high_pf.write_text(json.dumps({
            "strategy": "HighPF",
            "profit_factor": 1.2,
            "total_pnl": 1.0,
            "total_trades": 5,
        }), encoding="utf-8")

        rows = collect_wfo_results(root, root)
        markdown = build_markdown(rows)

        assert markdown.index("HighPF") < markdown.index("LowPF")
        assert "| Update |" in markdown
        assert "WFO Leaderboard" in markdown
    finally:
        shutil.rmtree(root, ignore_errors=True)
