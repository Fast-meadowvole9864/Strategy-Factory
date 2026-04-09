import argparse
import json
import re
from pathlib import Path


WINDOW_BARS = {
    "1h": {"month": 720, "year": 8640},
    "15m": {"month": 2880, "year": 34560},
    "1m": {"month": 43200, "year": 518400},
}

DEFAULT_WINDOWS = {
    "1h": (4320, 720),
    "15m": (17280, 2880),
    "1m": (259200, 43200),
}

NEW_STAGE2_RE = re.compile(
    r"^stage2_(?P<mode>[^_]+)_wfo_results_tf-(?P<timeframe>[^_]+)_train-(?P<train>\d+)b_test-(?P<test>\d+)b(?:_(?P<tag>.*))?\.json$"
)
NEW_STAGE3_RE = re.compile(
    r"^stage3_(?P<mode>[^_]+)_holdout_results_tf-(?P<timeframe>[^_]+)_train-(?P<train>\d+)b_test-(?P<test>\d+)b(?:_(?P<tag>.*))?\.json$"
)
LEGACY_STAGE2_RE = re.compile(r"^stage2_(?P<mode>[^_]+)_wfo_results(?:_(?P<label>.*))?\.json$")
LEGACY_STAGE3_RE = re.compile(r"^stage3_(?P<mode>[^_]+)_holdout_results(?:_(?P<label>.*))?\.json$")
WINDOW_LABEL_RE = re.compile(
    r"^(?P<train>\d+(?:\.\d+)?)years?_(?P<test>\d+(?:\.\d+)?)months?(?:_(?P<tag>.*))?$"
)

STAGE_ORDER = {"Stage 2 (WFO)": 0, "Stage 3 (Holdout)": 1}
TIMEFRAME_ORDER = {"1h": 0, "15m": 1, "1m": 2}
PUBLIC_COLUMNS = [
    "Stage",
    "Timeframe",
    "Mode",
    "Strategy",
    "Train Window",
    "Test Window",
    "Profit Factor",
    "Total PnL",
    "Total Trades",
    "Update",
    "Source File",
]


def parse_legacy_window_label(label: str | None, timeframe: str) -> tuple[int | None, int | None]:
    if timeframe not in WINDOW_BARS:
        return None, None

    if not label:
        return None, None

    if label == "default_windows" or label.startswith("default_windows_"):
        return DEFAULT_WINDOWS[timeframe]

    match = WINDOW_LABEL_RE.match(label)
    if not match:
        return None, None

    bases = WINDOW_BARS[timeframe]
    train_window = int(round(float(match.group("train")) * bases["year"]))
    test_window = int(round(float(match.group("test")) * bases["month"]))
    return train_window, test_window


def parse_wfo_filename(path: Path) -> dict | None:
    for pattern, stage_name in ((NEW_STAGE2_RE, "Stage 2 (WFO)"), (NEW_STAGE3_RE, "Stage 3 (Holdout)")):
        match = pattern.match(path.name)
        if match:
            return {
                "stage": stage_name,
                "mode": match.group("mode"),
                "timeframe": match.group("timeframe"),
                "train_window": int(match.group("train")),
                "test_window": int(match.group("test")),
            }

    for pattern, stage_name in ((LEGACY_STAGE2_RE, "Stage 2 (WFO)"), (LEGACY_STAGE3_RE, "Stage 3 (Holdout)")):
        match = pattern.match(path.name)
        label = match.group("label") if match else ""
        if match and not (label or "").startswith("tf-"):
            timeframe = path.parent.name
            train_window, test_window = parse_legacy_window_label(label, timeframe)
            return {
                "stage": stage_name,
                "mode": match.group("mode"),
                "timeframe": timeframe,
                "train_window": train_window,
                "test_window": test_window,
            }

    return None


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def format_window(value) -> str:
    if value is None:
        return "unknown"
    return f"{int(value)}b"


def format_source(path: Path, base_dir: Path) -> str:
    try:
        source = path.relative_to(base_dir)
    except ValueError:
        source = path
    return source.as_posix()


def update_marker(path: Path) -> str:
    filename = path.name.lower()
    if "pre_update" in filename:
        return "Pre Update"
    if "post_update" in filename:
        return "Post Update"
    return ""


def parse_wfo_report(path: Path, base_dir: Path) -> dict | None:
    filename_meta = parse_wfo_filename(path)
    if not filename_meta:
        return None

    data = read_json(path)
    timeframe = data.get("timeframe", filename_meta["timeframe"])
    strategy_name = data.get("strategy") or path.parent.parent.name
    if str(strategy_name).endswith("_OLD"):
        return None

    train_window = data.get("train_window", filename_meta["train_window"])
    test_window = data.get("test_window", filename_meta["test_window"])
    profit_factor = as_float(data.get("profit_factor", 0.0))
    total_pnl = as_float(data.get("total_pnl", 0.0))
    total_trades = as_int(data.get("total_trades", 0))

    return {
        "Stage": filename_meta["stage"],
        "Timeframe": timeframe,
        "Mode": filename_meta["mode"],
        "Strategy": strategy_name,
        "Train Window": format_window(train_window),
        "Test Window": format_window(test_window),
        "Profit Factor": f"{profit_factor:.6f}",
        "Total PnL": f"{total_pnl:.6f}",
        "Total Trades": total_trades,
        "Update": update_marker(path),
        "Source File": format_source(path, base_dir),
        "_profit_factor_value": profit_factor,
        "_total_pnl_value": total_pnl,
        "_total_trades_value": total_trades,
    }


def collect_wfo_results(reports_dir: Path, base_dir: Path) -> list[dict]:
    results = []
    if not reports_dir.exists():
        return results

    for path in sorted(reports_dir.rglob("*.json")):
        try:
            row = parse_wfo_report(path, base_dir)
        except Exception as exc:
            print(f"Failed to parse {path}: {exc}")
            continue
        if row:
            results.append(row)

    return results


def sort_key(row: dict) -> tuple:
    return (
        STAGE_ORDER.get(row["Stage"], 99),
        TIMEFRAME_ORDER.get(str(row["Timeframe"]), 99),
        row["Mode"],
        -row["_profit_factor_value"],
        -row["_total_pnl_value"],
        -row["_total_trades_value"],
    )


def markdown_escape(value) -> str:
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict]) -> str:
    lines = []
    lines.append("| " + " | ".join(PUBLIC_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(PUBLIC_COLUMNS)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(row[column]) for column in PUBLIC_COLUMNS) + " |")
    return "\n".join(lines)


def build_markdown(results: list[dict]) -> str:
    if not results:
        return "# Quant Engine: WFO Leaderboard\n\nNo valid Stage 2/3 WFO results found.\n"

    results = sorted(results, key=sort_key)
    output = "# Quant Engine: WFO Leaderboard\n\n"
    output += "This leaderboard scans Stage 2 WFO and Stage 3 holdout JSON reports only.\n\n"

    stages = sorted({row["Stage"] for row in results}, key=lambda stage: STAGE_ORDER.get(stage, 99))
    for stage in stages:
        output += f"## {stage}\n"
        stage_rows = [row for row in results if row["Stage"] == stage]
        timeframes = sorted({row["Timeframe"] for row in stage_rows}, key=lambda tf: TIMEFRAME_ORDER.get(str(tf), 99))

        for timeframe in timeframes:
            output += f"### Timeframe: {timeframe}\n"
            tf_rows = [row for row in stage_rows if row["Timeframe"] == timeframe]
            output += markdown_table(tf_rows) + "\n\n"

    return output


def generate_wfo_leaderboard(base_dir: Path | None = None, reports_dir: Path | None = None, output_file: Path | None = None) -> list[dict]:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    if reports_dir is None:
        reports_dir = base_dir / "reports"
    if output_file is None:
        output_file = base_dir / "WFO_LEADERBOARD.md"

    results = collect_wfo_results(reports_dir, base_dir)
    markdown = build_markdown(results)
    output_file.write_text(markdown, encoding="utf-8")
    print(f"Successfully generated {output_file.name} ranking {len(results)} WFO reports!")
    return results


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate a Stage 2/3 WFO-only leaderboard.")
    parser.add_argument("--reports-dir", type=Path, default=base_dir / "reports", help="Reports directory to scan")
    parser.add_argument("--output", type=Path, default=base_dir / "WFO_LEADERBOARD.md", help="Markdown output file")
    args = parser.parse_args()

    generate_wfo_leaderboard(base_dir=base_dir, reports_dir=args.reports_dir, output_file=args.output)


if __name__ == "__main__":
    main()
