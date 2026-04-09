import argparse
import re
from dataclasses import dataclass
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

STAGE2_LEGACY_RE = re.compile(r"^stage2_(?P<mode>[^_]+)_wfo_results(?:_(?P<label>.*))?\.json$")
STAGE3_LEGACY_RE = re.compile(r"^stage3_(?P<mode>[^_]+)_holdout_results(?:_(?P<label>.*))?\.json$")
WINDOW_LABEL_RE = re.compile(
    r"^(?P<train>\d+(?:\.\d+)?)years?_(?P<test>\d+(?:\.\d+)?)months?(?:_(?P<tag>.*))?$"
)


@dataclass
class RenameAction:
    source: Path
    destination: Path | None
    status: str
    reason: str


def infer_timeframe(path: Path) -> str | None:
    timeframe = path.parent.name
    if timeframe in WINDOW_BARS:
        return timeframe
    return None


def parse_legacy_window_label(label: str | None, timeframe: str) -> tuple[int | None, int | None, str | None]:
    if timeframe not in WINDOW_BARS:
        return None, None, None

    if not label:
        return None, None, None

    if label == "default_windows" or label.startswith("default_windows_"):
        tag = label.removeprefix("default_windows").strip("_") or None
        train_window, test_window = DEFAULT_WINDOWS[timeframe]
        return train_window, test_window, tag

    match = WINDOW_LABEL_RE.match(label)
    if not match:
        return None, None, None

    bases = WINDOW_BARS[timeframe]
    train_window = int(round(float(match.group("train")) * bases["year"]))
    test_window = int(round(float(match.group("test")) * bases["month"]))
    return train_window, test_window, match.group("tag")


def build_canonical_wfo_name(stage: str, mode: str, timeframe: str, train_window: int, test_window: int, tag: str | None = None) -> str:
    if stage == "stage2":
        stem = f"stage2_{mode}_wfo_results_tf-{timeframe}_train-{train_window}b_test-{test_window}b"
    elif stage == "stage3":
        stem = f"stage3_{mode}_holdout_results_tf-{timeframe}_train-{train_window}b_test-{test_window}b"
    else:
        raise ValueError(f"Unsupported WFO stage: {stage}")

    if tag:
        stem = f"{stem}_{tag}"
    return f"{stem}.json"


def parse_legacy_wfo_file(path: Path) -> tuple[str, str, str | None] | None:
    if "_tf-" in path.name or "permutation" in path.name:
        return None

    match = STAGE2_LEGACY_RE.match(path.name)
    if match:
        return "stage2", match.group("mode"), match.group("label")

    match = STAGE3_LEGACY_RE.match(path.name)
    if match:
        return "stage3", match.group("mode"), match.group("label")

    return None


def collect_rename_actions(reports_dir: Path) -> list[RenameAction]:
    actions = []
    for path in sorted(reports_dir.rglob("*.json")):
        parsed = parse_legacy_wfo_file(path)
        if not parsed:
            continue

        timeframe = infer_timeframe(path)
        if timeframe is None:
            actions.append(RenameAction(path, None, "skip", "could not infer timeframe from parent folder"))
            continue

        stage, mode, label = parsed
        train_window, test_window, tag = parse_legacy_window_label(label, timeframe)
        if train_window is None or test_window is None:
            actions.append(RenameAction(path, None, "skip", f"could not parse legacy window label: {label}"))
            continue

        destination = path.with_name(build_canonical_wfo_name(stage, mode, timeframe, train_window, test_window, tag))
        if destination.exists():
            actions.append(RenameAction(path, destination, "skip", "destination already exists"))
        else:
            actions.append(RenameAction(path, destination, "ready", "rename ready"))

    return actions


def print_actions(actions: list[RenameAction], apply_changes: bool) -> None:
    if not actions:
        print("No legacy WFO report files found.")
        return

    for action in actions:
        if action.destination is None:
            print(f"SKIP  {action.source} ({action.reason})")
            continue

        verb = "MOVE" if apply_changes and action.status == "ready" else "DRY "
        if action.status != "ready":
            verb = "SKIP"
        print(f"{verb}  {action.source} -> {action.destination} ({action.reason})")


def run_renames(reports_dir: Path, apply_changes: bool = False) -> list[RenameAction]:
    actions = collect_rename_actions(reports_dir)
    print_actions(actions, apply_changes)

    if apply_changes:
        for action in actions:
            if action.status == "ready" and action.destination is not None:
                action.source.rename(action.destination)

    return actions


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Preview or apply canonical WFO report filename renames.")
    parser.add_argument("--reports-dir", type=Path, default=base_dir / "reports", help="Reports directory to scan")
    parser.add_argument("--apply", action="store_true", help="Actually rename files. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Preview only. This is the default.")
    args = parser.parse_args()

    if not args.reports_dir.exists():
        print(f"No reports directory found: {args.reports_dir}")
        return

    run_renames(args.reports_dir, apply_changes=args.apply)


if __name__ == "__main__":
    main()
