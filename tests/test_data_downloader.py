import pytest

from scripts.data_downloader import ensure_execution_timeframe, normalize_asset


class FakeTTY:
    def isatty(self) -> bool:
        return True


class FakePipe:
    def isatty(self) -> bool:
        return False


def test_normalize_asset_accepts_common_public_inputs():
    assert normalize_asset("btc") == "BTCUSDT.P"
    assert normalize_asset("BTCUSDT") == "BTCUSDT.P"
    assert normalize_asset("BTCUSDT.P") == "BTCUSDT.P"


def test_ensure_execution_timeframe_keeps_existing_1m():
    assert ensure_execution_timeframe(["1h", "1m"]) == ["1h", "1m"]


def test_ensure_execution_timeframe_accepts_default_yes(monkeypatch):
    monkeypatch.setattr("scripts.data_downloader.sys.stdin", FakeTTY())
    monkeypatch.setattr("builtins.input", lambda _: "")

    assert ensure_execution_timeframe(["1h", "15m"]) == ["1h", "15m", "1m"]


def test_ensure_execution_timeframe_aborts_when_user_declines(monkeypatch):
    monkeypatch.setattr("scripts.data_downloader.sys.stdin", FakeTTY())
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(SystemExit, match="require 1m data"):
        ensure_execution_timeframe(["1h"])


def test_ensure_execution_timeframe_auto_adds_for_non_interactive_runs(monkeypatch):
    monkeypatch.setattr("scripts.data_downloader.sys.stdin", FakePipe())

    assert ensure_execution_timeframe(["15m"]) == ["15m", "1m"]
