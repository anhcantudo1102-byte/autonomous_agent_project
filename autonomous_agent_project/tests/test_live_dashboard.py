"""Kiểm thử run_simulation_with_dashboard và cờ --live-dashboard (giá Binance được mock, không cần mạng)."""

from __future__ import annotations

import contextlib
import io
import itertools
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from core import dashboard as dash
from core.price_feed import BINANCE_ENDPOINTS


def mock_prices(start: float = 82000.0):
    """``fetch_price`` giả: giá tăng dần 0.1% mỗi lần gọi."""
    counter = itertools.count()
    return lambda *a, **k: (start * (1.001 ** next(counter)), BINANCE_ENDPOINTS[0])


class TestLiveDashboardFlag(unittest.TestCase):
    def test_mac_dinh_khong_doi(self) -> None:
        a = main.parse_args([])
        self.assertEqual((a.steps, a.live_price, a.dashboard, a.live_dashboard), (500, False, False, False))
        self.assertEqual(a.tick, main.DEFAULT_TICK_SIMULATED)

    def test_viet_tat(self) -> None:
        a = main.parse_args(["--live-dashboard"])
        self.assertEqual((a.live_price, a.dashboard, a.steps, a.tick, a.symbol), (True, True, 30, 2.0, "BTCUSDT"))

    def test_co_ghi_ro_duoc_uu_tien(self) -> None:
        a = main.parse_args(["--steps", "5", "--live-dashboard", "--tick", "0", "--symbol", "ETHUSDT"])
        self.assertEqual((a.steps, a.tick, a.symbol, a.live_price, a.dashboard), (5, 0.0, "ETHUSDT", True, True))

    def test_kwargs_to_argv(self) -> None:
        self.assertEqual(
            main.kwargs_to_argv({"seed": 7, "reset_memory": True, "quiet": False, "memory": None, "checkpoint_every": 10}),
            ["--seed", "7", "--reset-memory", "--checkpoint-every", "10"],
        )


class TestRunSimulationWithDashboard(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def run_fn(self, **kwargs) -> tuple[int, str]:
        kwargs.setdefault("memory", self.dir / "m.json")
        kwargs.setdefault("quiet", True)
        kwargs.setdefault("no_fsync", True)
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = main.run_simulation_with_dashboard(**kwargs)
        return code, out.getvalue()

    @unittest.skipUnless(dash.RICH_AVAILABLE, "chưa cài rich")
    def test_mo_phong_that_voi_gia_mock_va_dashboard(self) -> None:
        from rich.console import Console

        console = Console(file=io.StringIO(), width=150, height=50, force_terminal=True, color_system=None,
                          _environ={"TERM": "xterm-256color"})
        handler_before = signal.getsignal(signal.SIGINT)
        with mock.patch("core.price_feed.fetch_price", side_effect=mock_prices()), \
                mock.patch.object(dash, "Console", return_value=console):
            code, out = self.run_fn(symbol="ethusdt", steps=6, tick_interval=0, seed=3,
                                    checkpoint_every=3, checkpoint_file=self.dir / "c.json")
        self.assertEqual(code, 0)
        self.assertEqual(signal.getsignal(signal.SIGINT), handler_before)
        screen = console.file.getvalue()
        for s in ("ETHUSDT", "Binance ETHUSDT (thật)", "6 / 6", "+0.10%", "Quần thể đang sống (5)", "A-001", "ID: A-0"):
            self.assertIn(s, screen)
        self.assertIn("BÁO CÁO CUỐI CÙNG sau 6 tick", out)
        self.assertTrue((self.dir / "c.json").exists())

    @unittest.skipUnless(dash.RICH_AVAILABLE, "chưa cài rich")
    def test_khong_phai_tty_va_api_loi_van_chay(self) -> None:
        with mock.patch("core.price_feed.fetch_price", return_value=(None, None)):
            with self.assertLogs("he_sinh_thai", level="WARNING") as logs:
                code, out = self.run_fn(steps=4, tick_interval=0, seed=1)
        self.assertEqual(code, 0)
        text = "\n".join(logs.output)
        self.assertIn("Kích hoạt chế độ giả lập với giá khởi điểm thực tế: 82,000.00 cho BTCUSDT", text)
        self.assertIn("Không bật được --dashboard", text)
        self.assertIn("BÁO CÁO CUỐI CÙNG sau 4 tick", out)

    def test_cli_live_dashboard_goi_cung_luong(self) -> None:
        with mock.patch("core.price_feed.fetch_price", side_effect=mock_prices()), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = main.main(["--live-dashboard", "--steps", "3", "--tick", "0", "-q", "--no-fsync",
                              "--memory", str(self.dir / "m.json")])
        self.assertEqual(code, 0)
        self.assertIn("BÁO CÁO CUỐI CÙNG sau 3 tick", out.getvalue())


if __name__ == "__main__":
    unittest.main()
