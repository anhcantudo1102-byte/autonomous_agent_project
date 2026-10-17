"""Kiểm thử dashboard rich: render ra chuỗi (không cần TTY), sự kiện, tích hợp main.py, dự phòng."""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import main
from core import checkpoint as ckpt
from core import dashboard as dash
from core.dashboard import (
    AgentRow,
    DashboardEvent,
    DashboardUnavailable,
    EcosystemDashboard,
    EcosystemSnapshot,
    generate_dashboard_view,
    render_to_text,
)

needs_rich = unittest.skipUnless(dash.RICH_AVAILABLE, "chưa cài rich")


def make_console(width: int = 150, height: int = 50):
    from rich.console import Console

    return Console(file=io.StringIO(), width=width, height=height, force_terminal=True, color_system=None,
                   _environ={"TERM": "xterm-256color"})


@needs_rich
class TestGenerateView(unittest.TestCase):
    def view(self, **kw) -> str:
        args = dict(step=3, generation=2, symbol="BTCUSDT", current_price=82046.5, price_change=0.15,
                    population_stats=100, best_agent={"id": 7, "fitness": 145.2})
        args.update(kw)
        return render_to_text(generate_dashboard_view(**args), width=150, height=45)

    def test_ban_goc_render_duoc(self) -> None:
        text = self.view()
        for s in ("AUTONOMOUS SURVIVAL AGENT DASHBOARD", "BTCUSDT", "82,046.50 USDT", "+0.15%", "ID: 7", "145.20",
                  "100 / 0", "Ctrl+C"):
            self.assertIn(s, text)

    def test_best_agent_none_hien_na(self) -> None:
        text = self.view(best_agent=None)
        self.assertIn("ID: N/A", text)

    def test_best_agent_thieu_id_va_fitness_dang_dict(self) -> None:
        self.assertIn("ID: N/A", self.view(best_agent={"fitness": 1.0}))
        self.assertIn("3.25", self.view(best_agent={"id": "A-002", "fitness": {"score": 3.25}}))

    def test_bang_quan_the_xep_theo_fitness_va_cat_bot(self) -> None:
        rows = [AgentRow(f"A-{i:03d}", i % 3, 20.0 + i, float(i), "ACTIVE", "A-001" if i % 2 else None)
                for i in range(1, 16)]
        stats = EcosystemSnapshot(alive=15, dead=4, diversity=0.321, mutation_rate=0.14, price_source="Binance BTCUSDT (thật)",
                                  agents=rows)
        text = self.view(population_stats=stats)
        self.assertLess(text.index("A-015"), text.index("A-010"))
        self.assertIn("… và 3 agent khác", text)
        self.assertNotIn("A-003 ", text)
        for s in ("15 / 4", "0.321", "0.14", "Binance BTCUSDT (thật)", "ACTIVE", "$35.00"):
            self.assertIn(s, text)

    def test_su_kien_gan_nhat_moi_nhat_truoc(self) -> None:
        events = [DashboardEvent(5, "sinh", "A-006 ra đời từ A-001"), DashboardEvent(9, "chet", "A-002 tự hủy"),
                  DashboardEvent(12, "cam", "Quy tắc CẤM mới QT-0003")]
        text = self.view(population_stats=EcosystemSnapshot(alive=1, events=events))
        self.assertLess(text.index("QT-0003"), text.index("A-002 tự hủy"))
        self.assertLess(text.index("A-002 tự hủy"), text.index("A-006 ra đời"))

    def test_population_stats_dict_va_none(self) -> None:
        self.assertIn("4 / 2", self.view(population_stats={"alive": 4, "dead": 2, "khong_biet": 1}))
        self.assertIn("Chưa có agent nào đang sống", self.view(population_stats=None))


class TestKhongCoRich(unittest.TestCase):
    def test_bao_loi_ro_rang(self) -> None:
        with mock.patch.object(dash, "RICH_AVAILABLE", False):
            with self.assertRaises(DashboardUnavailable):
                generate_dashboard_view(1, 0, "X", 1.0, 0.0, None, None)
            with self.assertRaises(DashboardUnavailable):
                EcosystemDashboard(SimpleNamespace())


def fake_eco(rules=None):
    memory = SimpleNamespace(rules=lambda: list(rules or []))
    return SimpleNamespace(agents=[], graveyard=[], memory=memory)


def fake_agent(agent_id, parent=None, generation=0, cause=None, death_tick=None):
    return SimpleNamespace(agent_id=agent_id, parent_id=parent, generation=generation,
                           death_cause=cause, death_tick=death_tick)


@needs_rich
class TestCollectEvents(unittest.TestCase):
    def test_sinh_chet_va_quy_tac_cam(self) -> None:
        rules = [{"id": "QT-0001", "hanh_dong": "CAM", "mo_ta": "cũ"}]
        eco = fake_eco(rules)
        eco.agents = [fake_agent("A-001")]
        d = EcosystemDashboard(eco, console=make_console())
        self.assertEqual(d.collect_events(1), [])

        child = fake_agent("A-002", "A-001", 1)
        eco.agents.append(child)
        rules += [{"id": "QT-0002", "hanh_dong": "CAM", "mo_ta": "Cấm MUA khi biến động cao"},
                  {"id": "QT-0003", "hanh_dong": "GIAM_QUY_MO", "mo_ta": "giảm"}]
        events = d.collect_events(5)
        self.assertEqual([e.kind for e in events], ["sinh", "cam"])
        self.assertIn("A-002 ra đời từ A-001 (thế hệ 1)", events[0].text)
        self.assertIn("QT-0002", events[1].text)

        eco.agents.remove(child)
        eco.graveyard.append(fake_agent("A-002", "A-001", 1,
                                        "Vốn cạn còn $0.40 (≤ $0.50) tại tick 7. Chết đói: chi phí vận hành bào mòn vốn", 7))
        events = d.collect_events(8)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].kind, events[0].tick), ("chet", 7))
        self.assertIn("Chết đói", events[0].text)
        self.assertEqual(d.collect_events(9), [])

    def test_tu_choi_terminal_khong_ho_tro(self) -> None:
        from rich.console import Console

        plain = Console(file=io.StringIO(), force_terminal=False)
        with self.assertRaises(DashboardUnavailable):
            EcosystemDashboard(fake_eco(), console=plain)
        EcosystemDashboard(fake_eco(), console=plain, require_terminal=False)


class TestMainDashboard(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def eco(self, *argv: str) -> main.Ecosystem:
        args = main.parse_args(["--tick", "0", "--no-fsync", "--memory", str(self.dir / "m.json"),
                                "--checkpoint-file", str(self.dir / "c.json"), *argv])
        return main.Ecosystem(args)

    def run_eco(self, eco: main.Ecosystem) -> str:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            eco.run()
        return out.getvalue()

    def test_mac_dinh_tat(self) -> None:
        self.assertFalse(main.parse_args([]).dashboard)

    def test_khong_phai_tty_thi_canh_bao_va_chay_log_thuong(self) -> None:
        eco = self.eco("--steps", "10", "--seed", "1", "--dashboard", "-q")
        with self.assertLogs("he_sinh_thai", level="WARNING") as logs:
            out = self.run_eco(eco)
        self.assertTrue(any("Không bật được --dashboard" in m for m in logs.output))
        self.assertIn("BÁO CÁO CUỐI CÙNG sau 10 tick", out)

    def test_chua_cai_rich_thi_canh_bao(self) -> None:
        eco = self.eco("--steps", "5", "--seed", "1", "--dashboard", "-q")
        with mock.patch.object(dash, "RICH_AVAILABLE", False), self.assertLogs("he_sinh_thai", level="WARNING") as logs:
            out = self.run_eco(eco)
        self.assertTrue(any("rich" in m for m in logs.output))
        self.assertIn("BÁO CÁO CUỐI CÙNG", out)

    @needs_rich
    def test_dashboard_chay_that_va_ctrl_c_luu_checkpoint(self) -> None:
        console = make_console()
        eco = self.eco("--steps", "400", "--seed", "7", "--dashboard", "--checkpoint-every", "1000", "-q")
        real_step = eco.market.step

        def step():
            s = real_step()
            if s.tick == 250:
                eco.request_stop()
            return s

        eco.market.step = step
        handlers_before = list(logging.getLogger().handlers)
        with mock.patch.object(dash, "Console", return_value=console), self.assertLogs("he_sinh_thai", "WARNING"):
            out = self.run_eco(eco)
        self.assertEqual(logging.getLogger().handlers, handlers_before)
        screen = console.file.getvalue()
        for s in ("AUTONOMOUS SURVIVAL AGENT DASHBOARD", "Quần thể đang sống", "250 / 400", "Checkpoint: c.json",
                  "Đa dạng gen", "Biên độ đột biến", "giả lập"):
            self.assertIn(s, screen)
        self.assertIn("ra đời từ", screen)
        self.assertIn("BÁO CÁO CUỐI CÙNG sau 250 tick", out)
        self.assertEqual(ckpt.read_checkpoint(self.dir / "c.json", ckpt.KIND_ECOSYSTEM)["tick"], 250)

    @needs_rich
    def test_dashboard_khong_doi_ket_qua_mo_phong(self) -> None:
        plain = self.eco("--steps", "120", "--seed", "4", "-q")
        self.run_eco(plain)
        (self.dir / "m.json").unlink()
        shown = self.eco("--steps", "120", "--seed", "4", "--dashboard", "-q")
        with mock.patch.object(dash, "Console", return_value=make_console()):
            self.run_eco(shown)
        key = lambda e: [a.to_state() for a in sorted(e.agents + e.graveyard, key=lambda a: a.agent_id)]
        self.assertEqual(key(plain), key(shown))


@needs_rich
class TestDashboardDemo(unittest.TestCase):
    def test_demo_chay_doc_lap(self) -> None:
        import dashboard_demo

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dashboard_demo.run_dashboard_demo(["--steps", "8", "--delay", "0"]), 0)
        self.assertIn("AUTONOMOUS SURVIVAL AGENT DASHBOARD", out.getvalue())

    def test_demo_khong_co_rich(self) -> None:
        import dashboard_demo

        with mock.patch.object(dashboard_demo, "RICH_AVAILABLE", False), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(dashboard_demo.run_dashboard_demo(["--steps", "1"]), 1)
        self.assertIn("pip install rich", err.getvalue())


if __name__ == "__main__":
    unittest.main()
