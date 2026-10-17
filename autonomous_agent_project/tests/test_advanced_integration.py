"""Kiểm thử tích hợp --advanced-analysis vào main.py: cờ opt-in, gen tin đa khung, checkpoint, dashboard."""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from core import checkpoint as ckpt
from core import dashboard as dash
from core.advanced_strategy import ConfluenceFeed


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class TestAdvancedAnalysisMain(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def eco(self, *argv: str, memory: str = "m.json") -> main.Ecosystem:
        args = main.parse_args([
            "--tick", "0", "--no-fsync", "-q",
            "--memory", str(self.dir / memory), "--checkpoint-file", str(self.dir / "c.json"), *argv,
        ])
        return main.Ecosystem(args)

    def final(self, eco: main.Ecosystem) -> tuple[list[dict], str]:
        _, out = quiet(eco.run)
        return [ag.to_state() for ag in sorted(eco.agents + eco.graveyard, key=lambda a: a.agent_id)], out

    def test_mac_dinh_tat(self) -> None:
        eco = self.eco("--steps", "40", "--seed", "3")
        self.assertFalse(eco.args.advanced_analysis)
        self.assertIsNone(eco.analysis_feed)
        states, out = self.final(eco)
        self.assertTrue(all("confluence_weight" not in s["params"] for s in states))
        self.assertNotIn("Phân tích đa khung", out)

    def test_bat_co_thi_agent_co_gen_va_bao_cao_co_phan_tich(self) -> None:
        eco = self.eco("--steps", "120", "--seed", "3", "--advanced-analysis", "--htf-ticks", "4")
        self.assertIsInstance(eco.analysis_feed, ConfluenceFeed)
        self.assertEqual(eco.analysis_feed.htf.ticks_per_candle, 4)
        self.assertTrue(all(0 <= ag.params.confluence_weight <= 1 for ag in eco.agents))
        seen: list[float | None] = []
        real_step = eco.agents[0].step

        def spy(snap, **kw):
            seen.append(snap.confluence)
            return real_step(snap, **kw)

        with mock.patch.object(eco.agents[0], "step", side_effect=spy):
            states, out = self.final(eco)
        self.assertTrue(seen and all(isinstance(s, float) for s in seen))
        self.assertEqual(seen[-1], eco.analysis["confluence_score"])
        self.assertIn("Phân tích đa khung (nến gom từ tick)", out)
        self.assertTrue(all("confluence_weight" in s["params"] for s in states))

    def test_cung_seed_hai_che_do_cung_gen_goc(self) -> None:
        plain = self.eco("--steps", "10", "--seed", "11", memory="a.json")
        adv = self.eco("--steps", "10", "--seed", "11", "--advanced-analysis", memory="b.json")
        self.assertEqual(plain.agents[0].params.momentum_threshold, adv.agents[0].params.momentum_threshold)

    def test_resume_tat_dinh_va_luu_trang_thai_phan_tich(self) -> None:
        flags = ("--seed", "5", "--advanced-analysis")
        straight, _ = self.final(self.eco("--steps", "140", *flags, memory="a.json"))
        self.final(self.eco("--steps", "70", *flags, "--checkpoint-every", "70", memory="b.json"))
        data = ckpt.read_checkpoint(self.dir / "c.json", ckpt.KIND_ECOSYSTEM)
        self.assertIn("advanced", data)
        self.assertEqual(data["advanced"]["last"]["tick"], 70)
        resumed_eco = self.eco("--steps", "70", *flags, "--resume", memory="b.json")
        self.assertEqual(resumed_eco.analysis["tick"], 70)
        resumed, _ = self.final(resumed_eco)
        self.assertEqual(resumed, straight)

    def test_resume_checkpoint_cu_thi_gan_gen_moi(self) -> None:
        self.final(self.eco("--steps", "30", "--seed", "6", "--checkpoint-every", "30"))
        eco = self.eco("--steps", "20", "--seed", "6", "--resume", "--advanced-analysis")
        self.assertEqual(eco.start_tick, 30)
        self.assertTrue(all(ag.params.confluence_weight is not None for ag in eco.agents))
        self.assertTrue(all(ag.strategy.params is ag.params for ag in eco.agents))
        self.final(eco)

    def test_tham_so_khong_hop_le(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--htf-ticks", "0"])

    @unittest.skipUnless(dash.RICH_AVAILABLE, "chưa cài rich")
    def test_dashboard_hien_thi_phan_tich(self) -> None:
        from rich.console import Console

        eco = self.eco("--steps", "120", "--seed", "3", "--advanced-analysis")
        eco._run_ticks(120)
        board = dash.EcosystemDashboard(eco, console=Console(file=io.StringIO()), require_terminal=False)
        text = dash.render_to_text(board.render(120), width=150, height=46)
        for label in ("Xu hướng HTF / cấu trúc", "Hidden Base (LTF)", "Sóng Elliott", "Điểm hợp lưu",
                      "Khuyến nghị", "Tin đa khung", eco.analysis["recommended_action"]):
            self.assertIn(label, text)

        plain = self.eco("--steps", "20", "--seed", "3", memory="p.json")
        plain._run_ticks(20)
        board = dash.EcosystemDashboard(plain, console=Console(file=io.StringIO()), require_terminal=False)
        text = dash.render_to_text(board.render(20), width=150, height=46)
        self.assertNotIn("Điểm hợp lưu", text)
        self.assertNotIn("Tin đa khung", text)


if __name__ == "__main__":
    unittest.main()
