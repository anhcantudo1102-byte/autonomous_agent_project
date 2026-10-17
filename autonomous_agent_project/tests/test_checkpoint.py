"""Kiểm thử checkpoint: ghi atomic, phiên bản schema, tệp hỏng, bản sao best_agent, resume main.py."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evolution_demo
import main
from core import checkpoint as ckpt
from core.agent import Agent, AgentState
from core.memory import MemoryBank
from core.strategy import Direction, Order, StrategyParams

from tests.helpers import ScriptedEngine, rng, snap


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestDocGhi(TempDirCase):
    def test_ghi_co_schema_version_va_doc_lai(self) -> None:
        path = ckpt.write_checkpoint(self.dir / "a.json", "x", {"n": 1})
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema_version"], ckpt.SCHEMA_VERSION)
        self.assertEqual(raw["kind"], "x")
        self.assertIn("timestamp", raw)
        self.assertEqual(ckpt.read_checkpoint(path, "x", ("n",))["n"], 1)
        self.assertIsNone(ckpt.read_checkpoint(self.dir / "khong_co.json", "x"))

    def test_loi_khi_ghi_giu_nguyen_tep_cu_va_khong_de_lai_tep_tam(self) -> None:
        path = ckpt.write_checkpoint(self.dir / "a.json", "x", {"n": 1})
        with mock.patch("core.checkpoint.os.replace", side_effect=OSError("đĩa đầy")):
            with self.assertRaises(ckpt.CheckpointError):
                ckpt.write_checkpoint(path, "x", {"n": 2})
        self.assertEqual(ckpt.read_checkpoint(path, "x")["n"], 1)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["a.json"])

    def test_du_lieu_khong_tuan_tu_hoa_duoc(self) -> None:
        with self.assertRaises(ckpt.CheckpointError):
            ckpt.write_checkpoint(self.dir / "a.json", "x", {"n": object()})
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_duong_dan_mac_dinh_theo_thu_muc_du_an(self) -> None:
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            path = ckpt.resolve_path()
        finally:
            os.chdir(cwd)
        self.assertEqual(path, Path(main.__file__).resolve().parent / "checkpoints" / ckpt.DEFAULT_FILENAME)
        self.assertEqual(ckpt.resolve_path(self.dir / "b.json", "/khac"), self.dir / "b.json")

    def test_tep_hong_bi_cach_ly(self) -> None:
        path = self.dir / "a.json"
        path.write_text("{hỏng", encoding="utf-8")
        msgs: list[str] = []
        self.assertIsNone(ckpt.load_or_quarantine(path, "x", report=msgs.append))
        self.assertFalse(path.exists())
        self.assertTrue((self.dir / "a.corrupt.json").exists())
        self.assertIn("hỏng", msgs[0])

    def test_schema_moi_hon_bi_tu_choi_nhung_khong_xoa(self) -> None:
        path = self.dir / "a.json"
        path.write_text(json.dumps({"schema_version": ckpt.SCHEMA_VERSION + 1, "kind": "x"}), encoding="utf-8")
        self.assertIsNone(ckpt.load_or_quarantine(path, "x", report=lambda _: None))
        self.assertTrue(path.exists())

    def test_sai_loai_hoac_thieu_truong(self) -> None:
        path = ckpt.write_checkpoint(self.dir / "a.json", "x", {"n": 1})
        with self.assertRaises(ckpt.CheckpointError):
            ckpt.read_checkpoint(path, "y")
        with self.assertRaises(ckpt.CheckpointError):
            ckpt.read_checkpoint(path, "x", ("thieu",))

    def test_rng_khoi_phuc_y_het(self) -> None:
        r = rng(5)
        r.random()
        state = json.loads(json.dumps(ckpt.rng_state(r)))
        self.assertEqual(ckpt.restore_rng(state).random(), r.random())


class TestApiDemo(TempDirCase):
    def test_luu_ban_sao_khong_bi_thay_doi_theo_quan_the(self) -> None:
        pop = [{"id": 0, "genes": [0.5], "fitness": 1.0}]
        quiet(ckpt.save_checkpoint, 1, pop, pop[0], directory=self.dir)
        pop[0]["fitness"] = 99.0
        data, out = quiet(ckpt.load_checkpoint, directory=self.dir)
        self.assertEqual(data["best_agent"]["fitness"], 1.0)
        self.assertEqual(data["population"][0]["fitness"], 1.0)
        self.assertIn("Khôi phục thành công thế hệ 1", out)

    def test_doc_tep_dinh_dang_cu_khong_co_schema(self) -> None:
        legacy = {"generation": 3, "timestamp": "x", "population": [], "best_agent": None}
        (self.dir / ckpt.DEFAULT_FILENAME).write_text(json.dumps(legacy), encoding="utf-8")
        data, _ = quiet(ckpt.load_checkpoint, directory=self.dir)
        self.assertEqual(data["generation"], 3)

    def test_tep_sai_kieu_bi_cach_ly(self) -> None:
        bad = {"schema_version": 1, "kind": ckpt.KIND_EVOLUTION, "generation": "3", "population": [], "best_agent": None}
        (self.dir / ckpt.DEFAULT_FILENAME).write_text(json.dumps(bad), encoding="utf-8")
        data, out = quiet(ckpt.load_checkpoint, directory=self.dir)
        self.assertIsNone(data)
        self.assertTrue((self.dir / "latest_checkpoint.corrupt.json").exists())

    def test_khong_luu_duoc_tra_ve_false(self) -> None:
        with mock.patch("core.checkpoint.write_checkpoint", side_effect=ckpt.CheckpointError("lỗi")):
            ok, out = quiet(ckpt.save_checkpoint, 1, [], None, directory=self.dir)
        self.assertFalse(ok)
        self.assertIn("Lỗi khi lưu checkpoint", out)


class TestEvolutionDemo(TempDirCase):
    def run_demo(self, *argv: str) -> dict:
        return quiet(evolution_demo.run_evolution_with_checkpoint, ["--dir", str(self.dir), *argv])[0]

    def test_chay_hai_lan_tiep_tuc_va_y_het_chay_lien(self) -> None:
        first = self.run_demo("--generations", "3")
        second = self.run_demo("--generations", "3")
        self.assertEqual(first["generation"], 3)
        self.assertEqual(second["generation"], 6)
        straight = self.run_demo("--generations", "6", "--reset", "--file", "lien.json")
        self.assertEqual(straight["population"], second["population"])
        self.assertEqual(straight["best_agent"], second["best_agent"])

    def test_best_agent_la_ban_sao(self) -> None:
        result = self.run_demo("--generations", "4")
        best = result["best_agent"]
        twin = next(a for a in result["population"] if a["id"] == best["id"])
        self.assertIsNot(best, twin)
        self.assertNotEqual(best["genes"], twin["genes"])  # gen của quần thể đã đột biến sau khi ghi nhận

    def test_tep_hong_thi_khoi_tao_lai(self) -> None:
        (self.dir / ckpt.DEFAULT_FILENAME).write_text("không phải JSON", encoding="utf-8")
        self.assertEqual(self.run_demo("--generations", "2")["generation"], 2)


class TestAgentState(unittest.TestCase):
    def test_vong_tron_to_state_from_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            memory = MemoryBank(Path(d) / "m.json", durable=False)
            engine = ScriptedEngine([0.5, -0.2])
            a = Agent("A-007", memory, StrategyParams(leverage=3.0), engine=engine, rng=rng(1),
                      generation=2, parent_id="A-003", birth_tick=9)
            for t in (1, 2, 3):
                a.step(snap(tick=t, momentum=0.02))
            self.assertIsNotNone(a.position)
            state = json.loads(json.dumps(a.to_state()))
            b = Agent.from_state(state, memory, engine)
            self.assertEqual(b.to_state(), a.to_state())
            self.assertEqual(b.position.order, a.position.order)
            self.assertEqual(b.rng.random(), a.rng.random())

    def test_agent_da_chet_khong_giu_vi_the(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            memory = MemoryBank(Path(d) / "m.json", durable=False)
            a = Agent("A-001", memory, rng=rng(0), engine=ScriptedEngine([]))
            a.position = ScriptedEngine([]).open_position(Order(Direction.LONG, 0.5, 2.0), 20, snap(), {})
            a.terminate(4, "thử")
            state = a.to_state()
            state["position"] = {"direction": "LONG", "fraction": 0.5, "leverage": 2.0, "entry_price": 1,
                                 "entry_tick": 1, "notional": 1, "margin": 1, "entry_fee": 0}
            b = Agent.from_state(state, memory)
            self.assertIs(b.state, AgentState.TERMINATED)
            self.assertIsNone(b.position)
            self.assertEqual(b.death_tick, 4)


class TestMainResume(TempDirCase):
    def setUp(self) -> None:
        super().setUp()
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def eco(self, *argv: str, memory: str = "m.json") -> main.Ecosystem:
        args = main.parse_args([
            "--tick", "0", "--no-fsync", "-q",
            "--memory", str(self.dir / memory), "--checkpoint-file", str(self.dir / "c.json"), *argv,
        ])
        return main.Ecosystem(args)

    def final(self, eco: main.Ecosystem) -> list[dict]:
        quiet(eco.run)
        return [ag.to_state() for ag in sorted(eco.agents + eco.graveyard, key=lambda a: a.agent_id)]

    def test_mac_dinh_khong_tao_checkpoint(self) -> None:
        self.final(self.eco("--steps", "30", "--seed", "1"))
        self.assertFalse((self.dir / "c.json").exists())

    def test_resume_cho_ket_qua_y_het_chay_lien(self) -> None:
        straight = self.final(self.eco("--steps", "160", "--seed", "4", memory="a.json"))
        self.final(self.eco("--steps", "90", "--seed", "4", "--checkpoint-every", "45", memory="b.json"))
        data = ckpt.read_checkpoint(self.dir / "c.json", ckpt.KIND_ECOSYSTEM)
        self.assertEqual(data["tick"], 90)
        for key in ("agents", "graveyard", "rng", "market", "fitness_stats", "best_agent", "next_id"):
            self.assertIn(key, data)
        resumed_eco = self.eco("--steps", "70", "--seed", "999", "--resume", memory="b.json")
        self.assertEqual(resumed_eco.start_tick, 90)
        resumed = self.final(resumed_eco)
        self.assertEqual(resumed_eco.current_tick, 160)
        self.assertEqual(resumed, straight)

    def test_ctrl_c_van_luu_checkpoint(self) -> None:
        eco = self.eco("--steps", "100", "--seed", "2", "--checkpoint-every", "1000")
        real_step = eco.market.step

        def step():
            s = real_step()
            if s.tick == 17:
                eco.request_stop()
            return s

        eco.market.step = step
        quiet(eco.run)
        self.assertEqual(ckpt.read_checkpoint(self.dir / "c.json", ckpt.KIND_ECOSYSTEM)["tick"], 17)

    def test_best_agent_la_ban_sao_trang_thai(self) -> None:
        eco = self.eco("--steps", "40", "--seed", "3", "--checkpoint-every", "20")
        quiet(eco.run)
        best = eco.best_agent
        self.assertIsInstance(best, dict)
        saved = best["fitness"]["score"]
        for ag in eco.agents:
            ag.capital += 1000
        self.assertEqual(eco.best_agent["fitness"]["score"], saved)

    def test_resume_khong_co_tep_hoac_tep_hong_thi_khoi_tao_moi(self) -> None:
        eco = self.eco("--steps", "5", "--resume", "--agents", "3")
        self.assertEqual((eco.start_tick, len(eco.agents)), (0, 3))
        (self.dir / "c.json").write_text(json.dumps({"schema_version": 1, "kind": "he_sinh_thai", "tick": 1}),
                                          encoding="utf-8")
        eco = self.eco("--steps", "5", "--resume", "--agents", "3")
        self.assertEqual(eco.start_tick, 0)
        self.assertTrue((self.dir / "c.corrupt.json").exists())

    def test_resume_cung_reset_memory_bi_tu_choi(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--resume", "--reset-memory"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--checkpoint-every", "-1"])


if __name__ == "__main__":
    unittest.main()
