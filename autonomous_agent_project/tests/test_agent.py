"""Kiểm thử vòng đời agent: Tự hủy và Tự nhân bản."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.agent import (
    DEATH_THRESHOLD,
    INITIAL_CAPITAL,
    REPLICATION_COST,
    Agent,
    AgentState,
)
from core.memory import MemoryBank
from core.strategy import PARAM_BOUNDS, StrategyParams
from tests.helpers import ScriptedEngine, rng, snap

PARAMS = StrategyParams(momentum_threshold=0.005, position_fraction=0.5, leverage=2.0, vol_tolerance=0.05)


class AgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryBank(Path(self._tmp.name) / "memory_bank.json", durable=False)
        self.ids = iter(f"C-{i:03d}" for i in range(1, 100))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_agent(self, pnls: list[float], **kw) -> tuple[Agent, ScriptedEngine]:
        engine = ScriptedEngine(pnls)
        agent = Agent("A-001", self.memory, PARAMS, engine=engine, rng=rng(1), operating_cost=0.0, **kw)
        return agent, engine

    def run_steps(self, agent: Agent, n: int, start: int = 1) -> list[Agent]:
        children: list[Agent] = []
        for t in range(start, start + n):
            children += agent.step(snap(tick=t), id_factory=lambda: next(self.ids))
        return children


class TestSelfDestruct(AgentTestCase):
    def test_von_khoi_diem_chinh_xac_20(self) -> None:
        agent, _ = self.make_agent([])
        self.assertEqual(agent.capital, 20.0)
        self.assertEqual(INITIAL_CAPITAL, 20.0)

    def test_tu_huy_khi_von_bang_nguong(self) -> None:
        agent, _ = self.make_agent([], capital=DEATH_THRESHOLD)
        self.assertTrue(agent.check_self_destruct(tick=5))
        self.assertIs(agent.state, AgentState.TERMINATED)
        self.assertFalse(agent.is_alive)

    def test_song_sot_khi_von_tren_nguong(self) -> None:
        agent, _ = self.make_agent([], capital=0.51)
        self.assertFalse(agent.check_self_destruct(tick=5))
        self.assertIs(agent.state, AgentState.ACTIVE)

    def test_tu_huy_sau_lenh_lo_va_ghi_nguyen_nhan_vao_bo_nho(self) -> None:
        agent, _ = self.make_agent([-19.6])
        self.run_steps(agent, 2)  # tick 1 mở lệnh, tick 2 đóng lệnh lỗ → vốn 0.4
        self.assertIs(agent.state, AgentState.TERMINATED)
        self.assertAlmostEqual(agent.capital, 0.4, places=6)
        self.assertIsNotNone(agent.death_cause)

        data = self.memory.load()
        self.assertEqual(len(data["cai_chet"]), 1)
        death = data["cai_chet"][0]
        self.assertEqual(death["agent_id"], "A-001")
        self.assertIn("Vốn cạn", death["nguyen_nhan"])
        # Lệnh chí mạng trở thành quy tắc CẤM cho cả hệ sinh thái.
        rule_id = death["quy_tac_rut_ra"]
        rule = next(r for r in data["quy_tac"] if r["id"] == rule_id)
        self.assertEqual(rule["hanh_dong"], "CAM")
        self.assertEqual(rule["muc_do"], "chet_nguoi")

    def test_agent_da_chet_ngung_moi_hoat_dong(self) -> None:
        agent, engine = self.make_agent([-19.6])
        self.run_steps(agent, 2)
        opened, reads = len(engine.opened), self.memory.read_count
        self.assertEqual(self.run_steps(agent, 5, start=3), [])
        self.assertEqual(len(engine.opened), opened)
        self.assertEqual(self.memory.read_count, reads)
        self.assertIsNone(agent.position)

    def test_chi_phi_van_hanh_co_the_giet_agent(self) -> None:
        agent, _ = self.make_agent([], capital=0.55)
        agent.operating_cost = 0.1
        agent.step(snap(tick=1, momentum=0.0))
        self.assertIs(agent.state, AgentState.TERMINATED)
        self.assertIn("Chết đói", agent.death_cause or "")


class TestReplication(AgentTestCase):
    def test_nhan_ban_khi_tong_tai_san_dat_40(self) -> None:
        agent, _ = self.make_agent([20.0])
        children = self.run_steps(agent, 2)  # vốn 20 + 20 = 40 → đủ điều kiện
        self.assertEqual(len(children), 1)
        child = children[0]
        self.assertAlmostEqual(agent.capital, 40.0 - REPLICATION_COST)
        self.assertEqual(child.capital, 20.0)
        self.assertEqual(child.generation, agent.generation + 1)
        self.assertEqual(child.parent_id, "A-001")
        self.assertEqual(child.agent_id, "C-001")
        self.assertIs(child.memory, agent.memory)
        self.assertIs(child.state, AgentState.ACTIVE)

        births = self.memory.load()["pha_he"]
        self.assertEqual(len(births), 1)
        self.assertEqual(births[0]["cha"], "A-001")
        self.assertEqual(births[0]["con"], "C-001")

    def test_khong_nhan_ban_duoi_nguong(self) -> None:
        agent, _ = self.make_agent([19.99])
        self.assertEqual(self.run_steps(agent, 2), [])
        self.assertAlmostEqual(agent.capital, 39.99)

    def test_khong_nhan_ban_khi_he_sinh_thai_day(self) -> None:
        agent, _ = self.make_agent([25.0])
        agent.step(snap(tick=1), id_factory=lambda: "X")
        self.assertEqual(agent.step(snap(tick=2), id_factory=lambda: "X", can_replicate=False), [])
        self.assertAlmostEqual(agent.capital, 45.0)

    def test_gen_con_dot_bien_nhe_va_nam_trong_bien(self) -> None:
        agent, _ = self.make_agent([20.0])
        child = self.run_steps(agent, 2)[0]
        parent, kid = agent.params.to_dict(), child.params.to_dict()
        self.assertNotEqual(parent, kid)
        for name, (lo, hi) in PARAM_BOUNDS.items():
            self.assertGreaterEqual(kid[name], lo)
            self.assertLessEqual(kid[name], hi)
            # Đột biến "nhẹ": không lệch quá 50% so với gen cha.
            self.assertLess(abs(kid[name] - parent[name]) / parent[name], 0.5)

    def test_con_ke_thua_bo_nho_chung(self) -> None:
        agent, _ = self.make_agent([20.0])
        child = self.run_steps(agent, 2)[0]
        self.memory.post_mortem(
            agent_id="A-001",
            tick=3,
            context={"direction": "LONG", "volatility": 0.01, "momentum": 0.02, "leverage": 5.0, "fraction": 0.9},
            pnl=-10.0,
            capital_before=20.0,
        )
        view = child.memory.read_view(4)
        self.assertTrue(any(r["hanh_dong"] == "GIOI_HAN_DON_BAY" for r in view.rules))


class TestMutation(unittest.TestCase):
    def test_dot_bien_luon_trong_bien(self) -> None:
        r = rng(42)
        p = StrategyParams()
        for _ in range(500):
            p = p.mutate(r, rate=0.5)
            for name, (lo, hi) in PARAM_BOUNDS.items():
                self.assertTrue(lo <= getattr(p, name) <= hi, name)

    def test_ty_le_dot_bien_bang_khong_giu_nguyen_gen(self) -> None:
        p = StrategyParams()
        q = p.mutate(rng(1), rate=0.0, flip_prob=0.0)
        self.assertEqual(p, q)


if __name__ == "__main__":
    unittest.main()
