"""Kiểm thử chiến lược và bộ lọc quy tắc cấm kỵ."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.agent import Agent
from core.memory import MemoryBank, MemoryView
from core.strategy import (
    Direction,
    MemoryNotConsultedError,
    Order,
    RuleFilter,
    StrategyParams,
    TradingStrategy,
)
from tests.helpers import ScriptedEngine, rng, snap


def rule(rid: str, action: str, conds: list[dict], params: dict | None = None, active: bool = True) -> dict:
    return {
        "id": rid,
        "mo_ta": f"quy tắc thử {rid}",
        "dieu_kien": conds,
        "hanh_dong": action,
        "tham_so": params or {},
        "hieu_luc": active,
    }


LONG_HIGH_VOL = [
    {"truong": "direction", "toan_tu": "==", "gia_tri": "LONG"},
    {"truong": "volatility", "toan_tu": ">=", "gia_tri": 0.02},
]


class TestRuleFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.f = RuleFilter()
        self.order = Order(Direction.LONG, 0.6, 3.0)

    def test_quy_tac_cam_chan_lenh_khop(self) -> None:
        d = self.f.apply(self.order, snap(volatility=0.03), [rule("QT-1", "CAM", LONG_HIGH_VOL)])
        self.assertTrue(d.blocked)
        self.assertEqual(d.blocked_by, "QT-1")
        self.assertIs(d.order.direction, Direction.HOLD)

    def test_quy_tac_khong_khop_thi_cho_qua(self) -> None:
        rules = [rule("QT-1", "CAM", LONG_HIGH_VOL)]
        self.assertFalse(self.f.apply(self.order, snap(volatility=0.01), rules).blocked)
        short = Order(Direction.SHORT, 0.6, 3.0)
        self.assertFalse(self.f.apply(short, snap(volatility=0.03), rules).blocked)

    def test_giam_quy_mo(self) -> None:
        d = self.f.apply(self.order, snap(volatility=0.03), [rule("QT-2", "GIAM_QUY_MO", LONG_HIGH_VOL, {"he_so": 0.5})])
        self.assertFalse(d.blocked)
        self.assertAlmostEqual(d.order.fraction, 0.3)
        self.assertEqual(d.applied_rules, ("QT-2",))

    def test_gioi_han_don_bay(self) -> None:
        conds = [
            {"truong": "volatility", "toan_tu": ">=", "gia_tri": 0.02},
            {"truong": "leverage", "toan_tu": ">", "gia_tri": 1.5},
        ]
        d = self.f.apply(self.order, snap(volatility=0.03), [rule("QT-3", "GIOI_HAN_DON_BAY", conds, {"don_bay_toi_da": 1.5})])
        self.assertAlmostEqual(d.order.leverage, 1.5)
        self.assertAlmostEqual(d.order.fraction, 0.6)

    def test_cam_duoc_uu_tien_truoc_dieu_chinh(self) -> None:
        rules = [
            rule("QT-2", "GIAM_QUY_MO", LONG_HIGH_VOL, {"he_so": 0.5}),
            rule("QT-1", "CAM", LONG_HIGH_VOL),
        ]
        self.assertEqual(self.f.apply(self.order, snap(volatility=0.03), rules).blocked_by, "QT-1")

    def test_quy_tac_het_hieu_luc_bi_bo_qua(self) -> None:
        d = self.f.apply(self.order, snap(volatility=0.03), [rule("QT-1", "CAM", LONG_HIGH_VOL, active=False)])
        self.assertFalse(d.blocked)

    def test_dieu_kien_sai_dinh_dang_khong_khop(self) -> None:
        bad = [
            rule("QT-4", "CAM", [{"truong": "khong_ton_tai", "toan_tu": "==", "gia_tri": 1}]),
            rule("QT-5", "CAM", [{"truong": "volatility", "toan_tu": "~=", "gia_tri": 0.0}]),
            rule("QT-6", "CAM", []),
        ]
        self.assertFalse(self.f.apply(self.order, snap(volatility=0.03), bad).blocked)

    def test_lenh_hold_khong_qua_bo_loc(self) -> None:
        hold = Order(Direction.HOLD, 0.0, 1.0)
        d = self.f.apply(hold, snap(volatility=0.03), [rule("QT-1", "CAM", [])])
        self.assertFalse(d.blocked)


class TestStrategyMemoryGuard(unittest.TestCase):
    def test_tu_choi_giao_dich_khi_chua_doc_bo_nho(self) -> None:
        s = TradingStrategy(StrategyParams())
        with self.assertRaises(MemoryNotConsultedError):
            s.decide(snap(tick=3), None)

    def test_tu_choi_anh_chup_bo_nho_cu(self) -> None:
        s = TradingStrategy(StrategyParams())
        with self.assertRaises(MemoryNotConsultedError):
            s.decide(snap(tick=3), MemoryView(tick=2, rules=(), data_version=0))

    def test_chien_luoc_dong_luong_va_dao_chieu(self) -> None:
        view = MemoryView(tick=1, rules=(), data_version=0)
        p = StrategyParams(momentum_threshold=0.005, vol_tolerance=0.05)
        self.assertIs(TradingStrategy(p).decide(snap(momentum=0.02), view).order.direction, Direction.LONG)
        c = StrategyParams(momentum_threshold=0.005, vol_tolerance=0.05, contrarian=True)
        self.assertIs(TradingStrategy(c).decide(snap(momentum=0.02), view).order.direction, Direction.SHORT)
        self.assertIs(TradingStrategy(p).decide(snap(momentum=0.001), view).order.direction, Direction.HOLD)


class TestAgentObeysMemory(unittest.TestCase):
    """Agent phải đọc memory_bank.json trước khi giao dịch và tôn trọng quy tắc trong đó."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "memory_bank.json"
        self.memory = MemoryBank(self.path, durable=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_quy_tac_tren_dia_chan_lenh_cua_agent(self) -> None:
        # Một tiến trình khác (MemoryBank thứ hai) ghi quy tắc CẤM vào cùng tệp.
        other = MemoryBank(self.path, durable=False)
        with other._transaction() as data:
            data["quy_tac"].append(rule("QT-9", "CAM", LONG_HIGH_VOL))

        engine = ScriptedEngine([])
        params = StrategyParams(momentum_threshold=0.005, vol_tolerance=0.05)
        agent = Agent("A-001", self.memory, params, engine=engine, rng=rng(), operating_cost=0.0)
        agent.step(snap(tick=1, momentum=0.02, volatility=0.03))
        self.assertEqual(engine.opened, [])
        self.assertEqual(agent.stats.blocked, 1)
        self.assertEqual(self.memory.read_count, 1)

        agent.step(snap(tick=2, momentum=0.02, volatility=0.01))
        self.assertEqual(len(engine.opened), 1)
        self.assertEqual(self.memory.read_count, 2)


if __name__ == "__main__":
    unittest.main()
