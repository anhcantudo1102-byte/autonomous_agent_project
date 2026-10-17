"""Kiểm thử chiến lược và bộ lọc quy tắc cấm kỵ."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from core.agent import Agent
from core.memory import MemoryBank, MemoryView
from core.strategy import (
    Direction,
    MarketSnapshot,
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


def csnap(score: float | None, momentum: float = 0.02, volatility: float = 0.01) -> MarketSnapshot:
    return replace(snap(momentum=momentum, volatility=volatility), confluence=score)


class TestConfluenceGene(unittest.TestCase):
    """Điểm hợp lưu đa khung làm bộ lọc/thiên hướng, mức tin tưởng là gen ``confluence_weight``."""

    def strat(self, weight: float | None, **kw: object) -> TradingStrategy:
        genes = {"momentum_threshold": 0.01, "position_fraction": 0.4, "vol_tolerance": 0.05, **kw}
        params = StrategyParams(confluence_weight=weight, **genes)
        return TradingStrategy(params, RuleFilter())

    def test_khong_co_gen_thi_bo_qua_diem(self) -> None:
        for score in (-7.5, 0.0, 7.5):
            o = self.strat(None).propose(csnap(score))
            self.assertEqual((o.direction, o.fraction, o.reason), (Direction.LONG, 0.4, "theo động lượng"))
        o = self.strat(1.0).propose(csnap(None))
        self.assertEqual((o.direction, o.fraction), (Direction.LONG, 0.4))

    def test_cung_chieu_tang_ty_trong(self) -> None:
        o = self.strat(0.5).propose(csnap(5.0))  # độ mạnh 0.5 → ×1.25
        self.assertIs(o.direction, Direction.LONG)
        self.assertAlmostEqual(o.fraction, 0.5)
        self.assertIn("đa khung ủng hộ", o.reason)
        self.assertLessEqual(self.strat(1.0, position_fraction=0.8).propose(csnap(7.5)).fraction, 0.9)

    def test_nguoc_chieu_manh_bi_phu_quyet(self) -> None:
        o = self.strat(1.0).propose(csnap(-5.0))
        self.assertIs(o.direction, Direction.HOLD)
        self.assertIn("phủ quyết", o.reason)

    def test_nguoc_chieu_yeu_giam_ty_trong(self) -> None:
        o = self.strat(0.2).propose(csnap(-5.0))  # độ mạnh 0.2 → ×0.8
        self.assertIs(o.direction, Direction.LONG)
        self.assertAlmostEqual(o.fraction, 0.32)

    def test_dong_luong_yeu_nhung_da_khung_manh_thi_vao_lenh(self) -> None:
        o = self.strat(0.8).propose(csnap(-5.0, momentum=0.0))  # độ mạnh 0.8 ≥ 0.6
        self.assertIs(o.direction, Direction.SHORT)
        self.assertAlmostEqual(o.fraction, 0.32)
        weak = self.strat(0.5).propose(csnap(-5.0, momentum=0.0))  # độ mạnh 0.5 < 0.6
        self.assertIs(weak.direction, Direction.HOLD)

    def test_bien_dong_cao_van_dung_ngoai(self) -> None:
        o = self.strat(1.0).propose(csnap(7.5, momentum=0.0, volatility=0.08))
        self.assertIs(o.direction, Direction.HOLD)

    def test_quy_tac_cam_van_chan_lenh_do_da_khung_mo(self) -> None:
        s = self.strat(1.0)
        view = MemoryView(tick=1, rules=(rule("QT-1", "CAM", LONG_HIGH_VOL),), data_version=0)
        d = s.decide(csnap(7.5, momentum=0.0, volatility=0.03), view)
        self.assertTrue(d.blocked)
        self.assertEqual(d.blocked_by, "QT-1")

    def test_gen_di_truyen_qua_dot_bien_va_lai_ghep(self) -> None:
        a = StrategyParams(confluence_weight=0.0)
        b = StrategyParams(confluence_weight=0.9)
        mutated = a.mutate(rng(3), rate=0.3)
        self.assertIsNotNone(mutated.confluence_weight)
        self.assertTrue(0.0 <= mutated.confluence_weight <= 1.0)
        child = StrategyParams.crossover([a, b], rng(1))
        self.assertIn(child.confluence_weight, (0.0, 0.9))
        self.assertEqual(len(b.genome()), len(StrategyParams().genome()) + 1)
        self.assertEqual(StrategyParams.from_dict(b.to_dict()), b)

    def test_khong_co_gen_thi_khong_doi_rng_va_tep_nho(self) -> None:
        plain = StrategyParams()
        self.assertNotIn("confluence_weight", plain.to_dict())
        r1, r2 = rng(5), rng(5)
        plain.mutate(r1)
        StrategyParams(momentum_threshold=plain.momentum_threshold).mutate(r2)
        self.assertEqual(r1.random(), r2.random())
        self.assertIsNone(StrategyParams.crossover([plain, plain], rng(1)).confluence_weight)


if __name__ == "__main__":
    unittest.main()
