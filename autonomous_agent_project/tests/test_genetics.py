"""Kiểm thử thuật toán di truyền: Fitness, lai ghép, đột biến thích ứng, đa dạng quần thể."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.agent import (
    DRAWDOWN_FLOOR,
    FITNESS_THRESHOLD,
    Agent,
    GeneticConfig,
    GeneticController,
)
from core.memory import MemoryBank
from core.strategy import (
    PARAM_BOUNDS,
    StrategyParams,
    adaptive_mutation_rate,
    diversify,
    nearest_distance,
    population_diversity,
)
from tests.helpers import ScriptedEngine, rng, snap

PARAMS = StrategyParams(momentum_threshold=0.005, position_fraction=0.5, leverage=2.0, vol_tolerance=0.05)
CTX = {"direction": "LONG", "volatility": 0.012, "momentum": 0.015, "leverage": 1.2, "fraction": 0.5}


class GeneticTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryBank(Path(self._tmp.name) / "memory_bank.json", durable=False)
        self.ids = iter(f"C-{i:03d}" for i in range(1, 100))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_agent(self, agent_id: str = "A-001", pnls: list[float] | None = None, params=PARAMS, **kw) -> Agent:
        return Agent(
            agent_id, self.memory, params, engine=ScriptedEngine(pnls or []), rng=rng(1), operating_cost=0.0, **kw
        )

    def play(self, agent: Agent, n: int, genetics: GeneticController | None = None) -> list[Agent]:
        children: list[Agent] = []
        for t in range(1, n + 1):
            children += agent.step(snap(tick=t), id_factory=lambda: next(self.ids), genetics=genetics)
        return children


class TestFitness(GeneticTestCase):
    def test_chua_giao_dich_thi_fitness_bang_khong(self) -> None:
        self.assertEqual(self.make_agent().fitness().score, 0.0)

    def test_cong_thuc_fitness(self) -> None:
        # Vốn 20 → 18 (lỗ 10%) → 24 (lãi 33%): Max DD 10%, lợi nhuận ròng +20%.
        agent = self.make_agent(pnls=[-2.0, 6.0])
        self.play(agent, 3)
        f = agent.fitness()
        self.assertAlmostEqual(f.max_drawdown, 0.1)
        self.assertAlmostEqual(f.net_return, 0.2)
        self.assertAlmostEqual(f.win_rate, 0.5)
        std = abs(6.0 / 18.0 - (-0.1)) / 2
        self.assertAlmostEqual(f.return_std, std)
        self.assertAlmostEqual(f.stability, 0.5 / (1 + 10 * std))
        self.assertAlmostEqual(f.score, (0.2 / 0.1) * f.stability)

    def test_drawdown_co_san_toi_thieu(self) -> None:
        agent = self.make_agent(pnls=[1.0])
        self.play(agent, 2)
        f = agent.fitness()
        self.assertEqual(f.max_drawdown, 0.0)
        self.assertAlmostEqual(f.score, (1.0 / 20.0) / DRAWDOWN_FLOOR * 1.0)

    def test_von_trao_cho_con_khong_bi_tinh_la_sut_giam(self) -> None:
        agent = self.make_agent(pnls=[20.0])
        self.assertEqual(len(self.play(agent, 2)), 1)
        self.assertAlmostEqual(agent.capital, 20.0)
        self.assertEqual(agent.stats.max_drawdown, 0.0)
        self.assertAlmostEqual(agent.fitness().net_return, 1.0)

    def test_du_40_nhung_fitness_thap_thi_khong_nhan_ban(self) -> None:
        # 20 → 4 (sụt 80%) → 40: đủ tiền nhưng rủi ro quá lớn và lệnh thất thường.
        agent = self.make_agent(pnls=[-16.0, 36.0])
        self.assertEqual(self.play(agent, 3), [])
        self.assertAlmostEqual(agent.capital, 40.0)
        self.assertLess(agent.fitness().score, FITNESS_THRESHOLD)
        self.assertGreaterEqual(agent.stats.replication_denied, 1)

    def test_ha_nguong_fitness_thi_duoc_nhan_ban(self) -> None:
        agent = self.make_agent(pnls=[-16.0, 36.0], fitness_threshold=0.0)
        self.assertEqual(len(self.play(agent, 3)), 1)


class TestGeneOperators(unittest.TestCase):
    def test_lai_ghep_moi_gen_lay_tu_mot_cha_me(self) -> None:
        a = StrategyParams(0.002, 0.2, 1.5, 0.01, False)
        b = StrategyParams(0.02, 0.8, 4.5, 0.07, True)
        r = rng(3)
        for _ in range(50):
            child = StrategyParams.crossover([a, b], r)
            for name in (*PARAM_BOUNDS, "contrarian"):
                self.assertIn(getattr(child, name), {getattr(a, name), getattr(b, name)})

    def test_lai_ghep_uu_tien_fitness_cao(self) -> None:
        a = StrategyParams(leverage=1.0)
        b = StrategyParams(leverage=5.0)
        r = rng(4)
        picks = [StrategyParams.crossover([a, b], r, weights=[9.0, 1.0]).leverage for _ in range(400)]
        self.assertGreater(picks.count(1.0), picks.count(5.0) * 3)

    def test_lai_ghep_can_it_nhat_mot_bo_gen(self) -> None:
        with self.assertRaises(ValueError):
            StrategyParams.crossover([], rng())

    def test_khoang_cach_gen(self) -> None:
        a = StrategyParams()
        self.assertEqual(a.distance(a), 0.0)
        flipped = StrategyParams(contrarian=True)
        self.assertAlmostEqual(a.distance(flipped), (1 / 5) ** 0.5)
        self.assertEqual(nearest_distance(a, []), 1.0)
        self.assertEqual(population_diversity([a]), 0.0)
        self.assertGreater(population_diversity([a, flipped]), 0.0)

    def test_dot_bien_thich_ung_tang_theo_ap_luc_that_bai(self) -> None:
        kw = dict(base_rate=0.1, min_rate=0.03, max_rate=0.35, high_pressure=0.4)
        self.assertAlmostEqual(adaptive_mutation_rate(0.0, **kw), 0.03)
        self.assertAlmostEqual(adaptive_mutation_rate(0.2, **kw), 0.1)
        self.assertAlmostEqual(adaptive_mutation_rate(0.4, **kw), 0.35)
        self.assertAlmostEqual(adaptive_mutation_rate(5.0, **kw), 0.35)
        rates = [adaptive_mutation_rate(p / 100, **kw) for p in range(0, 50)]
        self.assertEqual(rates, sorted(rates))

    def test_diversify_tach_khoi_quan_the_trung_lap(self) -> None:
        clone = StrategyParams()
        res = diversify(clone, [clone, clone], rng(5), min_distance=0.1, rate=0.05)
        self.assertGreaterEqual(res.nearest, 0.1)
        self.assertGreaterEqual(nearest_distance(res.params, [clone]), 0.1)
        for name, (lo, hi) in PARAM_BOUNDS.items():
            self.assertTrue(lo <= getattr(res.params, name) <= hi)

    def test_diversify_dung_gen_nhap_cu_khi_khong_the_dot_bien(self) -> None:
        clone = StrategyParams()
        res = diversify(clone, [clone], rng(6), min_distance=0.2, rate=0.0, max_attempts=0)
        self.assertTrue(res.immigrant)
        self.assertNotEqual(res.params, clone)


class TestGeneticController(GeneticTestCase):
    def controller(self, agents: list[Agent], **cfg) -> GeneticController:
        return GeneticController(self.memory, lambda: agents, rng(7), GeneticConfig(**cfg))

    def test_bien_do_dot_bien_tang_khi_that_bai_nhieu(self) -> None:
        agents = [self.make_agent("A-001")]
        gc = self.controller(agents, failure_window=10)
        calm = gc.mutation_rate(tick=10)
        for t in range(1, 11):
            for _ in range(4):
                self.memory.post_mortem(agent_id="A-001", tick=t, context=CTX, pnl=-0.1, capital_before=20.0)
        self.assertEqual(self.memory.recent_failures(10, 10), 40)
        self.assertEqual(self.memory.recent_failures(10, 5), 20)
        stressed = gc.mutation_rate(tick=10)
        self.assertAlmostEqual(calm, gc.config.min_mutation_rate)
        self.assertAlmostEqual(stressed, gc.config.max_mutation_rate)
        # Hết cửa sổ thất bại → biên độ thu nhỏ lại để tinh chỉnh.
        self.assertAlmostEqual(gc.mutation_rate(tick=100), gc.config.min_mutation_rate)

    def test_tinh_hoa_va_lai_ghep(self) -> None:
        strong = self.make_agent("A-001", params=StrategyParams(leverage=1.0, contrarian=False))
        mid = self.make_agent("A-002", params=StrategyParams(leverage=5.0, contrarian=True))
        weak = self.make_agent("A-003", params=StrategyParams(leverage=3.0, position_fraction=0.1))
        for ag, pnls in ((strong, [10.0, 9.0]), (mid, [8.0, 0.5]), (weak, [-5.0, 1.0])):
            ag.engine = ScriptedEngine(pnls)
            self.play(ag, 3)
        gc = self.controller([strong, mid, weak], crossover_prob=1.0, elite_size=2)
        elite = gc.elite(mid)
        self.assertEqual([a.agent_id for a in elite], ["A-001", "A-002"])
        self.assertNotIn(weak, elite)

        breed = gc.breed(mid, tick=4)
        self.assertEqual(breed.method, "lai_ghep")
        self.assertEqual(set(breed.parent_ids), {"A-001", "A-002"})
        self.assertGreaterEqual(nearest_distance(breed.params, [a.params for a in (strong, mid, weak)]), 0.05)

    def test_khong_du_tinh_hoa_thi_tu_dot_bien(self) -> None:
        solo = self.make_agent("A-001", pnls=[20.0])
        self.play(solo, 2)
        breed = self.controller([solo], crossover_prob=1.0).breed(solo, tick=3)
        self.assertEqual(breed.method, "tu_dot_bien")
        self.assertEqual(breed.parent_ids, ("A-001",))

    def test_nhan_ban_qua_bo_dieu_khien_ghi_pha_he(self) -> None:
        agents: list[Agent] = []
        parent = self.make_agent("A-001", pnls=[20.0])
        agents.append(parent)
        gc = self.controller(agents)
        children = self.play(parent, 2, genetics=gc)
        self.assertEqual(len(children), 1)
        birth = self.memory.load()["pha_he"][0]
        self.assertIn(birth["phuong_thuc"], {"lai_ghep", "tu_dot_bien"})
        self.assertIn("bien_do_dot_bien", birth)
        self.assertGreater(birth["fitness_cha"]["score"], FITNESS_THRESHOLD)
        self.assertGreaterEqual(nearest_distance(children[0].params, [parent.params]), gc.config.min_genome_distance)

    def test_ra_soat_dong_huyet_thong(self) -> None:
        twin_params = StrategyParams()
        strong = self.make_agent("A-001", pnls=[5.0], params=twin_params)
        weak = self.make_agent("A-002", pnls=[-1.0], params=twin_params)
        other = self.make_agent("A-003", params=StrategyParams(contrarian=True, leverage=4.5))
        for ag in (strong, weak):
            self.play(ag, 2)
        gc = self.controller([strong, weak, other])
        shifts = gc.enforce_diversity(tick=3)
        self.assertEqual([(s.agent_id, s.twin_id) for s in shifts], [("A-002", "A-001")])
        self.assertEqual(strong.params, twin_params)
        self.assertGreaterEqual(weak.params.distance(strong.params), gc.config.min_genome_distance)
        self.assertIs(weak.strategy.params, weak.params)
        self.assertEqual(weak.stats.niche_shifts, 1)
        self.assertEqual(gc.enforce_diversity(tick=4), [])


if __name__ == "__main__":
    unittest.main()
