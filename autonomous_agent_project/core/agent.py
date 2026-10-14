"""Vòng đời của một Autonomous Adaptive Agent và bộ điều khiển di truyền của quần thể.

Mỗi tick, agent:

1. Đóng vị thế đang mở → nếu lỗ, chạy Post-Mortem và cập nhật bộ nhớ chung.
2. Trả chi phí vận hành (tiền "điện" để tồn tại).
3. Kiểm tra sinh tồn: vốn ≤ $0.5 → TỰ HỦY (TERMINATED), ghi nguyên nhân cái chết.
4. Kiểm tra sinh sản: tổng tài sản ≥ $40 **và** điểm Fitness đạt ngưỡng → trích $20
   sinh agent con. Gen của con do ``GeneticController`` tạo ra (lai ghép từ nhóm tinh
   hoa + đột biến thích ứng + kiểm soát đa dạng), hoặc tự đột biến nếu chạy đơn lẻ.
5. Đọc lại ``memory_bank.json`` (bắt buộc) rồi mới đề xuất và lọc lệnh mới.

Điểm Fitness = (lợi nhuận ròng / Max Drawdown) × hệ số ổn định lệnh, trong đó:

* lợi nhuận ròng = (vốn hiện tại + vốn đã trao cho con − vốn khởi điểm) / vốn khởi điểm;
* Max Drawdown  = mức sụt giảm lớn nhất từ đỉnh của đường tài sản (không thấp hơn
  ``DRAWDOWN_FLOOR`` để tránh chia cho 0);
* hệ số ổn định = tỷ lệ thắng / (1 + ``STABILITY_PENALTY`` × độ lệch chuẩn lợi suất mỗi lệnh).
"""

from __future__ import annotations

import logging
import random
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

from .memory import MemoryBank, describe_context
from .strategy import (
    DiversityResult,
    ExecutionEngine,
    MarketSnapshot,
    Position,
    RuleFilter,
    SimulatedExecutionEngine,
    StrategyParams,
    TradingStrategy,
    adaptive_mutation_rate,
    clamp_capital,
    diversify,
    population_diversity,
)

INITIAL_CAPITAL = 20.0
DEATH_THRESHOLD = 0.5
REPLICATION_THRESHOLD = 40.0
REPLICATION_COST = 20.0

FITNESS_THRESHOLD = 1.5
DRAWDOWN_FLOOR = 0.05
STABILITY_PENALTY = 10.0
STABILITY_WINDOW = 50


class AgentState(str, Enum):
    ACTIVE = "ACTIVE"
    TERMINATED = "TERMINATED"


@dataclass
class AgentStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    blocked: int = 0
    adjusted: int = 0
    holds: int = 0
    realized_pnl: float = 0.0
    fees: float = 0.0
    operating_cost: float = 0.0
    children: int = 0
    loss_streak: int = 0
    max_capital: float = INITIAL_CAPITAL
    spawned_capital: float = 0.0
    peak_equity: float = INITIAL_CAPITAL
    max_drawdown: float = 0.0
    replication_denied: int = 0
    niche_shifts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class FitnessReport:
    """Điểm thích nghi của một agent và các thành phần tạo nên nó."""

    score: float
    net_return: float
    max_drawdown: float
    stability: float
    win_rate: float
    return_std: float

    def describe(self) -> str:
        return (
            f"lợi nhuận ròng {self.net_return:+.0%}, Max DD {self.max_drawdown:.0%}, "
            f"ổn định {self.stability:.2f} (thắng {self.win_rate:.0%}, σ lệnh {self.return_std:.1%})"
        )

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 6) for k, v in self.__dict__.items()}


class Agent:
    """Một thực thể kinh tế tự trị: tự kiếm tiền, tự học, tự hủy, tự nhân bản."""

    def __init__(
        self,
        agent_id: str,
        memory: MemoryBank,
        params: StrategyParams | None = None,
        *,
        capital: float = INITIAL_CAPITAL,
        generation: int = 0,
        parent_id: str | None = None,
        engine: ExecutionEngine | None = None,
        rng: random.Random | None = None,
        operating_cost: float = 0.02,
        birth_tick: int = 0,
        mutation_rate: float = 0.1,
        fitness_threshold: float = FITNESS_THRESHOLD,
    ) -> None:
        self.agent_id = agent_id
        self.memory = memory
        self.rng = rng or random.Random()
        self.params = params or StrategyParams()
        self.strategy = TradingStrategy(self.params, RuleFilter())
        self.engine = engine or SimulatedExecutionEngine(self.rng)
        self.capital = float(capital)
        self.initial_capital = float(capital)
        self.generation = generation
        self.parent_id = parent_id
        self.operating_cost = operating_cost
        self.birth_tick = birth_tick
        self.mutation_rate = mutation_rate
        self.fitness_threshold = fitness_threshold
        self.state = AgentState.ACTIVE
        self.position: Position | None = None
        self.stats = AgentStats(max_capital=self.capital, peak_equity=self.capital)
        self.death_cause: str | None = None
        self.death_tick: int | None = None
        self._last_loss_context: dict[str, Any] | None = None
        self._trade_returns: deque[float] = deque(maxlen=STABILITY_WINDOW)
        self._fitness_gate_closed = False
        self.log = logging.getLogger(f"agent.{agent_id}")

    # ---------------------------------------------------------- Trạng thái
    @property
    def is_alive(self) -> bool:
        return self.state is AgentState.ACTIVE

    def total_assets(self, snapshot: MarketSnapshot | None = None) -> float:
        """Tổng tài sản = tiền mặt + lãi/lỗ chưa thực hiện của vị thế đang mở."""
        if self.position is None or snapshot is None:
            return self.capital
        return clamp_capital(self.capital + self.position.unrealized_pnl(snapshot.price))

    @property
    def equity(self) -> float:
        """Đường tài sản dùng cho Fitness: vốn trao cho con không bị tính là thua lỗ."""
        return self.capital + self.stats.spawned_capital

    def _update_equity(self) -> None:
        s = self.stats
        s.peak_equity = max(s.peak_equity, self.equity)
        if s.peak_equity > 0:
            s.max_drawdown = max(s.max_drawdown, (s.peak_equity - self.equity) / s.peak_equity)

    def fitness(self) -> FitnessReport:
        """Hàm thích nghi: (lợi nhuận ròng / Max Drawdown) × hệ số ổn định lệnh."""
        s = self.stats
        net_return = (self.equity - self.initial_capital) / self.initial_capital if self.initial_capital else 0.0
        drawdown = max(s.max_drawdown, DRAWDOWN_FLOOR)
        win_rate = s.wins / s.trades if s.trades else 0.0
        std = statistics.pstdev(self._trade_returns) if len(self._trade_returns) >= 2 else 0.0
        stability = win_rate / (1.0 + STABILITY_PENALTY * std)
        return FitnessReport(
            score=(net_return / drawdown) * stability,
            net_return=net_return,
            max_drawdown=s.max_drawdown,
            stability=stability,
            win_rate=win_rate,
            return_std=std,
        )

    def adopt_params(self, params: StrategyParams) -> None:
        """Thay bộ gen chiến lược đang dùng (phân hoá ngách khi trùng lặp gen)."""
        self.params = params
        self.strategy.params = params

    # ------------------------------------------------------------ Vòng đời
    def step(
        self,
        snapshot: MarketSnapshot,
        *,
        id_factory: Callable[[], str] | None = None,
        can_replicate: bool = True,
        genetics: "GeneticController | None" = None,
    ) -> list["Agent"]:
        """Chạy một nhịp sống. Trả về danh sách agent con vừa được sinh ra."""
        if not self.is_alive:
            return []

        self._settle_position(snapshot)

        self.capital = clamp_capital(self.capital - self.operating_cost)
        self.stats.operating_cost += self.operating_cost
        self._update_equity()

        if self.check_self_destruct(snapshot.tick):
            return []

        children: list[Agent] = []
        if can_replicate and id_factory is not None:
            child = self.try_replicate(snapshot, id_factory, genetics)
            if child is not None:
                children.append(child)

        self._act(snapshot)
        self.stats.max_capital = max(self.stats.max_capital, self.capital)
        return children

    def _settle_position(self, snapshot: MarketSnapshot) -> None:
        if self.position is None:
            return
        pos = self.position
        self.position = None
        result = self.engine.close_position(pos, snapshot)
        pnl = max(result.pnl, -self.capital)  # bị thanh lý: không lỗ quá số vốn hiện có
        capital_before = self.capital
        self.capital = clamp_capital(self.capital + pnl)
        self.stats.trades += 1
        self.stats.realized_pnl += pnl
        self.stats.fees += result.fees
        if capital_before > 0:
            self._trade_returns.append(pnl / capital_before)
        self._update_equity()
        if pnl >= 0:
            self.stats.wins += 1
            self.stats.loss_streak = 0
            self.memory.record_win(pos.context, pnl)
            self.log.debug("✅ Chốt lãi $%.3f | vốn $%.2f", pnl, self.capital)
            return

        self.stats.losses += 1
        self.stats.loss_streak += 1
        self._last_loss_context = pos.context
        pm = self.memory.post_mortem(
            agent_id=self.agent_id,
            tick=snapshot.tick,
            context=pos.context,
            pnl=pnl,
            capital_before=capital_before,
        )
        self.log.debug("📉 Post-Mortem: %s", pm.lesson)
        for event, rule in pm.rule_events:
            verb = {"tao_moi": "đúc kết quy tắc mới", "nang_cap": "nâng cấp quy tắc", "cung_co": "củng cố quy tắc"}[event]
            level = logging.INFO if event != "cung_co" else logging.DEBUG
            self.log.log(level, "🧠 %s %s [%s]: %s", verb.capitalize(), rule["id"], rule["hanh_dong"], rule["mo_ta"])

    # ------------------------------------------------------------ Tự hủy
    def check_self_destruct(self, tick: int) -> bool:
        """Vốn ≤ $0.5 → TERMINATED. Trả về True nếu agent vừa tự hủy."""
        if not self.is_alive:
            return True
        if self.capital > DEATH_THRESHOLD:
            return False
        self.terminate(tick, self._diagnose_death(tick))
        return True

    def _diagnose_death(self, tick: int) -> str:
        s = self.stats
        parts = [f"Vốn cạn còn ${self.capital:.2f} (≤ ${DEATH_THRESHOLD:.2f}) tại tick {tick}."]
        if s.trades == 0 or (self._last_loss_context is None and s.operating_cost >= s.max_capital * 0.5):
            parts.append(
                "Chết đói: chi phí vận hành bào mòn vốn vì gần như không giao dịch "
                f"(giữ lệnh {s.holds} lần, bị chặn {s.blocked} lần)."
            )
        else:
            parts.append(
                f"{s.losses}/{s.trades} lệnh thua, chuỗi thua cuối {s.loss_streak} lệnh, "
                f"lãi/lỗ thực hiện ${s.realized_pnl:.2f}, phí ${s.fees:.2f}, Max DD {s.max_drawdown:.0%}."
            )
            if self._last_loss_context is not None:
                parts.append(f"Lệnh chí mạng: {describe_context(self._last_loss_context)}.")
        return " ".join(parts)

    def terminate(self, tick: int, cause: str) -> None:
        """Dừng mọi hoạt động, ghi nguyên nhân cái chết vào bộ nhớ chung."""
        if not self.is_alive:
            return
        self.state = AgentState.TERMINATED
        self.position = None
        self.death_cause = cause
        self.death_tick = tick
        rule = self.memory.record_death(
            agent_id=self.agent_id,
            tick=tick,
            cause=cause,
            final_capital=self.capital,
            generation=self.generation,
            parent_id=self.parent_id,
            params=self.params.to_dict(),
            stats=self.stats.to_dict(),
            fatal_context=self._last_loss_context,
        )
        self.log.warning("💀 TERMINATED — %s", cause)
        if rule is not None:
            self.log.warning("🪦 Di chúc để lại quy tắc %s: %s", rule["id"], rule["mo_ta"])

    # -------------------------------------------------------- Nhân bản
    def should_replicate(self, snapshot: MarketSnapshot | None = None) -> bool:
        """Đủ $40 tổng tài sản **và** Fitness đạt ngưỡng tối thiểu."""
        if not self.is_alive or self.total_assets(snapshot) < REPLICATION_THRESHOLD:
            self._fitness_gate_closed = False
            return False
        report = self.fitness()
        if report.score >= self.fitness_threshold:
            self._fitness_gate_closed = False
            return True
        self.stats.replication_denied += 1
        if not self._fitness_gate_closed:
            self._fitness_gate_closed = True
            self.log.info(
                "⛔ Đủ $%.2f nhưng Fitness %.2f < ngưỡng %.2f (%s) — chưa được nhân bản",
                self.total_assets(snapshot),
                report.score,
                self.fitness_threshold,
                report.describe(),
            )
        return False

    def try_replicate(
        self,
        snapshot: MarketSnapshot,
        id_factory: Callable[[], str],
        genetics: "GeneticController | None" = None,
    ) -> "Agent | None":
        if not self.should_replicate(snapshot):
            return None
        if genetics is None:
            return self.replicate(id_factory(), snapshot.tick)
        breed = genetics.breed(self, snapshot.tick)
        return self.replicate(id_factory(), snapshot.tick, breed=breed)

    def replicate(self, child_id: str, tick: int, *, breed: "BreedResult | None" = None) -> "Agent":
        """Trích $20 sinh agent con, thừa hưởng toàn bộ memory_bank.

        ``breed`` (từ ``GeneticController``) cung cấp gen lai ghép + đột biến thích ứng;
        nếu không có, con nhận gen của chính agent này với đột biến nhẹ.
        """
        fitness = self.fitness()
        self.capital -= REPLICATION_COST
        self.stats.spawned_capital += REPLICATION_COST
        self.stats.children += 1
        child_params = breed.params if breed else self.params.mutate(self.rng, rate=self.mutation_rate)
        child = Agent(
            child_id,
            self.memory,
            child_params,
            capital=REPLICATION_COST,
            generation=self.generation + 1,
            parent_id=self.agent_id,
            engine=self.engine,
            rng=random.Random(self.rng.random()),
            operating_cost=self.operating_cost,
            birth_tick=tick,
            mutation_rate=breed.mutation_rate if breed else self.mutation_rate,
            fitness_threshold=self.fitness_threshold,
        )
        extra: dict[str, Any] = {"fitness_cha": fitness.to_dict()}
        if breed:
            extra.update(breed.to_record())
        self.memory.record_birth(
            parent_id=self.agent_id,
            child_id=child_id,
            tick=tick,
            generation=child.generation,
            parent_params=self.params.to_dict(),
            child_params=child_params.to_dict(),
            extra=extra,
        )
        how = breed.describe() if breed else f"tự đột biến (biên độ {self.mutation_rate:.2f})"
        self.log.info(
            "🧬 Tự nhân bản → %s (thế hệ %d), Fitness cha %.2f. Cha còn $%.2f, con nhận $%.2f. "
            "Gen: %s. Thay đổi so với cha: %s",
            child_id,
            child.generation,
            fitness.score,
            self.capital,
            child.capital,
            how,
            _diff_params(self.params, child_params),
        )
        return child

    # -------------------------------------------------------- Hành động
    def _act(self, snapshot: MarketSnapshot) -> None:
        # Bắt buộc: đọc lại bộ nhớ chung từ đĩa để né "vết xe đổ" trước mọi lệnh.
        view = self.memory.read_view(snapshot.tick)
        decision = self.strategy.decide(snapshot, view)
        if decision.blocked:
            self.stats.blocked += 1
            self.log.debug("🛑 %s", decision.order.reason)
            return
        order = decision.order
        if not order.is_trade:
            self.stats.holds += 1
            return
        if decision.applied_rules:
            self.stats.adjusted += 1
            self.log.debug(
                "⚖️  Điều chỉnh lệnh theo %s: tỷ trọng %.0f%%→%.0f%%, đòn bẩy %.1fx→%.1fx",
                ", ".join(decision.applied_rules),
                decision.original.fraction * 100,
                order.fraction * 100,
                decision.original.leverage,
                order.leverage,
            )
        context = RuleFilter.build_context(order, snapshot)
        self.position = self.engine.open_position(order, self.capital, snapshot, context)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "the_he": self.generation,
            "cha": self.parent_id,
            "trang_thai": self.state.value,
            "von": round(self.capital, 4),
            "fitness": self.fitness().to_dict(),
            "tham_so": self.params.to_dict(),
            "thong_ke": self.stats.to_dict(),
            "nguyen_nhan_chet": self.death_cause,
        }


# ======================================================================
# Bộ điều khiển di truyền của quần thể
# ======================================================================
@dataclass(frozen=True)
class GeneticConfig:
    elite_size: int = 3
    crossover_prob: float = 0.7
    base_mutation_rate: float = 0.1
    min_mutation_rate: float = 0.03
    max_mutation_rate: float = 0.35
    failure_window: int = 30
    high_failure_pressure: float = 0.4
    min_genome_distance: float = 0.05
    max_diversity_attempts: int = 6


@dataclass(frozen=True)
class BreedResult:
    """Kết quả tạo gen cho agent con."""

    params: StrategyParams
    method: str  # "lai_ghep" | "tu_dot_bien"
    parent_ids: tuple[str, ...]
    mutation_rate: float
    failure_pressure: float
    diversity: DiversityResult

    def describe(self) -> str:
        src = "lai ghép " + "×".join(self.parent_ids) if self.method == "lai_ghep" else "tự đột biến"
        parts = [f"{src}, biên độ đột biến {self.mutation_rate:.2f} (áp lực thất bại {self.failure_pressure:.2f})"]
        if self.diversity.immigrant:
            parts.append("bơm gen nhập cư để tránh trùng lặp")
        elif self.diversity.attempts:
            parts.append(f"đột biến thêm {self.diversity.attempts} lần để tránh trùng lặp")
        parts.append(f"khoảng cách gen gần nhất {self.diversity.nearest:.3f}")
        return ", ".join(parts)

    def to_record(self) -> dict[str, Any]:
        return {
            "phuong_thuc": self.method,
            "nguon_gen": list(self.parent_ids),
            "bien_do_dot_bien": round(self.mutation_rate, 4),
            "ap_luc_that_bai": round(self.failure_pressure, 4),
            "khoang_cach_gen_gan_nhat": round(self.diversity.nearest, 4),
            "so_lan_da_dang_hoa": self.diversity.attempts,
            "gen_nhap_cu": self.diversity.immigrant,
        }


@dataclass(frozen=True)
class NicheShift:
    """Một agent bị buộc phân hoá ngách vì trùng gen với agent mạnh hơn."""

    agent_id: str
    twin_id: str
    distance_before: float
    result: DiversityResult


class GeneticController:
    """Điều phối tiến hoá: chọn lọc tinh hoa, lai ghép, đột biến thích ứng, giữ đa dạng."""

    def __init__(
        self,
        memory: MemoryBank,
        population: Callable[[], Iterable[Agent]],
        rng: random.Random,
        config: GeneticConfig | None = None,
    ) -> None:
        self.memory = memory
        self.population = population
        self.rng = rng
        self.config = config or GeneticConfig()
        self.log = logging.getLogger("di_truyen")

    def active_agents(self) -> list[Agent]:
        return [a for a in self.population() if a.is_alive]

    def failure_pressure(self, tick: int) -> float:
        """Số lệnh lỗ trong memory_bank trên mỗi agent mỗi tick, trong cửa sổ gần nhất."""
        c = self.config
        failures = self.memory.recent_failures(tick, c.failure_window)
        return failures / (c.failure_window * max(1, len(self.active_agents())))

    def mutation_rate(self, tick: int) -> float:
        c = self.config
        return adaptive_mutation_rate(
            self.failure_pressure(tick),
            base_rate=c.base_mutation_rate,
            min_rate=c.min_mutation_rate,
            max_rate=c.max_mutation_rate,
            high_pressure=c.high_failure_pressure,
        )

    def elite(self, parent: Agent) -> list[Agent]:
        """Top agent có Fitness cao nhất (đạt ngưỡng); luôn bao gồm agent đang sinh sản."""
        scored = [(a.fitness().score, a) for a in self.active_agents()]
        qualified = [(s, a) for s, a in scored if s >= a.fitness_threshold and s > 0]
        qualified.sort(key=lambda sa: sa[0], reverse=True)
        top = [a for _, a in qualified[: self.config.elite_size]]
        if parent not in top:
            top = top[: max(0, self.config.elite_size - 1)] + [parent]
        return top

    def breed(self, parent: Agent, tick: int) -> BreedResult:
        c = self.config
        pressure = self.failure_pressure(tick)
        rate = adaptive_mutation_rate(
            pressure,
            base_rate=c.base_mutation_rate,
            min_rate=c.min_mutation_rate,
            max_rate=c.max_mutation_rate,
            high_pressure=c.high_failure_pressure,
        )
        elite = self.elite(parent)
        if len(elite) >= 2 and self.rng.random() < c.crossover_prob:
            weights = [max(a.fitness().score, 1e-6) for a in elite]
            base = StrategyParams.crossover([a.params for a in elite], self.rng, weights)
            method, ids = "lai_ghep", tuple(a.agent_id for a in elite)
        else:
            base, method, ids = parent.params, "tu_dot_bien", (parent.agent_id,)
        candidate = base.mutate(self.rng, rate=rate)
        others = [a.params for a in self.active_agents()]
        div = diversify(
            candidate,
            others,
            self.rng,
            min_distance=c.min_genome_distance,
            rate=rate,
            max_attempts=c.max_diversity_attempts,
        )
        return BreedResult(div.params, method, ids, rate, pressure, div)

    def enforce_diversity(self, tick: int) -> list[NicheShift]:
        """Rà soát trùng lặp gen: agent yếu hơn trong cặp "đồng huyết thống" phải đổi ngách."""
        c = self.config
        ranked = sorted(self.active_agents(), key=lambda a: a.fitness().score, reverse=True)
        kept: list[Agent] = []
        shifts: list[NicheShift] = []
        rate: float | None = None
        for agent in ranked:
            if kept:
                twin = min(kept, key=lambda k: agent.params.distance(k.params))
                dist = agent.params.distance(twin.params)
                if dist < c.min_genome_distance:
                    rate = self.mutation_rate(tick) if rate is None else rate
                    res = diversify(
                        agent.params,
                        [k.params for k in kept],
                        self.rng,
                        min_distance=c.min_genome_distance,
                        rate=rate,
                        max_attempts=c.max_diversity_attempts,
                    )
                    agent.adopt_params(res.params)
                    agent.stats.niche_shifts += 1
                    shifts.append(NicheShift(agent.agent_id, twin.agent_id, dist, res))
                    self.log.info(
                        "🧪 Phân hoá ngách: %s trùng gen với %s (khoảng cách %.3f) → đổi gen, khoảng cách mới %.3f%s",
                        agent.agent_id,
                        twin.agent_id,
                        dist,
                        res.nearest,
                        " (gen nhập cư)" if res.immigrant else "",
                    )
            kept.append(agent)
        return shifts

    def diversity(self) -> float:
        return population_diversity([a.params for a in self.active_agents()])


def _diff_params(a: StrategyParams, b: StrategyParams) -> str:
    da, db = a.to_dict(), b.to_dict()
    parts = []
    for k in da:
        if da[k] != db[k]:
            if isinstance(da[k], bool):
                parts.append(f"{k}: {da[k]}→{db[k]}")
            else:
                parts.append(f"{k}: {da[k]:.4g}→{db[k]:.4g}")
    return ", ".join(parts) or "không đổi"

