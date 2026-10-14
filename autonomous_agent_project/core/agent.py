"""Vòng đời của một Autonomous Adaptive Agent.

Mỗi tick, agent:

1. Đóng vị thế đang mở → nếu lỗ, chạy Post-Mortem và cập nhật bộ nhớ chung.
2. Trả chi phí vận hành (tiền "điện" để tồn tại).
3. Kiểm tra sinh tồn: vốn ≤ $0.5 → TỰ HỦY (TERMINATED), ghi nguyên nhân cái chết.
4. Kiểm tra sinh sản: tổng tài sản ≥ $40 → trích $20 sinh agent con (gen đột biến).
5. Đọc lại ``memory_bank.json`` (bắt buộc) rồi mới đề xuất và lọc lệnh mới.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .memory import MemoryBank, describe_context
from .strategy import (
    ExecutionEngine,
    MarketSnapshot,
    Position,
    RuleFilter,
    SimulatedExecutionEngine,
    StrategyParams,
    TradingStrategy,
    clamp_capital,
)

INITIAL_CAPITAL = 20.0
DEATH_THRESHOLD = 0.5
REPLICATION_THRESHOLD = 40.0
REPLICATION_COST = 20.0


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

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


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
    ) -> None:
        self.agent_id = agent_id
        self.memory = memory
        self.rng = rng or random.Random()
        self.params = params or StrategyParams()
        self.strategy = TradingStrategy(self.params, RuleFilter())
        self.engine = engine or SimulatedExecutionEngine(self.rng)
        self.capital = float(capital)
        self.generation = generation
        self.parent_id = parent_id
        self.operating_cost = operating_cost
        self.birth_tick = birth_tick
        self.mutation_rate = mutation_rate
        self.state = AgentState.ACTIVE
        self.position: Position | None = None
        self.stats = AgentStats(max_capital=self.capital)
        self.death_cause: str | None = None
        self.death_tick: int | None = None
        self._last_loss_context: dict[str, Any] | None = None
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

    # ------------------------------------------------------------ Vòng đời
    def step(
        self,
        snapshot: MarketSnapshot,
        *,
        id_factory: Callable[[], str] | None = None,
        can_replicate: bool = True,
    ) -> list["Agent"]:
        """Chạy một nhịp sống. Trả về danh sách agent con vừa được sinh ra."""
        if not self.is_alive:
            return []

        self._settle_position(snapshot)

        self.capital = clamp_capital(self.capital - self.operating_cost)
        self.stats.operating_cost += self.operating_cost

        if self.check_self_destruct(snapshot.tick):
            return []

        children: list[Agent] = []
        if can_replicate and id_factory is not None:
            child = self.try_replicate(snapshot, id_factory)
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
                f"lãi/lỗ thực hiện ${s.realized_pnl:.2f}, phí ${s.fees:.2f}."
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
        return self.is_alive and self.total_assets(snapshot) >= REPLICATION_THRESHOLD

    def try_replicate(self, snapshot: MarketSnapshot, id_factory: Callable[[], str]) -> "Agent | None":
        if not self.should_replicate(snapshot):
            return None
        return self.replicate(id_factory(), snapshot.tick)

    def replicate(self, child_id: str, tick: int) -> "Agent":
        """Trích $20 sinh agent con: thừa hưởng toàn bộ memory_bank và gen đột biến nhẹ."""
        self.capital -= REPLICATION_COST
        self.stats.children += 1
        child_params = self.params.mutate(self.rng, rate=self.mutation_rate)
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
            mutation_rate=self.mutation_rate,
        )
        self.memory.record_birth(
            parent_id=self.agent_id,
            child_id=child_id,
            tick=tick,
            generation=child.generation,
            parent_params=self.params.to_dict(),
            child_params=child_params.to_dict(),
        )
        self.log.info(
            "🧬 Tự nhân bản → %s (thế hệ %d). Cha còn $%.2f, con nhận $%.2f. Gen đột biến: %s",
            child_id,
            child.generation,
            self.capital,
            child.capital,
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
            "tham_so": self.params.to_dict(),
            "thong_ke": self.stats.to_dict(),
            "nguyen_nhan_chet": self.death_cause,
        }


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
