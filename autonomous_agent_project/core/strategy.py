"""Chiến lược kiếm tiền (mô phỏng giao dịch) và bộ lọc quy tắc cấm kỵ.

Thiết kế tách lớp:

* ``MarketSimulator``      – thị trường giả lập chuyển chế độ (tăng/giảm/đi ngang/hỗn loạn).
* ``ExecutionEngine``      – interface thực thi lệnh. ``SimulatedExecutionEngine`` là bản
  mô phỏng; muốn giao dịch thật chỉ cần viết lớp con gọi API sàn và truyền vào agent.
* ``RuleFilter``           – áp quy tắc sinh tồn đọc từ bộ nhớ chung lên lệnh đề xuất.
* ``TradingStrategy``      – sinh lệnh theo động lượng từ bộ gen ``StrategyParams``;
  từ chối ra quyết định nếu không được cung cấp ảnh chụp bộ nhớ mới đọc ở đúng tick.
"""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from .memory import (
    HANH_DONG_CAM,
    HANH_DONG_GIAM_QUY_MO,
    HANH_DONG_GIOI_HAN_DON_BAY,
    MemoryView,
)


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class MemoryNotConsultedError(RuntimeError):
    """Ném ra khi agent định giao dịch mà chưa đọc bộ nhớ ở tick hiện tại."""


# ======================================================================
# Thị trường mô phỏng
# ======================================================================
@dataclass(frozen=True)
class MarketSnapshot:
    tick: int
    price: float
    ret_1: float
    momentum: float
    volatility: float
    regime: str

    def features(self) -> dict[str, float]:
        """Các đặc trưng agent được phép quan sát (chế độ thị trường bị ẩn)."""
        return {
            "price": round(self.price, 6),
            "ret_1": round(self.ret_1, 6),
            "momentum": round(self.momentum, 6),
            "volatility": round(self.volatility, 6),
        }


REGIMES: dict[str, tuple[float, float]] = {
    # tên chế độ: (độ trôi kỳ vọng mỗi tick, độ lệch chuẩn mỗi tick)
    "tang_truong": (0.006, 0.009),
    "suy_thoai": (-0.006, 0.011),
    "di_ngang": (0.0, 0.005),
    "hon_loan": (0.0, 0.035),
}

TEN_CHE_DO = {
    "tang_truong": "Tăng trưởng",
    "suy_thoai": "Suy thoái",
    "di_ngang": "Đi ngang",
    "hon_loan": "Hỗn loạn",
}


class MarketSimulator:
    """Giá tài sản chạy theo mô hình bước ngẫu nhiên chuyển chế độ + sập giá bất ngờ."""

    def __init__(
        self,
        rng: random.Random,
        *,
        start_price: float = 100.0,
        switch_prob: float = 0.025,
        crash_prob: float = 0.004,
        lookback: int = 12,
        vol_window: int = 10,
    ) -> None:
        self.rng = rng
        self.switch_prob = switch_prob
        self.crash_prob = crash_prob
        self.lookback = lookback
        self.vol_window = vol_window
        self.regime = rng.choice(list(REGIMES))
        self.tick = 0
        self.prices: list[float] = [start_price]
        self.returns: list[float] = []
        for _ in range(max(lookback, vol_window)):
            self._advance()
        self._snapshot = self._build_snapshot()

    def _advance(self) -> None:
        if self.rng.random() < self.switch_prob:
            self.regime = self.rng.choice([r for r in REGIMES if r != self.regime])
        drift, sigma = REGIMES[self.regime]
        r = self.rng.gauss(drift, sigma)
        if self.rng.random() < self.crash_prob:
            r += self.rng.choice((-1, 1)) * self.rng.uniform(0.05, 0.12)
        r = max(r, -0.5)
        self.prices.append(self.prices[-1] * (1 + r))
        self.returns.append(r)
        keep = max(self.lookback, self.vol_window) + 2
        del self.prices[:-keep]
        del self.returns[:-keep]

    def _build_snapshot(self) -> MarketSnapshot:
        price = self.prices[-1]
        sma = sum(self.prices[-self.lookback :]) / min(len(self.prices), self.lookback)
        vol = statistics.pstdev(self.returns[-self.vol_window :]) if len(self.returns) > 1 else 0.0
        return MarketSnapshot(
            tick=self.tick,
            price=price,
            ret_1=self.returns[-1] if self.returns else 0.0,
            momentum=price / sma - 1,
            volatility=vol,
            regime=self.regime,
        )

    def snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def step(self) -> MarketSnapshot:
        self.tick += 1
        self._advance()
        self._snapshot = self._build_snapshot()
        return self._snapshot


# ======================================================================
# Bộ gen chiến lược
# ======================================================================
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "momentum_threshold": (0.001, 0.03),
    "position_fraction": (0.05, 0.9),
    "leverage": (1.0, 5.0),
    "vol_tolerance": (0.005, 0.08),
}


@dataclass(frozen=True)
class StrategyParams:
    """Bộ tham số chiến lược — "bộ gen" được di truyền và đột biến."""

    momentum_threshold: float = 0.004
    position_fraction: float = 0.4
    leverage: float = 2.0
    vol_tolerance: float = 0.03
    contrarian: bool = False

    @classmethod
    def random(cls, rng: random.Random) -> "StrategyParams":
        return cls(
            momentum_threshold=rng.uniform(0.002, 0.012),
            position_fraction=rng.uniform(0.3, 0.8),
            leverage=rng.uniform(1.5, 4.5),
            vol_tolerance=rng.uniform(0.01, 0.06),
            contrarian=rng.random() < 0.2,
        )

    def mutate(self, rng: random.Random, rate: float = 0.1, flip_prob: float = 0.05) -> "StrategyParams":
        """Đột biến gen nhẹ: nhân mỗi tham số với nhiễu Gauss ~ N(1, rate), kẹp trong biên."""
        values: dict[str, Any] = {}
        for name, (lo, hi) in PARAM_BOUNDS.items():
            v = getattr(self, name) * rng.gauss(1.0, rate)
            values[name] = round(min(hi, max(lo, v)), 6)
        values["contrarian"] = (not self.contrarian) if rng.random() < flip_prob else self.contrarian
        return StrategyParams(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyParams":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# ======================================================================
# Lệnh, vị thế và interface thực thi
# ======================================================================
@dataclass(frozen=True)
class Order:
    direction: Direction
    fraction: float
    leverage: float
    reason: str = ""

    @property
    def is_trade(self) -> bool:
        return self.direction is not Direction.HOLD

    @property
    def sign(self) -> int:
        return {Direction.LONG: 1, Direction.SHORT: -1}.get(self.direction, 0)


@dataclass
class Position:
    order: Order
    entry_price: float
    entry_tick: int
    notional: float
    margin: float
    entry_fee: float
    context: dict[str, Any] = field(default_factory=dict)

    def unrealized_pnl(self, price: float) -> float:
        return self.notional * self.order.sign * (price / self.entry_price - 1) - self.entry_fee


@dataclass(frozen=True)
class TradeResult:
    pnl: float
    fees: float
    exit_price: float
    return_pct: float


class ExecutionEngine(ABC):
    """Interface thực thi lệnh. Thay bằng lớp con gọi API sàn để giao dịch thật."""

    @abstractmethod
    def open_position(
        self, order: Order, capital: float, snapshot: MarketSnapshot, context: dict[str, Any]
    ) -> Position: ...

    @abstractmethod
    def close_position(self, position: Position, snapshot: MarketSnapshot) -> TradeResult: ...


class SimulatedExecutionEngine(ExecutionEngine):
    """Khớp lệnh giả lập theo giá mô phỏng, có phí giao dịch và trượt giá."""

    def __init__(self, rng: random.Random, fee_rate: float = 0.0005, max_slippage: float = 0.0005) -> None:
        self.rng = rng
        self.fee_rate = fee_rate
        self.max_slippage = max_slippage

    def _slip(self, price: float, sign: int) -> float:
        return price * (1 + sign * self.rng.uniform(0, self.max_slippage))

    def open_position(
        self, order: Order, capital: float, snapshot: MarketSnapshot, context: dict[str, Any]
    ) -> Position:
        margin = capital * order.fraction
        notional = margin * order.leverage
        return Position(
            order=order,
            entry_price=self._slip(snapshot.price, order.sign),
            entry_tick=snapshot.tick,
            notional=notional,
            margin=margin,
            entry_fee=notional * self.fee_rate,
            context=context,
        )

    def close_position(self, position: Position, snapshot: MarketSnapshot) -> TradeResult:
        exit_price = self._slip(snapshot.price, -position.order.sign)
        gross = position.notional * position.order.sign * (exit_price / position.entry_price - 1)
        exit_fee = position.notional * self.fee_rate
        fees = position.entry_fee + exit_fee
        pnl = gross - fees
        return TradeResult(
            pnl=pnl,
            fees=fees,
            exit_price=exit_price,
            return_pct=pnl / position.margin if position.margin else 0.0,
        )


# ======================================================================
# Bộ lọc quy tắc cấm kỵ
# ======================================================================
_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "in": lambda a, b: a in b,
}


@dataclass(frozen=True)
class FilterDecision:
    original: Order
    order: Order
    applied_rules: tuple[str, ...] = ()
    blocked_by: str | None = None

    @property
    def blocked(self) -> bool:
        return self.blocked_by is not None


class RuleFilter:
    """Đối chiếu lệnh đề xuất với các quy tắc sinh tồn trong bộ nhớ chung."""

    @staticmethod
    def condition_matches(cond: dict[str, Any], context: dict[str, Any]) -> bool:
        field_name = cond.get("truong")
        op = _OPS.get(cond.get("toan_tu", ""))
        if op is None or field_name not in context:
            return False
        try:
            return bool(op(context[field_name], cond.get("gia_tri")))
        except TypeError:
            return False

    @classmethod
    def rule_matches(cls, rule: dict[str, Any], context: dict[str, Any]) -> bool:
        conds = rule.get("dieu_kien", [])
        return bool(conds) and all(cls.condition_matches(c, context) for c in conds)

    @staticmethod
    def build_context(order: Order, snapshot: MarketSnapshot) -> dict[str, Any]:
        ctx: dict[str, Any] = snapshot.features()
        ctx.update(
            direction=order.direction.value,
            leverage=round(order.leverage, 6),
            fraction=round(order.fraction, 6),
        )
        return ctx

    def apply(self, order: Order, snapshot: MarketSnapshot, rules: list[dict[str, Any]]) -> FilterDecision:
        if not order.is_trade:
            return FilterDecision(order, order)
        current = order
        applied: list[str] = []
        # Quy tắc CẤM được xét trước, sau đó tới các quy tắc điều chỉnh.
        priority = {HANH_DONG_CAM: 0, HANH_DONG_GIOI_HAN_DON_BAY: 1, HANH_DONG_GIAM_QUY_MO: 2}
        for rule in sorted(rules, key=lambda r: priority.get(r.get("hanh_dong", ""), 9)):
            if not rule.get("hieu_luc", True):
                continue
            if not self.rule_matches(rule, self.build_context(current, snapshot)):
                continue
            action = rule.get("hanh_dong")
            params = rule.get("tham_so", {})
            if action == HANH_DONG_CAM:
                hold = Order(Direction.HOLD, 0.0, 1.0, f"bị chặn bởi {rule['id']}: {rule.get('mo_ta', '')}")
                return FilterDecision(order, hold, tuple(applied + [rule["id"]]), rule["id"])
            if action == HANH_DONG_GIOI_HAN_DON_BAY:
                cap = float(params.get("don_bay_toi_da", 1.0))
                current = replace(current, leverage=min(current.leverage, cap))
            elif action == HANH_DONG_GIAM_QUY_MO:
                current = replace(current, fraction=current.fraction * float(params.get("he_so", 0.5)))
            else:
                continue
            applied.append(rule["id"])
        return FilterDecision(order, current, tuple(applied))


# ======================================================================
# Chiến lược
# ======================================================================
class TradingStrategy:
    """Chiến lược động lượng (hoặc đảo chiều nếu gen ``contrarian`` bật)."""

    def __init__(self, params: StrategyParams, rule_filter: RuleFilter | None = None) -> None:
        self.params = params
        self.rule_filter = rule_filter or RuleFilter()

    def propose(self, snapshot: MarketSnapshot) -> Order:
        p = self.params
        if abs(snapshot.momentum) < p.momentum_threshold:
            return Order(Direction.HOLD, 0.0, 1.0, "tín hiệu động lượng quá yếu")
        if snapshot.volatility > p.vol_tolerance:
            return Order(Direction.HOLD, 0.0, 1.0, "biến động vượt ngưỡng chịu đựng của gen")
        up = snapshot.momentum > 0
        if p.contrarian:
            up = not up
        direction = Direction.LONG if up else Direction.SHORT
        return Order(direction, p.position_fraction, p.leverage, "theo động lượng" if not p.contrarian else "đánh đảo chiều")

    def decide(self, snapshot: MarketSnapshot, memory_view: MemoryView | None) -> FilterDecision:
        """Ra quyết định — BẮT BUỘC kèm ảnh chụp bộ nhớ vừa đọc ở đúng tick này."""
        if memory_view is None:
            raise MemoryNotConsultedError("Chưa đọc memory_bank.json trước khi giao dịch")
        if memory_view.tick != snapshot.tick:
            raise MemoryNotConsultedError(
                f"Ảnh chụp bộ nhớ đã cũ (tick {memory_view.tick} ≠ tick hiện tại {snapshot.tick})"
            )
        order = self.propose(snapshot)
        return self.rule_filter.apply(order, snapshot, memory_view.active_rules())


def clamp_capital(x: float) -> float:
    return 0.0 if math.isnan(x) else max(0.0, x)
