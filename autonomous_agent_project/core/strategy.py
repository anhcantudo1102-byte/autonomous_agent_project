"""Chiến lược kiếm tiền (mô phỏng giao dịch) và bộ lọc quy tắc cấm kỵ.

Thiết kế tách lớp:

* ``MarketSimulator``      – thị trường giả lập chuyển chế độ (tăng/giảm/đi ngang/hỗn loạn).
* ``ExecutionEngine``      – interface thực thi lệnh. ``SimulatedExecutionEngine`` là bản
  mô phỏng; muốn giao dịch thật chỉ cần viết lớp con gọi API sàn và truyền vào agent.
* ``RuleFilter``           – áp quy tắc sinh tồn đọc từ bộ nhớ chung lên lệnh đề xuất.
* ``TradingStrategy``      – sinh lệnh theo động lượng từ bộ gen ``StrategyParams``;
  từ chối ra quyết định nếu không được cung cấp ảnh chụp bộ nhớ mới đọc ở đúng tick.
* Toán tử di truyền trên bộ gen: lai ghép (``StrategyParams.crossover``), đột biến
  (``mutate``), đột biến thích ứng (``adaptive_mutation_rate``), đo khoảng cách gen và
  chống trùng lặp bộ gen (``diversify``).
"""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

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
    # Điểm hợp lưu đa khung (AdvancedMarketAnalyzer); None khi không bật --advanced-analysis.
    confluence: float | None = None

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
    "thuc_te": "Giá thật",
    "gia_lap_du_phong": "Giả lập (API lỗi)",
}


def build_snapshot(
    tick: int,
    prices: Sequence[float],
    returns: Sequence[float],
    *,
    lookback: int,
    vol_window: int,
    regime: str,
) -> MarketSnapshot:
    """Tính các chỉ báo agent quan sát được từ lịch sử giá (dùng chung cho giá giả lập và giá thật)."""
    price = prices[-1]
    window = prices[-lookback:]
    sma = sum(window) / len(window)
    vol = statistics.pstdev(returns[-vol_window:]) if len(returns) > 1 else 0.0
    return MarketSnapshot(
        tick=tick,
        price=price,
        ret_1=returns[-1] if returns else 0.0,
        momentum=price / sma - 1,
        volatility=vol,
        regime=regime,
    )


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
        return build_snapshot(
            self.tick,
            self.prices,
            self.returns,
            lookback=self.lookback,
            vol_window=self.vol_window,
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
# Gen tuỳ chọn: mức tin tưởng vào điểm hợp lưu đa khung (0 = bỏ qua, 1 = tin hoàn toàn).
# None = tính năng tắt; khi đó gen không tham gia đột biến/lai ghép/khoảng cách, nên không tốn
# thêm số ngẫu nhiên và kết quả theo seed giữ nguyên như trước khi có gen này.
CONFLUENCE_WEIGHT_BOUNDS = (0.0, 1.0)
CONFLUENCE_SCALE = 5.0        # |điểm| ≥ 5 coi là tín hiệu đầy đủ
CONFLUENCE_VETO = 0.5         # độ mạnh (tin tưởng × |điểm chuẩn hoá|) để phủ quyết lệnh ngược chiều
CONFLUENCE_ENTRY = 0.6        # độ mạnh để tự vào lệnh khi động lượng chưa đủ


@dataclass(frozen=True)
class StrategyParams:
    """Bộ tham số chiến lược — "bộ gen" được di truyền và đột biến."""

    momentum_threshold: float = 0.004
    position_fraction: float = 0.4
    leverage: float = 2.0
    vol_tolerance: float = 0.03
    contrarian: bool = False
    confluence_weight: float | None = None

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
        if self.confluence_weight is not None:
            lo, hi = CONFLUENCE_WEIGHT_BOUNDS
            # Cộng nhiễu (không nhân) để gen có thể rời khỏi 0.
            values["confluence_weight"] = round(min(hi, max(lo, self.confluence_weight + rng.gauss(0.0, rate))), 6)
        return StrategyParams(**values)

    @classmethod
    def crossover(
        cls,
        parents: Sequence["StrategyParams"],
        rng: random.Random,
        weights: Sequence[float] | None = None,
    ) -> "StrategyParams":
        """Lai ghép đồng nhất: mỗi gen lấy từ một cha/mẹ, chọn ngẫu nhiên theo trọng số Fitness."""
        if not parents:
            raise ValueError("Cần ít nhất một bộ gen để lai ghép")
        if weights is None or sum(max(0.0, w) for w in weights) <= 0:
            weights = [1.0] * len(parents)
        weights = [max(0.0, w) for w in weights]
        values = {
            name: getattr(rng.choices(parents, weights=weights, k=1)[0], name)
            for name in (*PARAM_BOUNDS, "contrarian")
        }
        if any(p.confluence_weight is not None for p in parents):
            donor = rng.choices(parents, weights=weights, k=1)[0]
            fallback = next(p.confluence_weight for p in parents if p.confluence_weight is not None)
            values["confluence_weight"] = donor.confluence_weight if donor.confluence_weight is not None else fallback
        return cls(**values)

    def genome(self) -> tuple[float, ...]:
        """Bộ gen chuẩn hoá về [0, 1] để đo khoảng cách giữa các chiến lược."""
        vec = [(getattr(self, n) - lo) / (hi - lo) for n, (lo, hi) in PARAM_BOUNDS.items()]
        vec.append(1.0 if self.contrarian else 0.0)
        if self.confluence_weight is not None:
            lo, hi = CONFLUENCE_WEIGHT_BOUNDS
            vec.append((self.confluence_weight - lo) / (hi - lo))
        return tuple(min(1.0, max(0.0, v)) for v in vec)

    def distance(self, other: "StrategyParams") -> float:
        """Khoảng cách Euclid chuẩn hoá trong [0, 1]; 0 nghĩa là hai bộ gen trùng hệt."""
        a, b = self.genome(), other.genome()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["confluence_weight"] is None:
            del data["confluence_weight"]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyParams":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def nearest_distance(candidate: StrategyParams, population: Iterable[StrategyParams]) -> float:
    """Khoảng cách tới bộ gen gần nhất trong quần thể (1.0 nếu quần thể rỗng)."""
    return min((candidate.distance(p) for p in population), default=1.0)


def population_diversity(population: Sequence[StrategyParams]) -> float:
    """Độ đa dạng quần thể = khoảng cách trung bình giữa mọi cặp bộ gen."""
    pairs = [(a, b) for i, a in enumerate(population) for b in population[i + 1 :]]
    if not pairs:
        return 0.0
    return sum(a.distance(b) for a, b in pairs) / len(pairs)


def adaptive_mutation_rate(
    failure_pressure: float,
    *,
    base_rate: float = 0.1,
    min_rate: float = 0.03,
    max_rate: float = 0.35,
    high_pressure: float = 0.4,
) -> float:
    """Biên độ đột biến thích ứng theo áp lực thất bại (số lệnh lỗ / agent / tick).

    Áp lực 0 → ``min_rate`` (tinh chỉnh chiến lược đang tốt); áp lực bằng một nửa
    ``high_pressure`` → ``base_rate``; áp lực ≥ ``high_pressure`` → ``max_rate``
    (mở rộng tìm kiếm chiến lược mới). Nội suy tuyến tính giữa các mốc.
    """
    x = min(1.0, max(0.0, failure_pressure / high_pressure)) if high_pressure > 0 else 1.0
    if x <= 0.5:
        return min_rate + (base_rate - min_rate) * (x / 0.5)
    return base_rate + (max_rate - base_rate) * ((x - 0.5) / 0.5)


@dataclass(frozen=True)
class DiversityResult:
    params: StrategyParams
    nearest: float
    attempts: int
    immigrant: bool


def diversify(
    candidate: StrategyParams,
    population: Sequence[StrategyParams],
    rng: random.Random,
    *,
    min_distance: float,
    rate: float,
    max_attempts: int = 6,
) -> DiversityResult:
    """Chống đồng huyết thống: đột biến lại (biên độ tăng dần) tới khi đủ khác quần thể.

    Nếu sau ``max_attempts`` lần vẫn trùng lặp, thay một nửa số gen bằng gen ngẫu nhiên
    ("gen nhập cư") để mở ra một ngách chiến lược mới.
    """
    params = candidate
    nearest = nearest_distance(params, population)
    attempts = 0
    current_rate = max(rate, 0.05)
    while nearest < min_distance and attempts < max_attempts:
        attempts += 1
        params = params.mutate(rng, rate=current_rate)
        nearest = nearest_distance(params, population)
        current_rate = min(1.0, current_rate * 1.5)
    if nearest >= min_distance:
        return DiversityResult(params, nearest, attempts, immigrant=False)
    stranger = StrategyParams.random(rng)
    names = list(PARAM_BOUNDS)
    swapped = set(rng.sample(names, k=max(1, len(names) // 2)))
    values = {n: getattr(stranger if n in swapped else params, n) for n in names}
    values["contrarian"] = stranger.contrarian if rng.random() < 0.5 else params.contrarian
    values["confluence_weight"] = params.confluence_weight
    params = StrategyParams(**values)
    return DiversityResult(params, nearest_distance(params, population), attempts, immigrant=True)


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
        order = self._momentum_order(snapshot)
        if self.params.confluence_weight is None or snapshot.confluence is None:
            return order
        return self._apply_confluence(order, snapshot)

    def _apply_confluence(self, order: Order, snapshot: MarketSnapshot) -> Order:
        """Dùng điểm hợp lưu đa khung làm bộ lọc/thiên hướng, theo mức tin tưởng ``confluence_weight``.

        Độ mạnh = tin tưởng × |điểm| / ``CONFLUENCE_SCALE`` (kẹp trong [0, 1]):

        * lệnh động lượng **cùng chiều** điểm: tăng tỷ trọng tới ×(1 + 0.5 × độ mạnh), tối đa 90%;
        * lệnh **ngược chiều**: độ mạnh ≥ ``CONFLUENCE_VETO`` → phủ quyết (HOLD), yếu hơn → giảm tỷ trọng;
        * động lượng chưa đủ mà độ mạnh ≥ ``CONFLUENCE_ENTRY`` → tự vào lệnh theo chiều điểm.

        Biến động vượt ngưỡng chịu đựng của gen vẫn luôn đứng ngoài. Lệnh sau đó vẫn qua bộ lọc
        quy tắc (CẤM/giảm quy mô/giới hạn đòn bẩy) của bộ nhớ chung.
        """
        p = self.params
        score = float(snapshot.confluence or 0.0)
        strength = min(1.0, max(0.0, p.confluence_weight or 0.0) * abs(score) / CONFLUENCE_SCALE)
        if strength == 0 or snapshot.volatility > p.vol_tolerance:
            return order
        want = Direction.LONG if score > 0 else Direction.SHORT
        lo_f, hi_f = PARAM_BOUNDS["position_fraction"]
        if order.is_trade:
            if order.direction == want:
                fraction = min(hi_f, order.fraction * (1 + 0.5 * strength))
                return replace(order, fraction=fraction, reason=f"{order.reason}, đa khung ủng hộ ({score:+.1f})")
            if strength >= CONFLUENCE_VETO:
                return Order(Direction.HOLD, 0.0, 1.0, f"đa khung phủ quyết (điểm {score:+.1f})")
            fraction = max(lo_f, order.fraction * (1 - strength))
            return replace(order, fraction=fraction, reason=f"{order.reason}, đa khung ngược chiều ({score:+.1f})")
        if strength >= CONFLUENCE_ENTRY:
            return Order(want, max(lo_f, p.position_fraction * strength), p.leverage,
                         f"vào lệnh theo đa khung (điểm {score:+.1f})")
        return order

    def _momentum_order(self, snapshot: MarketSnapshot) -> Order:
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
