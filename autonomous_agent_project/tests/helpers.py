"""Tiện ích dùng chung cho các bài kiểm thử."""

from __future__ import annotations

import random
from typing import Any

from core.strategy import (
    ExecutionEngine,
    MarketSnapshot,
    Order,
    Position,
    TradeResult,
)


def snap(tick: int = 1, price: float = 100.0, momentum: float = 0.02, volatility: float = 0.01) -> MarketSnapshot:
    return MarketSnapshot(
        tick=tick, price=price, ret_1=0.0, momentum=momentum, volatility=volatility, regime="tang_truong"
    )


class ScriptedEngine(ExecutionEngine):
    """Engine giả: mỗi lần đóng lệnh trả về lãi/lỗ định sẵn theo kịch bản."""

    def __init__(self, pnls: list[float]) -> None:
        self.pnls = list(pnls)
        self.opened: list[Order] = []

    def open_position(
        self, order: Order, capital: float, snapshot: MarketSnapshot, context: dict[str, Any]
    ) -> Position:
        self.opened.append(order)
        notional = capital * order.fraction * order.leverage
        return Position(order, snapshot.price, snapshot.tick, notional, capital * order.fraction, 0.0, context)

    def close_position(self, position: Position, snapshot: MarketSnapshot) -> TradeResult:
        pnl = self.pnls.pop(0) if self.pnls else 0.0
        return TradeResult(pnl=pnl, fees=0.0, exit_price=snapshot.price, return_pct=0.0)


def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)
