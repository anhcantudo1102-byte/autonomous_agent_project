"""Fitness nâng cao: thưởng/phạt kỷ luật giao dịch theo điểm hợp lưu đa khung (``confluence_score``).

Hàm này **độc lập** với ``Agent.fitness()`` (GA hiện tại vẫn dùng ``Agent.fitness()``), dùng để đánh giá
một chuỗi lệnh đã ghi lại cùng điểm hợp lưu tại thời điểm vào lệnh.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

ALIGNMENT_BONUS_PER_POINT = 10.0
COUNTER_TREND_PENALTY = 20.0
NOISE_PENALTY = 5.0
NOISE_THRESHOLD = 0.5
FITNESS_FLOOR = 0.1

_BUY_ACTIONS = frozenset({"BUY", "LONG", "STRONG_BUY"})
_SELL_ACTIONS = frozenset({"SELL", "SHORT", "STRONG_SELL"})


def _field(trade: Any, key: str) -> Any:
    if isinstance(trade, Mapping):
        return trade.get(key)
    return getattr(trade, key, None)


def _number(value: Any, name: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Lệnh #{index}: '{name}' phải là số, nhận {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Lệnh #{index}: '{name}' phải là số hữu hạn, nhận {value!r}")
    return value


def calculate_advanced_fitness(
    agent: Any,
    trades: Iterable[Any] | None,
    *,
    bonus_per_point: float = ALIGNMENT_BONUS_PER_POINT,
    counter_penalty: float = COUNTER_TREND_PENALTY,
    noise_penalty: float = NOISE_PENALTY,
    noise_threshold: float = NOISE_THRESHOLD,
    floor: float = FITNESS_FLOOR,
) -> float:
    """Fitness = tổng pnl + thưởng lệnh thuận hợp lưu − phạt lệnh ngược dòng/giao dịch trong nhiễu, sàn ``floor``.

    Với mỗi lệnh (dict hoặc đối tượng có thuộc tính ``pnl``, ``action``, ``confluence_score``):

    * ``pnl`` luôn được cộng vào (thiếu hoặc ``None`` → 0).
    * ``|score| < noise_threshold`` (vùng nhiễu, mặc định 0.5): phạt ``noise_penalty`` bất kể chiều lệnh —
      nhiễu được xét TRƯỚC, nên lệnh ngược chiều nhưng ``|score|`` nhỏ chỉ bị phạt nhiễu, không bị phạt 20.
      ``|score|`` đúng bằng ngưỡng được coi là tín hiệu rõ; ``score == 0`` luôn là nhiễu (kể cả khi ngưỡng 0).
    * Tín hiệu rõ và lệnh thuận chiều (mua khi score > 0, bán khi score < 0): thưởng ``|score| × bonus_per_point``.
    * Tín hiệu rõ và lệnh ngược chiều: phạt ``counter_penalty``.

    Quy ước cho các trường hợp mơ hồ:

    * ``action`` không phân biệt hoa thường; ``BUY``/``LONG``/``STRONG_BUY`` là mua,
      ``SELL``/``SHORT``/``STRONG_SELL`` là bán. ``HOLD``, chuỗi lạ hoặc thiếu ``action``: chỉ cộng pnl,
      không thưởng/phạt (không phải lệnh có chiều).
    * Thiếu ``confluence_score`` hoặc ``None`` (lệnh mở khi chưa bật phân tích đa khung): chỉ cộng pnl,
      không phạt nhiễu — không thể trách agent vì tín hiệu không tồn tại.
    * ``trades`` rỗng hoặc ``None`` → ``floor``.
    * ``pnl``/``confluence_score`` không phải số (kể cả bool), NaN hoặc vô cực → ``ValueError``; tham số
      cấu hình âm → ``ValueError``.

    ``agent`` hiện chưa được dùng trong công thức (giữ trong chữ ký theo đặc tả, để mở rộng sau).
    Kết quả làm tròn 6 chữ số thập phân để tránh sai số dấu phẩy động khi so sánh.
    """
    for name, value in (("bonus_per_point", bonus_per_point), ("counter_penalty", counter_penalty),
                        ("noise_penalty", noise_penalty), ("noise_threshold", noise_threshold)):
        if value < 0:
            raise ValueError(f"Tham số '{name}' phải ≥ 0, nhận {value!r}")

    total = 0.0
    for index, trade in enumerate(trades or ()):
        pnl = _field(trade, "pnl")
        total += 0.0 if pnl is None else _number(pnl, "pnl", index)

        raw_score = _field(trade, "confluence_score")
        action = str(_field(trade, "action") or "").upper()
        if raw_score is None:
            continue
        score = _number(raw_score, "confluence_score", index)
        if action in _BUY_ACTIONS:
            direction = 1
        elif action in _SELL_ACTIONS:
            direction = -1
        else:
            continue

        if abs(score) < noise_threshold or score == 0:
            total -= noise_penalty
        elif direction * score > 0:
            total += abs(score) * bonus_per_point
        else:
            total -= counter_penalty
    return round(max(floor, total), 6)
