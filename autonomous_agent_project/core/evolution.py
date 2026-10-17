"""Fitness nâng cao: thưởng/phạt kỷ luật giao dịch theo điểm hợp lưu đa khung (``confluence_score``).

Hai hàm ở đây **độc lập** với ``Agent.fitness()`` (GA hiện tại vẫn dùng ``Agent.fitness()``), dùng để đánh
giá một chuỗi lệnh đã ghi lại cùng điểm hợp lưu tại thời điểm vào lệnh:

* ``calculate_advanced_fitness``: điểm tuyệt đối (pnl ± thưởng/phạt cố định);
* ``calculate_advanced_fitness_scaled``: giá trị danh mục theo ROI, thưởng/phạt quy đổi thành % vốn.
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


def _direction(action: Any) -> int:
    """+1 nếu là lệnh mua, −1 nếu là lệnh bán, 0 nếu không có chiều (HOLD, chuỗi lạ, None)."""
    name = str(action or "").upper()
    if name in _BUY_ACTIONS:
        return 1
    if name in _SELL_ACTIONS:
        return -1
    return 0


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
        if raw_score is None:
            continue
        score = _number(raw_score, "confluence_score", index)
        direction = _direction(_field(trade, "action"))
        if direction == 0:
            continue

        if abs(score) < noise_threshold or score == 0:
            total -= noise_penalty
        elif direction * score > 0:
            total += abs(score) * bonus_per_point
        else:
            total -= counter_penalty
    return round(max(floor, total), 6)


SCALED_INITIAL_CAPITAL = 20.0
SCALED_STRONG_THRESHOLD = 2.0
SCALED_BONUS_RATE_PER_POINT = 0.002
SCALED_COUNTER_PENALTY_RATE = 0.005
SCALED_NOISE_PENALTY_RATE = 0.0025


def calculate_advanced_fitness_scaled(
    agent: Any,
    trade_history: Iterable[Any] | None,
    initial_capital: float = SCALED_INITIAL_CAPITAL,
    *,
    strong_threshold: float = SCALED_STRONG_THRESHOLD,
    bonus_rate_per_point: float = SCALED_BONUS_RATE_PER_POINT,
    counter_penalty_rate: float = SCALED_COUNTER_PENALTY_RATE,
    noise_penalty_rate: float = SCALED_NOISE_PENALTY_RATE,
    noise_threshold: float = NOISE_THRESHOLD,
    floor: float = FITNESS_FLOOR,
) -> float:
    """Fitness theo ROI, thưởng/phạt hợp lưu quy đổi thành % vốn để không lấn át lãi lỗ thực tế.

    ``tỷ lệ = tổng pnl / initial_capital + Σ thưởng − Σ phạt``; kết quả là giá trị danh mục quy đổi
    ``initial_capital × (1 + tỷ lệ)``, sàn ``floor``. Với mỗi lệnh có chiều (mua/bán):

    * thuận chiều và ``|score| ≥ strong_threshold`` (mặc định 2): thưởng ``|score| × 0.2%``;
    * ngược chiều và ``|score| ≥ strong_threshold``: phạt 0.5%;
    * ``|score| < noise_threshold`` (mặc định 0.5, ``score == 0`` luôn là nhiễu): phạt 0.25%;
    * ``noise_threshold ≤ |score| < strong_threshold``: không thưởng, không phạt.

    Quy ước (thống nhất với ``calculate_advanced_fitness``):

    * chưa có lệnh (rỗng/``None``) → trả về đúng ``initial_capital`` (không áp sàn);
    * ``action`` không phân biệt hoa thường, ``LONG``/``STRONG_BUY`` là mua, ``SHORT``/``STRONG_SELL`` là bán;
      ``HOLD``/chuỗi lạ/thiếu → chỉ tính pnl;
    * ``pnl`` thiếu hoặc ``None`` → 0; ``confluence_score`` thiếu hoặc ``None`` → không thưởng/phạt
      (bản gốc coi thiếu khoá là 0.0 nên phạt nhiễu, và ``None`` gây ``TypeError``);
    * lệnh là dict hoặc đối tượng có thuộc tính tương ứng;
    * số không hợp lệ (không phải số, bool, NaN, vô cực) → ``ValueError``; ``initial_capital`` phải là số
      hữu hạn > 0 (bản gốc âm thầm dùng ROI = 0); tham số tỷ lệ/ngưỡng âm → ``ValueError``.

    Lưu ý ngữ nghĩa (giữ đúng ý người dùng): agent chưa giao dịch nhận ``initial_capital`` (vd 20.0), cao hơn
    agent hoà vốn nhưng vi phạm kỷ luật, còn mọi agent lỗ gần hết vốn đều dồn về 0.1.
    ``agent`` chưa được dùng trong công thức. Kết quả làm tròn 6 chữ số thập phân.
    """
    valid = isinstance(initial_capital, (int, float)) and not isinstance(initial_capital, bool)
    if not (valid and math.isfinite(initial_capital) and initial_capital > 0):
        raise ValueError(f"initial_capital phải là số hữu hạn > 0, nhận {initial_capital!r}")
    capital = float(initial_capital)
    for name, value in (("strong_threshold", strong_threshold), ("bonus_rate_per_point", bonus_rate_per_point),
                        ("counter_penalty_rate", counter_penalty_rate), ("noise_penalty_rate", noise_penalty_rate),
                        ("noise_threshold", noise_threshold)):
        if value < 0:
            raise ValueError(f"Tham số '{name}' phải ≥ 0, nhận {value!r}")

    trades = list(trade_history or ())
    if not trades:
        return capital

    total_pnl = 0.0
    adjustment = 0.0
    for index, trade in enumerate(trades):
        pnl = _field(trade, "pnl")
        total_pnl += 0.0 if pnl is None else _number(pnl, "pnl", index)

        raw_score = _field(trade, "confluence_score")
        if raw_score is None:
            continue
        score = _number(raw_score, "confluence_score", index)
        direction = _direction(_field(trade, "action"))
        if direction == 0:
            continue

        if abs(score) >= strong_threshold and score != 0:
            if direction * score > 0:
                adjustment += abs(score) * bonus_rate_per_point
            else:
                adjustment -= counter_penalty_rate
        elif abs(score) < noise_threshold or score == 0:
            adjustment -= noise_penalty_rate

    ratio = total_pnl / capital + adjustment
    return round(max(floor, capital * (1.0 + ratio)), 6)
