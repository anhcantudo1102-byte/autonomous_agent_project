"""Fitness nâng cao: thưởng/phạt kỷ luật giao dịch theo điểm hợp lưu đa khung (``confluence_score``).

Hai hàm ở đây **độc lập** với ``Agent.fitness()`` (GA hiện tại vẫn dùng ``Agent.fitness()``), dùng để đánh
giá một chuỗi lệnh đã ghi lại cùng điểm hợp lưu tại thời điểm vào lệnh:

* ``calculate_advanced_fitness``: điểm tuyệt đối (pnl ± thưởng/phạt cố định);
* ``calculate_advanced_fitness_scaled``: giá trị danh mục theo ROI, thưởng/phạt quy đổi thành % vốn;
* ``calculate_production_fitness``: giá trị danh mục thực, thưởng/phạt hợp lưu nhẹ, phạt Max Drawdown sâu,
  phân hoá agent cháy tài khoản.
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


def _validate_capital(initial_capital: Any) -> float:
    valid = isinstance(initial_capital, (int, float)) and not isinstance(initial_capital, bool)
    if not (valid and math.isfinite(initial_capital) and initial_capital > 0):
        raise ValueError(f"initial_capital phải là số hữu hạn > 0, nhận {initial_capital!r}")
    return float(initial_capital)


def _validate_non_negative(**params: float) -> None:
    for name, value in params.items():
        if value < 0:
            raise ValueError(f"Tham số '{name}' phải ≥ 0, nhận {value!r}")


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
    _validate_non_negative(bonus_per_point=bonus_per_point, counter_penalty=counter_penalty,
                           noise_penalty=noise_penalty, noise_threshold=noise_threshold)

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
    capital = _validate_capital(initial_capital)
    _validate_non_negative(strong_threshold=strong_threshold, bonus_rate_per_point=bonus_rate_per_point,
                           counter_penalty_rate=counter_penalty_rate, noise_penalty_rate=noise_penalty_rate,
                           noise_threshold=noise_threshold)

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


PRODUCTION_INITIAL_CAPITAL = 20.0
PRODUCTION_STRONG_THRESHOLD = 2.0
PRODUCTION_ALIGNED_BONUS_RATE = 0.001
PRODUCTION_COUNTER_PENALTY_RATE = 0.002
PRODUCTION_DRAWDOWN_THRESHOLD = 0.20
PRODUCTION_DRAWDOWN_PENALTY_RATE = 0.10
PRODUCTION_FLOOR = 0.01
PRODUCTION_BLOWN_RATIO = 0.5


def max_drawdown_from_trades(pnls: Iterable[float], initial_capital: float) -> float:
    """Max Drawdown (0–1) của đường vốn ``initial_capital + pnl cộng dồn`` theo đúng thứ tự lệnh.

    Đỉnh tính cả vốn khởi đầu; sụt quá 100% (vốn âm) được kẹp về 1.0.
    """
    equity = peak = float(initial_capital)
    worst = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak)
    return min(1.0, worst)


def calculate_production_fitness(
    agent: Any,
    trade_history: Iterable[Any] | None,
    initial_capital: float = PRODUCTION_INITIAL_CAPITAL,
    *,
    strong_threshold: float = PRODUCTION_STRONG_THRESHOLD,
    aligned_bonus_rate: float = PRODUCTION_ALIGNED_BONUS_RATE,
    counter_penalty_rate: float = PRODUCTION_COUNTER_PENALTY_RATE,
    drawdown_threshold: float = PRODUCTION_DRAWDOWN_THRESHOLD,
    drawdown_penalty_rate: float = PRODUCTION_DRAWDOWN_PENALTY_RATE,
    floor: float = PRODUCTION_FLOOR,
    blown_ratio: float = PRODUCTION_BLOWN_RATIO,
    other_costs: float = 0.0,
) -> float:
    """Fitness "thương mại": giá trị danh mục thực, điều chỉnh nhẹ theo kỷ luật hợp lưu và Drawdown.

    ``vốn cuối = initial_capital + tổng pnl − other_costs`` (``other_costs``: chi phí ngoài lệnh đã trả, vd chi phí
    vận hành mỗi tick; mặc định 0). Nếu ``vốn cuối > 0``::

        điểm = vốn cuối
             + aligned_bonus_rate × vốn ban đầu × (số lệnh thuận chiều, |score| ≥ strong_threshold)
             − counter_penalty_rate × vốn ban đầu × (số lệnh ngược chiều, |score| ≥ strong_threshold)
             − drawdown_penalty_rate × vốn ban đầu × max(0, MaxDD − drawdown_threshold)
        điểm = max(floor, điểm)

    Mặc định: thưởng 0.1% vốn/lệnh, phạt 0.2% vốn/lệnh, chỉ phạt phần Max DD vượt 20% với hệ số 0.1
    (DD 50% → −3% vốn, tối đa −8% vốn khi DD 100%), nên không lấn át lãi lỗ thực tế. Max DD lấy từ
    đường vốn cộng dồn theo **thứ tự** ``trade_history`` (xem ``max_drawdown_from_trades``).
    **Không** phạt giao dịch trong vùng nhiễu (kể cả ``|score| < 0.5``); vùng ``|score| < strong_threshold``
    không thưởng/phạt.

    Agent cháy tài khoản (``vốn cuối ≤ 0``) nhận ``floor × blown_ratio / (1 + |vốn cuối| / initial_capital)``:
    luôn dương, giảm đơn điệu theo mức lỗ và luôn nhỏ hơn ``floor`` — tức nhỏ hơn mọi agent còn vốn.
    Dùng dạng hữu tỷ thay cho ``exp`` vì ``exp`` tràn dưới về 0 khi lỗ rất sâu. Nhánh này không làm tròn
    (để các giá trị rất nhỏ vẫn phân biệt được); nhánh còn lại làm tròn 6 chữ số.

    Quy ước chung với hai hàm trên: chưa có lệnh và ``other_costs == 0`` → đúng ``initial_capital`` (có chi phí thì
    tính như trên với tổng pnl = 0, để đứng ngoài không được miễn chi phí); ``action`` không phân biệt hoa
    thường, ``LONG``/``SHORT``/``STRONG_*`` được chấp nhận, ``HOLD``/lạ → chỉ tính pnl; ``pnl``/``score``
    thiếu hoặc ``None`` → 0 / không thưởng-phạt; dict hoặc đối tượng; số không hợp lệ, NaN, vô cực,
    ``initial_capital`` ≤ 0 hoặc tham số âm → ``ValueError``. ``agent`` chưa được dùng.

    Agent hoà vốn, không có lệnh ngược chiều mạnh và Max DD ≤ ngưỡng nhận đúng ``initial_capital`` (bằng
    đứng ngoài). Điểm vẫn thấp hơn ``initial_capital`` dù hoà vốn khi có lệnh ngược chiều mạnh (nhiều hơn
    số lệnh thuận chiều × tỷ lệ thưởng/phạt) hoặc Max DD giữa chừng vượt ngưỡng.
    """
    capital = _validate_capital(initial_capital)
    _validate_non_negative(strong_threshold=strong_threshold, aligned_bonus_rate=aligned_bonus_rate,
                           counter_penalty_rate=counter_penalty_rate, drawdown_threshold=drawdown_threshold,
                           drawdown_penalty_rate=drawdown_penalty_rate, floor=floor, blown_ratio=blown_ratio)
    valid = isinstance(other_costs, (int, float)) and not isinstance(other_costs, bool)
    if not (valid and math.isfinite(other_costs) and other_costs >= 0):
        raise ValueError(f"Tham số 'other_costs' phải là số hữu hạn ≥ 0, nhận {other_costs!r}")
    costs = float(other_costs)
    if blown_ratio >= 1:
        raise ValueError(f"blown_ratio phải < 1 để agent cháy tài khoản luôn dưới sàn, nhận {blown_ratio!r}")

    trades = list(trade_history or ())
    if not trades and costs == 0:
        return capital

    pnls: list[float] = []
    adjustment = 0.0
    for index, trade in enumerate(trades):
        pnl = _field(trade, "pnl")
        pnls.append(0.0 if pnl is None else _number(pnl, "pnl", index))

        raw_score = _field(trade, "confluence_score")
        if raw_score is None:
            continue
        score = _number(raw_score, "confluence_score", index)
        direction = _direction(_field(trade, "action"))
        if direction == 0 or score == 0 or abs(score) < strong_threshold:
            continue
        adjustment += aligned_bonus_rate * capital if direction * score > 0 else -counter_penalty_rate * capital

    portfolio = capital + sum(pnls) - costs
    if portfolio <= 0:
        return floor * blown_ratio / (1.0 + abs(portfolio) / capital)

    drawdown = max_drawdown_from_trades(pnls, capital)
    drawdown_penalty = drawdown_penalty_rate * capital * max(0.0, drawdown - drawdown_threshold)
    return round(max(floor, portfolio + adjustment - drawdown_penalty), 6)
