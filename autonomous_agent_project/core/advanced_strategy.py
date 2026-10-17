"""Phân tích đa khung thời gian: cấu trúc SMC, Price Action/Hidden Base và sóng Elliott (heuristic).

``MultiDimensionalAnalyzer`` nhận nến (klines) theo định dạng Binance
``[open_time, open, high, low, close, volume, ...]``. Chỉ cần 5 phần tử đầu, nên dữ liệu
giả lập không có volume vẫn dùng được.

* **HTF** (khung lớn, ví dụ 4h): xu hướng chủ đạo theo cấu trúc thị trường — đỉnh/đáy swing,
  Break of Structure (BOS).
* **LTF** (khung nhỏ, ví dụ 5m): điểm vào lệnh theo Price Action — phá vỡ nền giá ẩn (Hidden
  Base), nến nhấn chìm, nến có thân rõ ràng.
* **Elliott**: KHÔNG đếm sóng Elliott thật. Đây là heuristic động lượng — động lượng ngắn hạn
  cùng chiều và mạnh hơn trung hạn được coi là "sóng 3" (nhịp đẩy mạnh nhất).

Điểm hợp lưu: +2 nếu HTF BULLISH và LTF BUY, -2 nếu HTF BEARISH và LTF SELL; nhân 1.5 nếu đang ở
"sóng 3" cùng chiều với tín hiệu. Đây là công cụ minh hoạ, không phải lời khuyên đầu tư.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

Kline = Sequence[Any]

BULLISH, BEARISH, NEUTRAL = "BULLISH", "BEARISH", "NEUTRAL"
BUY, SELL, HOLD = "BUY", "SELL", "HOLD"
WAVE_3 = "IMPULSE_WAVE_3"
WAVE_5 = "IMPULSE_WAVE_5"
CORRECTIVE = "CORRECTIVE"
UNCLEAR = "UNCLEAR"

WAVE_3_MULTIPLIER = 1.5
FLAT_RATE = 0.0005  # dưới 0.05%/nến coi như không có động lượng


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_kline(cls, k: Kline) -> "Candle":
        return cls(float(k[1]), float(k[2]), float(k[3]), float(k[4]))

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low


def _candles(klines: Sequence[Kline] | None) -> list[Candle]:
    return [Candle.from_kline(k) for k in klines or []]


def find_swings(candles: Sequence[Candle], k: int = 2) -> tuple[list[float], list[float]]:
    """Đỉnh/đáy swing kiểu fractal: cao (thấp) nhất trong ``k`` nến mỗi bên."""
    highs: list[float] = []
    lows: list[float] = []
    for i in range(k, len(candles) - k):
        window = candles[i - k : i + k + 1]
        if candles[i].high == max(c.high for c in window) and candles[i].high > candles[i - 1].high:
            highs.append(candles[i].high)
        if candles[i].low == min(c.low for c in window) and candles[i].low < candles[i - 1].low:
            lows.append(candles[i].low)
    return highs, lows


class MultiDimensionalAnalyzer:
    """Bộ phân tích hợp lưu đa chiều cho một cặp giao dịch."""

    MIN_HTF_CANDLES = 10
    MIN_LTF_CANDLES = 5

    def __init__(self, symbol: str = "BTCUSDT", *, swing_k: int = 2, base_len: int = 4, base_max_range: float = 0.004):
        self.symbol = symbol.upper()
        self.swing_k = swing_k
        self.base_len = base_len
        self.base_max_range = base_max_range
        self.reasons: list[str] = []

    def analyze_market_structure(self, htf_klines: Sequence[Kline], ltf_klines: Sequence[Kline]) -> dict[str, Any]:
        """Tổng hợp HTF (xu hướng) + LTF (điểm vào) + Elliott (sức mạnh nhịp) thành điểm hợp lưu."""
        self.reasons = []
        htf_trend = self._check_smc_structure(htf_klines)
        ltf_signal = self._evaluate_price_action_and_bases(ltf_klines)
        wave_status, wave_direction = self._count_elliott_waves(htf_klines)

        score = 0.0
        if htf_trend == BULLISH and ltf_signal == BUY:
            score += 2
        elif htf_trend == BEARISH and ltf_signal == SELL:
            score -= 2

        # Chỉ khuếch đại khi "sóng 3" cùng chiều với điểm hợp lưu (sóng 3 tăng không cộng hưởng lệnh bán).
        if wave_status == WAVE_3 and score != 0 and (score > 0) == (wave_direction == BULLISH):
            score *= WAVE_3_MULTIPLIER

        return {
            "trend": htf_trend,
            "signal": ltf_signal,
            "wave": wave_status,
            "wave_direction": wave_direction,
            "confluence_score": score,
            "reasons": list(self.reasons),
        }

    # ------------------------------------------------------------- HTF: SMC
    def _check_smc_structure(self, klines: Sequence[Kline]) -> str:
        """Xu hướng HTF theo cấu trúc thị trường.

        1. BOS: close cuối vượt đỉnh swing gần nhất → BULLISH; thủng đáy swing gần nhất → BEARISH.
        2. Nếu chưa có BOS: hai đỉnh và hai đáy swing gần nhất cùng nâng lên (HH + HL) → BULLISH;
           cùng hạ xuống (LH + LL) → BEARISH.
        3. Không đủ swing: so close cuối với cao/thấp nhất của nửa đầu dữ liệu (mềm hơn
           ``close > max(mọi close trước)``, vốn gần như không bao giờ xảy ra).
        """
        candles = _candles(klines)
        if len(candles) < self.MIN_HTF_CANDLES:
            self.reasons.append(f"HTF chỉ có {len(candles)} nến (< {self.MIN_HTF_CANDLES}) → NEUTRAL")
            return NEUTRAL
        last = candles[-1].close
        highs, lows = find_swings(candles[:-1], self.swing_k)

        if highs and last > highs[-1]:
            self.reasons.append(f"HTF: BOS tăng — close {last:,.2f} vượt đỉnh swing {highs[-1]:,.2f}")
            return BULLISH
        if lows and last < lows[-1]:
            self.reasons.append(f"HTF: BOS giảm — close {last:,.2f} thủng đáy swing {lows[-1]:,.2f}")
            return BEARISH
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                self.reasons.append("HTF: đỉnh cao dần + đáy cao dần (HH/HL)")
                return BULLISH
            if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                self.reasons.append("HTF: đỉnh thấp dần + đáy thấp dần (LH/LL)")
                return BEARISH

        first_half = candles[: len(candles) // 2]
        if not highs and not lows:
            if last > max(c.high for c in first_half):
                self.reasons.append("HTF: không có swing, giá vượt vùng cao của nửa đầu dữ liệu")
                return BULLISH
            if last < min(c.low for c in first_half):
                self.reasons.append("HTF: không có swing, giá thủng vùng thấp của nửa đầu dữ liệu")
                return BEARISH
        self.reasons.append("HTF: cấu trúc chưa rõ xu hướng")
        return NEUTRAL

    # ---------------------------------------------------------- LTF: PA/Base
    def _evaluate_price_action_and_bases(self, klines: Sequence[Kline]) -> str:
        """Tín hiệu vào lệnh LTF, theo thứ tự ưu tiên.

        1. Hidden Base: ``base_len`` nến trước có biên độ gộp ≤ ``base_max_range`` (nền giá chặt),
           nến cuối đóng cửa vượt đỉnh nền → BUY, thủng đáy nền → SELL.
        2. Nến nhấn chìm: thân nến cuối bao trùm thân nến trước và ngược màu.
        3. Nến có thân ≥ 30% biên độ: xanh → BUY, đỏ → SELL. Nến doji → HOLD.
        """
        candles = _candles(klines)
        if len(candles) < self.MIN_LTF_CANDLES:
            self.reasons.append(f"LTF chỉ có {len(candles)} nến (< {self.MIN_LTF_CANDLES}) → HOLD")
            return HOLD
        last, prev = candles[-1], candles[-2]

        base = candles[-1 - self.base_len : -1]
        if len(base) == self.base_len:
            hi, lo = max(c.high for c in base), min(c.low for c in base)
            if lo > 0 and (hi - lo) / lo <= self.base_max_range:
                if last.close > hi:
                    self.reasons.append(f"LTF: phá vỡ Hidden Base lên (nền {lo:,.2f}–{hi:,.2f})")
                    return BUY
                if last.close < lo:
                    self.reasons.append(f"LTF: phá vỡ Hidden Base xuống (nền {lo:,.2f}–{hi:,.2f})")
                    return SELL

        prev_top, prev_bot = max(prev.open, prev.close), min(prev.open, prev.close)
        last_top, last_bot = max(last.open, last.close), min(last.open, last.close)
        engulfs = last_top >= prev_top and last_bot <= prev_bot and last.body > prev.body
        if engulfs and last.close > last.open and prev.close < prev.open:
            self.reasons.append("LTF: nến nhấn chìm tăng")
            return BUY
        if engulfs and last.close < last.open and prev.close > prev.open:
            self.reasons.append("LTF: nến nhấn chìm giảm")
            return SELL

        if last.range <= 0 or last.body / last.range < 0.3:
            self.reasons.append("LTF: nến cuối thân nhỏ (doji) → chờ")
            return HOLD
        if last.close > last.open:
            self.reasons.append("LTF: nến tăng thân rõ ràng")
            return BUY
        self.reasons.append("LTF: nến giảm thân rõ ràng")
        return SELL

    # -------------------------------------------------------- Elliott (heuristic)
    def _count_elliott_waves(self, klines: Sequence[Kline], short: int = 5, long: int = 20) -> tuple[str, str]:
        """Ước lượng pha sóng bằng động lượng — KHÔNG phải đếm sóng Elliott đầy đủ.

        So tốc độ tăng/giảm mỗi nến của ``short`` nến gần nhất với ``long`` nến gần nhất:

        * cùng chiều, ngắn hạn nhanh hơn → ``IMPULSE_WAVE_3`` (nhịp đẩy đang tăng tốc);
        * cùng chiều, ngắn hạn chậm lại → ``IMPULSE_WAVE_5`` (nhịp đẩy cuối, yếu dần);
        * ngược chiều → ``CORRECTIVE`` (đang điều chỉnh);
        * động lượng dưới ``FLAT_RATE`` mỗi nến hoặc thiếu dữ liệu → ``UNCLEAR``.
        """
        closes = [c.close for c in _candles(klines)]
        if len(closes) < short + 1:
            return UNCLEAR, NEUTRAL
        long = min(long, len(closes) - 1)
        rate_short = (closes[-1] / closes[-1 - short] - 1) / short
        rate_long = (closes[-1] / closes[-1 - long] - 1) / long
        direction = BULLISH if rate_long > 0 else BEARISH if rate_long < 0 else NEUTRAL
        if abs(rate_long) < FLAT_RATE and abs(rate_short) < FLAT_RATE:
            status, direction = UNCLEAR, NEUTRAL
        elif rate_short * rate_long < 0:
            status = CORRECTIVE
        elif abs(rate_short) > abs(rate_long):
            status = WAVE_3
        else:
            status = WAVE_5
        self.reasons.append(
            f"Elliott (heuristic): động lượng {short} nến {rate_short:+.3%}/nến, {long} nến {rate_long:+.3%}/nến → {status}"
        )
        return status, direction


def decide(score: float) -> str:
    """Quyết định giao dịch từ điểm hợp lưu: LONG / SHORT / WAIT."""
    if score > 0:
        return "LONG"
    if score < 0:
        return "SHORT"
    return "WAIT"


def generate_synthetic_klines(
    n: int,
    *,
    start_price: float = 80000.0,
    drift: float = 0.004,
    volatility: float = 0.006,
    rng: random.Random | None = None,
    interval_ms: int = 4 * 60 * 60 * 1000,
    start_time_ms: int = 0,
) -> list[list[float]]:
    """Sinh nến giả lập có xu hướng (``drift`` > 0 tăng, < 0 giảm), định dạng giống Binance."""
    rng = rng or random.Random()
    rows: list[list[float]] = []
    price = start_price
    for i in range(n):
        o = price
        c = o * (1 + rng.gauss(drift, volatility))
        h = max(o, c) * (1 + abs(rng.gauss(0, volatility / 2)))
        low = min(o, c) * (1 - abs(rng.gauss(0, volatility / 2)))
        rows.append([start_time_ms + i * interval_ms, round(o, 2), round(h, 2), round(low, 2), round(c, 2), 0.0])
        price = c
    return rows
