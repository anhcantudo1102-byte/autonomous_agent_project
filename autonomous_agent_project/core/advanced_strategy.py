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
from typing import Any, Callable, Sequence

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


# ======================================================================
# AdvancedMarketAnalyzer — API đa khung theo thiết kế của người dùng
# ======================================================================
SMC_NONE = "NONE"
SMC_CONSOLIDATION = "CONSOLIDATION"
BOS_UP, BOS_DOWN = "BOS_UP", "BOS_DOWN"
CHOCH_UP, CHOCH_DOWN = "CHOCH_UP", "CHOCH_DOWN"
BUY_FROM_BASE, SELL_FROM_BASE = "BUY_FROM_BASE", "SELL_FROM_BASE"
WAVE_3_IMPULSE = "WAVE_3_IMPULSE"
WAVE_5_IMPULSE = "WAVE_5_IMPULSE"
CORRECTIVE_WAVE = "CORRECTIVE_WAVE"
WAVE_UNKNOWN = "WAVE_UNKNOWN"
STRONG_BUY, STRONG_SELL = "STRONG_BUY", "STRONG_SELL"

_WAVE_NAMES = {WAVE_3: WAVE_3_IMPULSE, WAVE_5: WAVE_5_IMPULSE, CORRECTIVE: CORRECTIVE_WAVE, UNCLEAR: WAVE_UNKNOWN}

TREND_POINTS = 2.0
BASE_POINTS = 3.0
ORDER_BLOCK_POINTS = 1.0


class AdvancedMarketAnalyzer:
    """Phân tích đa khung: SMC (BOS/CHoCH + Order Block) trên HTF, Hidden Base trên LTF, Elliott (heuristic).

    Điểm hợp lưu:

    * xu hướng HTF: BULLISH +2, BEARISH −2;
    * Hidden Base phá vỡ **cùng chiều** xu hướng HTF: ±3;
    * giá LTF đang nằm trong Order Block cùng chiều xu hướng (hồi về vùng OB): ±1;
    * ×1.5 chỉ khi "sóng 3" (heuristic động lượng) **cùng chiều** với dấu của điểm.

    Elliott ở đây KHÔNG đếm sóng thật: dùng ``MultiDimensionalAnalyzer._count_elliott_waves``
    (so động lượng ngắn hạn với trung hạn). Công cụ minh hoạ, không phải lời khuyên đầu tư.
    """

    MIN_HTF_CANDLES = 10
    MIN_LTF_CANDLES = 5
    MIN_WAVE_CANDLES = 15

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        *,
        swing_k: int = 2,
        base_len: int = 4,
        breakout_ratio: float = 1.8,
        base_max_range: float = 0.006,
        order_block_lookback: int = 10,
    ) -> None:
        self.symbol = symbol.upper()
        self.swing_k = swing_k
        self.base_len = base_len
        self.breakout_ratio = breakout_ratio
        self.base_max_range = base_max_range
        self.order_block_lookback = order_block_lookback
        self._waves = MultiDimensionalAnalyzer(symbol, swing_k=swing_k)
        self.last_wave_direction = NEUTRAL
        self.reasons: list[str] = []

    # ------------------------------------------------------------ Tổng hợp
    def analyze_multi_timeframe(self, htf_klines: Sequence[Kline], ltf_klines: Sequence[Kline]) -> dict[str, Any]:
        """HTF: xu hướng, BOS/CHoCH, Order Block. LTF: Hidden Base. Trả về điểm hợp lưu + hành động."""
        self.reasons = []
        htf_structure = self.detect_smc_structure(htf_klines)
        ltf_bases = self.detect_hidden_bases(ltf_klines)
        wave_status = self.validate_elliott_waves(htf_klines)
        trend = htf_structure["trend"]

        score = 0.0
        if trend == BULLISH:
            score += TREND_POINTS
        elif trend == BEARISH:
            score -= TREND_POINTS

        if ltf_bases["signal"] == BUY_FROM_BASE and trend == BULLISH:
            score += BASE_POINTS
        elif ltf_bases["signal"] == SELL_FROM_BASE and trend == BEARISH:
            score -= BASE_POINTS

        ob = htf_structure.get("order_block")
        ltf = _candles(ltf_klines)
        if ob and ltf:
            lo, hi = ob["zone"]
            price = ltf[-1].close
            if lo <= price <= hi:
                if ob["type"] == "BULLISH_OB" and trend == BULLISH:
                    score += ORDER_BLOCK_POINTS
                    self.reasons.append(f"Giá {price:,.6g} hồi về Order Block tăng {lo:,.6g}–{hi:,.6g}")
                elif ob["type"] == "BEARISH_OB" and trend == BEARISH:
                    score -= ORDER_BLOCK_POINTS
                    self.reasons.append(f"Giá {price:,.6g} hồi về Order Block giảm {lo:,.6g}–{hi:,.6g}")

        wave_dir = self.last_wave_direction
        if wave_status == WAVE_3_IMPULSE and score != 0 and (score > 0) == (wave_dir == BULLISH):
            score *= WAVE_3_MULTIPLIER
            self.reasons.append("Sóng 3 cùng chiều → ×1.5")

        return {
            "htf_trend": trend,
            "market_structure_event": htf_structure["event"],
            "order_block": ob,
            "hidden_base": ltf_bases,
            "elliott_wave": wave_status,
            "wave_direction": wave_dir,
            "confluence_score": round(score, 2),
            "recommended_action": self._decide_action(score),
            "reasons": list(self.reasons),
        }

    # ------------------------------------------------------------ HTF: SMC
    def detect_smc_structure(self, klines: Sequence[Kline]) -> dict[str, Any]:
        """BOS / CHoCH dựa trên đỉnh-đáy swing (fractal), kèm Order Block của nhịp phá vỡ.

        * Cấu trúc trước đó (hai swing gần nhất) là HH/HL (tăng) hoặc LH/LL (giảm).
        * Close cuối vượt đỉnh swing gần nhất → ``BOS_UP`` nếu cấu trúc trước đó không phải giảm,
          ngược lại ``CHOCH_UP`` (đổi tính chất: giảm → tăng). Tương tự cho chiều xuống.
        * Chưa phá: xu hướng theo HH/HL hoặc LH/LL, sự kiện ``CONSOLIDATION``.
        * Không có swing: so close cuối với vùng giá nửa đầu dữ liệu (thay cho
          ``max(highs[:-3])`` của bản gốc, vốn gồm cả nến đang phá nên khó xảy ra).
        """
        candles = _candles(klines)
        if len(candles) < self.MIN_HTF_CANDLES:
            self.reasons.append(f"HTF chỉ có {len(candles)} nến (< {self.MIN_HTF_CANDLES})")
            return {"trend": NEUTRAL, "event": SMC_NONE, "order_block": None}
        last = candles[-1].close
        highs, lows = find_swings(candles[:-1], self.swing_k)
        prior = NEUTRAL
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                prior = BULLISH
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                prior = BEARISH

        if highs and last > highs[-1]:
            event = CHOCH_UP if prior == BEARISH else BOS_UP
            self.reasons.append(f"HTF {event}: close {last:,.6g} vượt đỉnh swing {highs[-1]:,.6g}")
            return {"trend": BULLISH, "event": event, "order_block": self._order_block(candles, BULLISH)}
        if lows and last < lows[-1]:
            event = CHOCH_DOWN if prior == BULLISH else BOS_DOWN
            self.reasons.append(f"HTF {event}: close {last:,.6g} thủng đáy swing {lows[-1]:,.6g}")
            return {"trend": BEARISH, "event": event, "order_block": self._order_block(candles, BEARISH)}
        if prior != NEUTRAL:
            self.reasons.append(f"HTF {'HH/HL' if prior == BULLISH else 'LH/LL'}, chưa phá cấu trúc")
            return {"trend": prior, "event": SMC_CONSOLIDATION, "order_block": self._order_block(candles, prior)}

        if not highs and not lows:
            first_half = candles[: len(candles) // 2]
            if last > max(c.high for c in first_half):
                self.reasons.append("HTF không có swing, giá vượt vùng cao nửa đầu dữ liệu → BOS_UP")
                return {"trend": BULLISH, "event": BOS_UP, "order_block": self._order_block(candles, BULLISH)}
            if last < min(c.low for c in first_half):
                self.reasons.append("HTF không có swing, giá thủng vùng thấp nửa đầu dữ liệu → BOS_DOWN")
                return {"trend": BEARISH, "event": BOS_DOWN, "order_block": self._order_block(candles, BEARISH)}
        self.reasons.append("HTF đi ngang, chưa rõ cấu trúc")
        return {"trend": NEUTRAL, "event": SMC_CONSOLIDATION, "order_block": None}

    def _order_block(self, candles: Sequence[Candle], trend: str) -> dict[str, Any] | None:
        """Order Block đơn giản: nến ngược màu cuối cùng trước nhịp đẩy (trong ``order_block_lookback`` nến).

        Xu hướng tăng → nến giảm cuối cùng (vùng cầu); xu hướng giảm → nến tăng cuối cùng (vùng cung).
        """
        window = list(candles[-self.order_block_lookback - 1 : -1])
        for c in reversed(window):
            if trend == BULLISH and c.close < c.open:
                return {"type": "BULLISH_OB", "zone": (c.low, c.high)}
            if trend == BEARISH and c.close > c.open:
                return {"type": "BEARISH_OB", "zone": (c.low, c.high)}
        return None

    # ------------------------------------------------------- LTF: Hidden Base
    def detect_hidden_bases(self, klines: Sequence[Kline]) -> dict[str, Any]:
        """Nền giá chặt (``base_len`` nến) rồi nến cuối bứt phá mạnh ra khỏi nền.

        Khác bản gốc: biên độ trung bình chỉ tính trên các nến **nền** (không gồm nến bứt phá),
        nến cuối phải đóng cửa **ra ngoài** vùng nền, và nền phải đủ chặt (≤ ``base_max_range``).
        """
        candles = _candles(klines)
        if len(candles) < max(self.MIN_LTF_CANDLES, self.base_len + 1):
            return {"signal": HOLD, "zone": (0, 0)}
        base = candles[-1 - self.base_len : -1]
        last = candles[-1]
        lo, hi = min(c.low for c in base), max(c.high for c in base)
        avg_range = sum(c.range for c in base) / len(base)
        tight = lo > 0 and (hi - lo) / lo <= self.base_max_range
        strong = last.range > 0 and (avg_range == 0 or last.range > avg_range * self.breakout_ratio)
        if tight and strong:
            zone = (round(lo, 8), round(hi, 8))
            if last.close > hi and last.close > last.open:
                self.reasons.append(f"LTF: bứt phá tăng khỏi Hidden Base {lo:,.6g}–{hi:,.6g}")
                return {"signal": BUY_FROM_BASE, "zone": zone}
            if last.close < lo and last.close < last.open:
                self.reasons.append(f"LTF: bứt phá giảm khỏi Hidden Base {lo:,.6g}–{hi:,.6g}")
                return {"signal": SELL_FROM_BASE, "zone": zone}
        return {"signal": HOLD, "zone": (0, 0)}

    # ---------------------------------------------------- Elliott (heuristic)
    def validate_elliott_waves(self, klines: Sequence[Kline]) -> str:
        """Pha sóng theo heuristic động lượng (không phải đếm sóng Elliott thật).

        Trả về ``WAVE_3_IMPULSE`` / ``WAVE_5_IMPULSE`` / ``CORRECTIVE_WAVE`` / ``WAVE_UNKNOWN``;
        chiều của nhịp lưu ở ``last_wave_direction``.
        """
        if not klines or len(klines) < self.MIN_WAVE_CANDLES:
            self.last_wave_direction = NEUTRAL
            return WAVE_UNKNOWN
        self._waves.reasons = []
        status, direction = self._waves._count_elliott_waves(klines)
        self.reasons.extend(self._waves.reasons)
        self.last_wave_direction = direction
        return _WAVE_NAMES[status]

    def _decide_action(self, score: float) -> str:
        if score >= 3.0:
            return STRONG_BUY
        if score > 0:
            return "BUY"
        if score <= -3.0:
            return STRONG_SELL
        if score < 0:
            return "SELL"
        return HOLD


class CandleBuilder:
    """Gom chuỗi giá tick thành nến OHLC (``ticks_per_candle`` tick mỗi nến) theo định dạng kline.

    Thị trường giả lập chỉ sinh giá đóng cửa mỗi tick, nên open/high/low/close của một nến được
    lấy từ các tick trong nến đó. Nến đang hình thành được nối vào cuối danh sách.
    """

    def __init__(self, ticks_per_candle: int, max_candles: int = 60) -> None:
        if ticks_per_candle < 1:
            raise ValueError("ticks_per_candle phải ≥ 1")
        self.ticks_per_candle = ticks_per_candle
        self.max_candles = max_candles
        self.candles: list[list[float]] = []
        self.current: list[float] | None = None
        self.count = 0

    def add(self, tick: int, price: float) -> None:
        if self.current is None:
            self.current = [float(tick), price, price, price, price, 0.0]
            self.count = 0
        c = self.current
        c[2] = max(c[2], price)
        c[3] = min(c[3], price)
        c[4] = price
        self.count += 1
        if self.count >= self.ticks_per_candle:
            self.candles.append(c)
            del self.candles[: -self.max_candles]
            self.current = None

    def klines(self) -> list[list[float]]:
        rows = [list(c) for c in self.candles]
        if self.current is not None:
            rows.append(list(self.current))
        return rows[-self.max_candles :]

    def to_state(self) -> dict[str, Any]:
        return {"ticks_per_candle": self.ticks_per_candle, "candles": [list(c) for c in self.candles],
                "current": list(self.current) if self.current else None, "count": self.count}

    def load_state(self, state: dict[str, Any]) -> None:
        self.candles = [[float(x) for x in c] for c in state.get("candles", [])][-self.max_candles :]
        cur = state.get("current")
        self.current = [float(x) for x in cur] if cur else None
        self.count = int(state.get("count", 0))


KlineFetcher = Callable[[str, int], "list[list[Any]] | None"]  # (interval, limit) → klines


class ConfluenceFeed:
    """Cấp điểm hợp lưu đa khung cho mỗi tick của mô phỏng.

    * Mặc định dựng nến từ chuỗi giá tick: HTF = ``htf_ticks`` tick/nến, LTF = ``ltf_ticks`` tick/nến.
    * Nếu có ``fetch`` (chế độ giá thật), lấy nến Binance ``htf_interval``/``ltf_interval`` và làm mới
      sau mỗi ``refresh_htf``/``refresh_ltf`` tick; lấy lỗi thì dùng nến gom từ tick.
    """

    def __init__(
        self,
        analyzer: AdvancedMarketAnalyzer | None = None,
        *,
        htf_ticks: int = 8,
        ltf_ticks: int = 2,
        htf_max: int = 60,
        ltf_max: int = 30,
        fetch: KlineFetcher | None = None,
        htf_interval: str = "1h",
        ltf_interval: str = "1m",
        refresh_htf: int = 30,
        refresh_ltf: int = 5,
    ) -> None:
        self.analyzer = analyzer or AdvancedMarketAnalyzer()
        self.htf = CandleBuilder(htf_ticks, htf_max)
        self.ltf = CandleBuilder(ltf_ticks, ltf_max)
        self.fetch = fetch
        self.htf_interval, self.ltf_interval = htf_interval, ltf_interval
        self.refresh_htf, self.refresh_ltf = refresh_htf, refresh_ltf
        self._remote: dict[str, tuple[int, list[list[Any]]]] = {}
        self.last: dict[str, Any] | None = None
        self.source = "nến gom từ tick"

    def _remote_klines(self, key: str, tick: int, interval: str, limit: int, every: int) -> list[list[Any]] | None:
        cached = self._remote.get(key)
        if cached is not None and tick - cached[0] < every:
            return cached[1]
        rows = self.fetch(interval, limit) if self.fetch else None
        if rows:
            self._remote[key] = (tick, rows)
            return rows
        self._remote.pop(key, None)
        return None

    def update(self, tick: int, price: float) -> dict[str, Any]:
        self.htf.add(tick, price)
        self.ltf.add(tick, price)
        htf = ltf = None
        if self.fetch is not None:
            htf = self._remote_klines("htf", tick, self.htf_interval, self.htf.max_candles, self.refresh_htf)
            ltf = self._remote_klines("ltf", tick, self.ltf_interval, self.ltf.max_candles, self.refresh_ltf)
        remote = htf is not None and ltf is not None
        self.source = f"Binance {self.htf_interval}/{self.ltf_interval}" if remote else "nến gom từ tick"
        if not remote:
            htf, ltf = self.htf.klines(), self.ltf.klines()
        result = self.analyzer.analyze_multi_timeframe(htf, ltf)
        result["source"] = self.source
        result["tick"] = tick
        self.last = result
        return result

    def to_state(self) -> dict[str, Any]:
        return {"htf": self.htf.to_state(), "ltf": self.ltf.to_state(), "last": self.last}

    def load_state(self, state: dict[str, Any]) -> None:
        self.htf.load_state(state.get("htf", {}))
        self.ltf.load_state(state.get("ltf", {}))
        self.last = state.get("last")


def describe_analysis(result: dict[str, Any] | None) -> str:
    """Một dòng tóm tắt kết quả phân tích để in log."""
    if not result:
        return "đa khung: chưa có dữ liệu"
    base = result["hidden_base"]["signal"]
    return (
        f"đa khung {result['htf_trend']}/{result['market_structure_event']}, base {base}, "
        f"{result['elliott_wave']}, điểm {result['confluence_score']:+.1f} → {result['recommended_action']}"
    )
