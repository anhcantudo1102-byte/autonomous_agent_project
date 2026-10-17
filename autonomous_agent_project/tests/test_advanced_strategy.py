"""Kiểm thử phân tích đa khung (SMC + Elliott heuristic + Hidden Base) — không cần mạng."""

from __future__ import annotations

import contextlib
import io
import random
import unittest
from unittest import mock

import advanced_demo
from core import price_feed
from core.advanced_strategy import (
    BEARISH,
    BULLISH,
    BUY,
    CORRECTIVE,
    HOLD,
    NEUTRAL,
    SELL,
    UNCLEAR,
    WAVE_3,
    WAVE_5,
    MultiDimensionalAnalyzer,
    decide,
    generate_synthetic_klines,
)

DUMMY_HTF = [[0, 80000, 82000, 79000, 81500]] * 15
DUMMY_LTF = [[0, 81000, 81600, 80900, 81400]] * 10


def k(o: float, h: float, low: float, c: float) -> list[float]:
    return [0, o, h, low, c]


def from_path(prices: list[float], pad: float = 0.2) -> list[list[float]]:
    """Nến nối tiếp theo đường giá: open = close trước, râu nến ``pad`` mỗi phía."""
    rows = []
    for prev, cur in zip(prices, prices[1:]):
        rows.append(k(prev, max(prev, cur) + pad, min(prev, cur) - pad, cur))
    return rows


def closes_path(closes: list[float]) -> list[list[float]]:
    return [k(c, c, c, c) for c in closes]


class TestSmcStructure(unittest.TestCase):
    def test_dummy_goc_phang_nen_neutral(self) -> None:
        a = MultiDimensionalAnalyzer()
        self.assertEqual(a._check_smc_structure(DUMMY_HTF), NEUTRAL)

    def test_xu_huong_tang_va_giam(self) -> None:
        a = MultiDimensionalAnalyzer()
        up = generate_synthetic_klines(60, drift=0.01, volatility=0.004, rng=random.Random(1))
        down = generate_synthetic_klines(60, drift=-0.01, volatility=0.004, rng=random.Random(1))
        self.assertEqual(a._check_smc_structure(up), BULLISH)
        self.assertEqual(a._check_smc_structure(down), BEARISH)

    def test_hh_hl_khong_can_close_cao_nhat(self) -> None:
        # Close cuối (106) thấp hơn đỉnh 107 nên luật gốc `close > max(...)` trả NEUTRAL,
        # nhưng cấu trúc đỉnh/đáy cao dần vẫn là xu hướng tăng.
        path = [97, 98, 99, 100, 103, 101, 105, 103, 107, 105, 106]
        candles = closes_path(path)
        closes = [c[4] for c in candles]
        self.assertFalse(closes[-1] > max(closes[:-1]))
        self.assertEqual(MultiDimensionalAnalyzer(swing_k=1)._check_smc_structure(candles), BULLISH)

    def test_thieu_nen(self) -> None:
        self.assertEqual(MultiDimensionalAnalyzer()._check_smc_structure(DUMMY_HTF[:9]), NEUTRAL)
        self.assertEqual(MultiDimensionalAnalyzer()._check_smc_structure([]), NEUTRAL)


class TestPriceAction(unittest.TestCase):
    def setUp(self) -> None:
        self.a = MultiDimensionalAnalyzer(base_len=4, base_max_range=0.004)
        self.filler = [k(100, 101, 99, 100.5)] * 3

    def test_doji_thi_cho(self) -> None:
        ltf = self.filler + [k(100, 101, 99, 99.0), k(100, 102, 98, 100.1)]
        self.assertEqual(self.a._evaluate_price_action_and_bases(ltf), HOLD)

    def test_nen_nhan_chim(self) -> None:
        bull = self.filler + [k(101, 101.2, 99.8, 100), k(99.8, 101.8, 99.6, 101.5)]
        bear = self.filler + [k(100, 101.2, 99.8, 101), k(101.2, 101.4, 99.0, 99.5)]
        self.assertEqual(self.a._evaluate_price_action_and_bases(bull), BUY)
        self.assertEqual(self.a._evaluate_price_action_and_bases(bear), SELL)

    def test_pha_vo_hidden_base(self) -> None:
        base = [k(100, 100.15, 99.95, 100.05), k(100.05, 100.2, 100, 100.1),
                k(100.1, 100.18, 99.98, 100.0), k(100, 100.12, 99.96, 100.08)]
        up = [k(100, 100.5, 99, 99.5)] + base + [k(100.08, 100.6, 100.05, 100.5)]
        down = [k(100, 100.5, 99, 99.5)] + base + [k(100.0, 100.05, 99.5, 99.6)]
        self.assertEqual(self.a._evaluate_price_action_and_bases(up), BUY)
        self.assertIn("Hidden Base", self.a.reasons[-1])
        self.assertEqual(self.a._evaluate_price_action_and_bases(down), SELL)

    def test_thieu_nen(self) -> None:
        self.assertEqual(self.a._evaluate_price_action_and_bases(DUMMY_LTF[:4]), HOLD)


class TestElliottHeuristic(unittest.TestCase):
    def setUp(self) -> None:
        self.a = MultiDimensionalAnalyzer()

    def test_tang_toc_la_song_3(self) -> None:
        closes = [100 * 1.002**i for i in range(15)] + [100 * 1.002**14 * 1.01**i for i in range(1, 7)]
        self.assertEqual(self.a._count_elliott_waves(closes_path(closes)), (WAVE_3, BULLISH))

    def test_cham_lai_la_song_5(self) -> None:
        closes = [100 * 1.01**i for i in range(15)] + [100 * 1.01**14 * 1.001**i for i in range(1, 7)]
        self.assertEqual(self.a._count_elliott_waves(closes_path(closes)), (WAVE_5, BULLISH))

    def test_nguoc_chieu_la_dieu_chinh(self) -> None:
        closes = [100 * 1.01**i for i in range(15)] + [100 * 1.01**14 * 0.995**i for i in range(1, 7)]
        status, direction = self.a._count_elliott_waves(closes_path(closes))
        self.assertEqual((status, direction), (CORRECTIVE, BULLISH))

    def test_phang_hoac_thieu_du_lieu(self) -> None:
        self.assertEqual(self.a._count_elliott_waves(DUMMY_HTF), (UNCLEAR, NEUTRAL))
        self.assertEqual(self.a._count_elliott_waves(DUMMY_HTF[:3]), (UNCLEAR, NEUTRAL))


class TestConfluence(unittest.TestCase):
    def score(self, trend: str, signal: str, wave: tuple[str, str]) -> float:
        a = MultiDimensionalAnalyzer()
        with mock.patch.object(a, "_check_smc_structure", return_value=trend), \
                mock.patch.object(a, "_evaluate_price_action_and_bases", return_value=signal), \
                mock.patch.object(a, "_count_elliott_waves", return_value=wave):
            return a.analyze_market_structure([], [])["confluence_score"]

    def test_diem_hop_luu(self) -> None:
        self.assertEqual(self.score(BULLISH, BUY, (WAVE_5, BULLISH)), 2)
        self.assertEqual(self.score(BULLISH, BUY, (WAVE_3, BULLISH)), 3.0)
        self.assertEqual(self.score(BEARISH, SELL, (WAVE_3, BEARISH)), -3.0)
        self.assertEqual(self.score(BULLISH, BUY, (WAVE_3, BEARISH)), 2)  # sóng 3 ngược chiều không khuếch đại
        self.assertEqual(self.score(BULLISH, SELL, (WAVE_3, BULLISH)), 0)
        self.assertEqual(self.score(NEUTRAL, BUY, (WAVE_3, BULLISH)), 0)

    def test_dummy_goc_luon_wait(self) -> None:
        r = MultiDimensionalAnalyzer().analyze_market_structure(DUMMY_HTF, DUMMY_LTF)
        self.assertEqual((r["trend"], r["signal"], r["confluence_score"]), (NEUTRAL, BUY, 0))
        self.assertEqual(decide(r["confluence_score"]), "WAIT")
        self.assertTrue(r["reasons"])

    def test_quyet_dinh(self) -> None:
        self.assertEqual([decide(3.0), decide(-2), decide(0)], ["LONG", "SHORT", "WAIT"])


@unittest.skipIf(price_feed.requests is None, "chưa cài requests")
class TestFetchKlines(unittest.TestCase):
    class Resp:
        def __init__(self, status: int, payload: object) -> None:
            self.status_code, self._payload = status, payload

        def json(self) -> object:
            return self._payload

    def test_bi_chan_vung_thi_dung_endpoint_du_phong(self) -> None:
        rows = [[1, "100", "101", "99", "100.5", "10"]]
        replies = [self.Resp(451, {"msg": "restricted"}), self.Resp(200, rows)]
        with mock.patch.object(price_feed.requests, "get", side_effect=replies) as get:
            data = price_feed.fetch_klines("btcusdt", "1h", 1, report=lambda _: None)
        self.assertEqual(data, rows)
        self.assertEqual(get.call_args.kwargs["params"], {"symbol": "BTCUSDT", "interval": "1h", "limit": 1})
        self.assertIn("data-api.binance.vision", get.call_args.args[0])

    def test_moi_endpoint_loi(self) -> None:
        bad = self.Resp(400, {"msg": "Invalid symbol."})
        with mock.patch.object(price_feed.requests, "get", return_value=bad):
            self.assertIsNone(price_feed.fetch_klines("KHONGCO", report=lambda _: None))


class TestDemo(unittest.TestCase):
    def run_demo(self, argv: list[str]) -> tuple[dict, str]:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = advanced_demo.run_advanced_agent_simulation(argv)
        return result, out.getvalue()

    def test_dummy_goc(self) -> None:
        result, out = self.run_demo(["--dummy-goc"])
        self.assertEqual(result["decision"], "WAIT")
        self.assertIn("WAIT FOR SETUP", out)

    def test_live_loi_thi_dung_gia_lap(self) -> None:
        with mock.patch.object(advanced_demo, "fetch_klines", return_value=None):
            result, out = self.run_demo(["--live", "--seed", "7"])
        self.assertIn("dùng nến GIẢ LẬP", out)
        self.assertIn(result["decision"], {"LONG", "SHORT", "WAIT"})

    def test_live_dung_nen_that(self) -> None:
        htf = generate_synthetic_klines(60, drift=0.01, volatility=0.003, rng=random.Random(2))
        with mock.patch.object(advanced_demo, "fetch_klines", side_effect=[htf, htf[-30:]]):
            result, out = self.run_demo(["--live"])
        self.assertIn("Binance BTCUSDT thật", out)
        self.assertEqual(result["trend"], BULLISH)


if __name__ == "__main__":
    unittest.main()
