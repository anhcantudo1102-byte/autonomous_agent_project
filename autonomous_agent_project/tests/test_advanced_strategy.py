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
    AdvancedMarketAnalyzer,
    CandleBuilder,
    ConfluenceFeed,
    MultiDimensionalAnalyzer,
    decide,
    describe_analysis,
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


class TestAdvancedMarketAnalyzer(unittest.TestCase):
    """Ba bài kiểm thử gốc của người dùng (giữ nguyên)."""

    def setUp(self):
        self.analyzer = AdvancedMarketAnalyzer("BTCUSDT")
        # Giả lập dữ liệu nến klines [timestamp, open, high, low, close, volume]
        self.dummy_htf = [[i, 80000, 81000, 79000, 80500 + i * 100, 1000] for i in range(15)]
        self.dummy_ltf = [[i, 80500, 82000, 80400, 81800, 500] for i in range(10)]

    def test_smc_structure(self):
        result = self.analyzer.detect_smc_structure(self.dummy_htf)
        self.assertIn(result["trend"], ["BULLISH", "BEARISH", "NEUTRAL"])

    def test_hidden_bases(self):
        result = self.analyzer.detect_hidden_bases(self.dummy_ltf)
        self.assertIn(result["signal"], ["BUY_FROM_BASE", "SELL_FROM_BASE", "HOLD"])

    def test_multi_timeframe_analysis(self):
        analysis = self.analyzer.analyze_multi_timeframe(self.dummy_htf, self.dummy_ltf)
        self.assertIn("confluence_score", analysis)
        self.assertIn("recommended_action", analysis)


def zigzag(points: list[float], n: int = 3) -> list[float]:
    """Nối các điểm bằng ``n`` bước thẳng để tạo đỉnh/đáy swing rõ ràng."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        out += [a + (b - a) * i / n for i in range(1, n + 1)]
    return out


BASE = [k(100, 100.1, 99.9, 100.05), k(100.05, 100.1, 99.9, 99.95),
        k(99.95, 100.1, 99.9, 100.0), k(100.0, 100.1, 99.9, 100.02)]
FILLER = [k(99, 99.5, 98.5, 99.2)]
UP_PATH = zigzag([100, 110, 105, 115, 110, 120, 116]) + [118, 122]
DOWN_PATH = zigzag([120, 110, 115, 105, 110, 100, 104]) + [99, 96]


class TestAdvancedMarketAnalyzerCuThe(unittest.TestCase):
    """Khẳng định cụ thể cho AdvancedMarketAnalyzer (không chỉ kiểm tra kiểu giá trị)."""

    def setUp(self) -> None:
        self.a = AdvancedMarketAnalyzer("btcusdt")

    def test_dummy_nguoi_dung_cho_ket_qua_xac_dinh(self) -> None:
        htf = [[i, 80000, 81000, 79000, 80500 + i * 100, 1000] for i in range(15)]
        ltf = [[i, 80500, 82000, 80400, 81800, 500] for i in range(10)]
        smc = self.a.detect_smc_structure(htf)
        self.assertEqual((smc["trend"], smc["event"]), (BULLISH, "BOS_UP"))
        self.assertEqual(self.a.detect_hidden_bases(ltf), {"signal": HOLD, "zone": (0, 0)})
        r = self.a.analyze_multi_timeframe(htf, ltf)
        self.assertEqual((r["confluence_score"], r["recommended_action"]), (2.0, "BUY"))
        self.assertEqual(self.a.symbol, "BTCUSDT")

    def test_bos_theo_xu_huong(self) -> None:
        up = self.a.detect_smc_structure(from_path(UP_PATH))
        down = self.a.detect_smc_structure(from_path(DOWN_PATH))
        self.assertEqual((up["trend"], up["event"]), (BULLISH, "BOS_UP"))
        self.assertEqual((down["trend"], down["event"]), (BEARISH, "BOS_DOWN"))

    def test_choch_khi_dao_chieu_cau_truc(self) -> None:
        down_then_up = zigzag([120, 110, 115, 105, 110, 100, 104]) + [108, 112]
        up_then_down = zigzag([100, 110, 105, 115, 110, 120, 116]) + [112, 108]
        r1 = self.a.detect_smc_structure(from_path(down_then_up))
        r2 = self.a.detect_smc_structure(from_path(up_then_down))
        self.assertEqual((r1["trend"], r1["event"]), (BULLISH, "CHOCH_UP"))
        self.assertEqual((r2["trend"], r2["event"]), (BEARISH, "CHOCH_DOWN"))

    def test_chua_pha_cau_truc_la_tich_luy(self) -> None:
        r = self.a.detect_smc_structure(from_path(zigzag([100, 110, 105, 115, 110, 120, 117])))
        self.assertEqual((r["trend"], r["event"]), (BULLISH, "CONSOLIDATION"))
        flat = self.a.detect_smc_structure(DUMMY_HTF)
        self.assertEqual((flat["trend"], flat["event"], flat["order_block"]), (NEUTRAL, "CONSOLIDATION", None))

    def test_order_block_la_nen_nguoc_mau_cuoi_cung(self) -> None:
        candles = from_path(UP_PATH)
        ob = self.a.detect_smc_structure(candles)["order_block"]
        last_bear = [c for c in candles[:-1] if c[4] < c[1]][-1]
        self.assertEqual(ob, {"type": "BULLISH_OB", "zone": (last_bear[3], last_bear[2])})
        bear_ob = self.a.detect_smc_structure(from_path(DOWN_PATH))["order_block"]
        self.assertEqual(bear_ob["type"], "BEARISH_OB")

    def test_it_nen_htf(self) -> None:
        r = self.a.detect_smc_structure(from_path(UP_PATH)[:5])
        self.assertEqual((r["trend"], r["event"], r["order_block"]), (NEUTRAL, "NONE", None))

    def test_pha_base_tang_va_giam(self) -> None:
        up = FILLER + BASE + [k(100.02, 100.6, 100.0, 100.5)]
        down = FILLER + BASE + [k(100.0, 100.02, 99.4, 99.5)]
        self.assertEqual(self.a.detect_hidden_bases(up), {"signal": "BUY_FROM_BASE", "zone": (99.9, 100.1)})
        self.assertEqual(self.a.detect_hidden_bases(down), {"signal": "SELL_FROM_BASE", "zone": (99.9, 100.1)})

    def test_bien_do_trung_binh_khong_tinh_nen_pha_vo(self) -> None:
        # Nền biên độ 0.2, nến phá biên độ 0.4 (= 2 lần): đạt ngưỡng 1.8 lần nếu chỉ tính nền,
        # nhưng sẽ trượt nếu (như bản gốc) gộp cả nến cuối vào trung bình (cần > 2.25 lần).
        breakout = k(100.08, 100.45, 100.05, 100.4)
        self.assertEqual(self.a.detect_hidden_bases(FILLER + BASE + [breakout])["signal"], "BUY_FROM_BASE")

    def test_base_khong_pha_hoac_nen_long_thi_hold(self) -> None:
        inside = FILLER + BASE + [k(99.9, 100.3, 99.7, 100.05)]  # nến dài nhưng đóng trong nền
        loose = FILLER + [k(100, 101, 99, 100.5)] * 4 + [k(100.5, 104, 100.4, 103.8)]  # nền quá rộng
        self.assertEqual(self.a.detect_hidden_bases(inside)["signal"], HOLD)
        self.assertEqual(self.a.detect_hidden_bases(loose)["signal"], HOLD)
        self.assertEqual(self.a.detect_hidden_bases(BASE)["signal"], HOLD)

    def test_song_elliott_khong_con_co_dinh(self) -> None:
        accel = [100 * 1.002**i for i in range(15)] + [100 * 1.002**14 * 1.01**i for i in range(1, 7)]
        slow = [100 * 1.01**i for i in range(15)] + [100 * 1.01**14 * 1.001**i for i in range(1, 7)]
        back = [100 * 1.01**i for i in range(15)] + [100 * 1.01**14 * 0.995**i for i in range(1, 7)]
        waves = [self.a.validate_elliott_waves(closes_path(c)) for c in (accel, slow, back)]
        self.assertEqual(waves, ["WAVE_3_IMPULSE", "WAVE_5_IMPULSE", "CORRECTIVE_WAVE"])
        self.assertEqual(self.a.validate_elliott_waves(DUMMY_HTF[:10]), "WAVE_UNKNOWN")

    def score(self, trend: str, base: str, wave: str, wave_dir: str) -> dict:
        def fake_wave(_klines: object) -> str:
            self.a.last_wave_direction = wave_dir
            return wave

        with mock.patch.object(self.a, "detect_smc_structure",
                               return_value={"trend": trend, "event": "X", "order_block": None}), \
                mock.patch.object(self.a, "detect_hidden_bases", return_value={"signal": base, "zone": (0, 0)}), \
                mock.patch.object(self.a, "validate_elliott_waves", side_effect=fake_wave):
            return self.a.analyze_multi_timeframe([], [])

    def test_he_so_song_3_chi_khi_cung_chieu(self) -> None:
        self.assertEqual(self.score(BULLISH, "BUY_FROM_BASE", "WAVE_3_IMPULSE", BULLISH)["confluence_score"], 7.5)
        self.assertEqual(self.score(BEARISH, "SELL_FROM_BASE", "WAVE_3_IMPULSE", BEARISH)["confluence_score"], -7.5)
        # Bản gốc nhân ×1.5 bất kể chiều: điểm giảm −2 thành −3 khi sóng 3 TĂNG. Nay giữ −2.
        self.assertEqual(self.score(BEARISH, HOLD, "WAVE_3_IMPULSE", BULLISH)["confluence_score"], -2.0)
        self.assertEqual(self.score(BULLISH, HOLD, "WAVE_5_IMPULSE", BULLISH)["confluence_score"], 2.0)
        # Base ngược chiều xu hướng HTF không được cộng điểm.
        self.assertEqual(self.score(BULLISH, "SELL_FROM_BASE", "WAVE_UNKNOWN", NEUTRAL)["confluence_score"], 2.0)
        self.assertEqual(self.score(NEUTRAL, "BUY_FROM_BASE", "WAVE_3_IMPULSE", BULLISH)["confluence_score"], 0.0)

    def test_xu_huong_tang_va_pha_base_la_strong_buy(self) -> None:
        r = self.a.analyze_multi_timeframe(from_path(UP_PATH), FILLER + BASE + [k(100.02, 100.6, 100.0, 100.5)])
        self.assertEqual((r["htf_trend"], r["hidden_base"]["signal"]), (BULLISH, "BUY_FROM_BASE"))
        self.assertEqual((r["confluence_score"], r["recommended_action"]), (5.0, "STRONG_BUY"))
        self.assertTrue(r["reasons"])

    def test_gia_hoi_ve_order_block_cong_diem(self) -> None:
        htf = from_path(UP_PATH)
        lo, hi = self.a.detect_smc_structure(htf)["order_block"]["zone"]
        mid = (lo + hi) / 2
        ltf = [k(mid, mid + 0.1, mid - 0.1, mid)] * 5
        r = self.a.analyze_multi_timeframe(htf, ltf)
        self.assertEqual(r["confluence_score"], 3.0)
        self.assertTrue(any("Order Block" in s for s in r["reasons"]))

    def test_nguong_hanh_dong(self) -> None:
        d = self.a._decide_action
        self.assertEqual([d(4.5), d(3.0), d(2.0), d(0.0), d(-1.0), d(-3.0)],
                         ["STRONG_BUY", "STRONG_BUY", "BUY", HOLD, "SELL", "STRONG_SELL"])

    def test_mo_ta_mot_dong(self) -> None:
        r = self.a.analyze_multi_timeframe(from_path(UP_PATH), FILLER + BASE + [k(100.02, 100.6, 100.0, 100.5)])
        line = describe_analysis(r)
        self.assertIn("BULLISH/BOS_UP", line)
        self.assertIn("BUY_FROM_BASE", line)
        self.assertIn("+5.0 → STRONG_BUY", line)
        self.assertIn("chưa có dữ liệu", describe_analysis(None))


class TestCandleBuilder(unittest.TestCase):
    def test_gom_tick_thanh_nen(self) -> None:
        b = CandleBuilder(3, max_candles=10)
        for t, p in enumerate([1, 3, 2, 4, 6, 5, 7], start=1):
            b.add(t, float(p))
        self.assertEqual(b.klines(), [[1, 1, 3, 1, 2, 0], [4, 4, 6, 4, 5, 0], [7, 7, 7, 7, 7, 0]])

    def test_gioi_han_so_nen_va_trang_thai(self) -> None:
        b = CandleBuilder(1, max_candles=3)
        for t in range(10):
            b.add(t, 100.0 + t)
        self.assertEqual([c[4] for c in b.klines()], [107, 108, 109])
        b.add(10, 50.0)
        clone = CandleBuilder(1, max_candles=3)
        clone.load_state(b.to_state())
        self.assertEqual(clone.klines(), b.klines())
        with self.assertRaises(ValueError):
            CandleBuilder(0)


class TestConfluenceFeed(unittest.TestCase):
    def test_dung_nen_gom_tu_tick(self) -> None:
        feed = ConfluenceFeed(htf_ticks=2, ltf_ticks=1)
        for t, p in enumerate(UP_PATH, start=1):
            r = feed.update(t, p)
        self.assertEqual(r["source"], "nến gom từ tick")
        self.assertEqual(r["tick"], len(UP_PATH))
        self.assertEqual(len(feed.htf.klines()), 11)  # 21 tick, 2 tick/nến (nến cuối đang hình thành)
        self.assertEqual((r["htf_trend"], r["market_structure_event"], r["confluence_score"]),
                         (BULLISH, "BOS_UP", 2.0))
        self.assertIs(feed.last, r)

    def test_nen_binance_va_lam_moi_theo_chu_ky(self) -> None:
        htf = from_path(UP_PATH)
        ltf = FILLER + BASE + [k(100.02, 100.6, 100.0, 100.5)]
        fetch = mock.Mock(side_effect=lambda interval, limit: htf if interval == "4h" else ltf)
        feed = ConfluenceFeed(fetch=fetch, htf_interval="4h", ltf_interval="5m", refresh_htf=10, refresh_ltf=2)
        r = feed.update(1, 100.0)
        self.assertEqual(r["source"], "Binance 4h/5m")
        self.assertEqual((r["confluence_score"], r["recommended_action"]), (5.0, "STRONG_BUY"))
        self.assertEqual([c.args for c in fetch.call_args_list], [("4h", 60), ("5m", 30)])
        feed.update(2, 100.0)
        self.assertEqual(fetch.call_count, 2)  # cả hai còn trong chu kỳ làm mới
        feed.update(3, 100.0)
        self.assertEqual(fetch.call_args.args, ("5m", 30))

    def test_loi_mang_thi_dung_nen_tu_tick_va_khoi_phuc_trang_thai(self) -> None:
        feed = ConfluenceFeed(fetch=lambda interval, limit: None, htf_ticks=2, ltf_ticks=1)
        for t, p in enumerate(UP_PATH, start=1):
            r = feed.update(t, p)
        self.assertEqual(r["source"], "nến gom từ tick")
        clone = ConfluenceFeed(htf_ticks=2, ltf_ticks=1)
        clone.load_state(feed.to_state())
        self.assertEqual(clone.htf.klines(), feed.htf.klines())
        self.assertEqual(clone.last, feed.last)


if __name__ == "__main__":
    unittest.main()
