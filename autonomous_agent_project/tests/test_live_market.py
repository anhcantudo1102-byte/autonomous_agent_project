"""Kiểm thử chế độ giá thật (--live-price) và cơ chế dự phòng về giá giả lập (không cần mạng)."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from core import price_feed
from core.price_feed import LiveMarketFeed, LivePriceUnavailable
from core.strategy import MarketSimulator


def scripted(prices: list[float | None]):
    it = iter(prices)
    return lambda: next(it, None)


class TestLiveMarketFeed(unittest.TestCase):
    def test_tinh_chi_bao_tu_gia_that(self) -> None:
        feed = LiveMarketFeed("btcusdt", fetch=scripted([100.0, 101.0, 102.0]))
        self.assertEqual(feed.symbol, "BTCUSDT")
        self.assertEqual(feed.snapshot().tick, 0)
        snap = feed.step()
        self.assertEqual((snap.tick, snap.price, snap.regime), (1, 101.0, "thuc_te"))
        self.assertAlmostEqual(snap.ret_1, 0.01)
        snap = feed.step()
        self.assertAlmostEqual(snap.momentum, 102.0 / ((100 + 101 + 102) / 3) - 1)
        self.assertGreater(snap.volatility, 0.0)

    def test_api_loi_giua_chung_dung_gia_gia_lap_roi_quay_lai_gia_that(self) -> None:
        feed = LiveMarketFeed("BTCUSDT", fetch=scripted([100.0, 101.0, None, None, 105.0]))
        feed.step()
        fallback = feed.step()
        self.assertEqual(fallback.regime, "gia_lap_du_phong")
        self.assertGreater(fallback.price, 0)
        self.assertNotEqual(fallback.price, 101.0)
        feed.step()
        self.assertEqual((feed.failed_fetches, feed.stale_streak), (2, 2))
        real = feed.step()
        self.assertEqual((real.price, real.regime, feed.stale_streak), (105.0, "thuc_te", 0))

    def test_khong_lay_duoc_gia_khoi_diem(self) -> None:
        with self.assertRaises(LivePriceUnavailable):
            LiveMarketFeed("BTCUSDT", fetch=lambda: None)

    @unittest.skipIf(price_feed.requests is None, "chưa cài requests")
    def test_uu_tien_endpoint_vua_thanh_cong(self) -> None:
        class Resp:
            def __init__(self, status: int, payload: dict) -> None:
                self.status_code, self._payload = status, payload

            def json(self) -> dict:
                return self._payload

        def fake_get(url: str, **_: object) -> Resp:
            if "api.binance.com" in url:
                return Resp(451, {"msg": "restricted location"})
            return Resp(200, {"price": "100.0"})

        with mock.patch.object(price_feed.requests, "get", side_effect=fake_get) as get, \
                self.assertLogs("gia_that", level="INFO"):
            feed = LiveMarketFeed("BTCUSDT")
            feed.step()
        urls = [c.args[0] for c in get.call_args_list]
        self.assertEqual(len(urls), 3)  # khởi động: bị chặn + dự phòng; tick 1: gọi thẳng endpoint dự phòng
        self.assertIn("data-api.binance.vision", urls[2])


class TestMainLivePrice(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.mem = str(Path(self._tmp.name) / "memory_bank.json")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_tham_so_cli(self) -> None:
        a = main.parse_args(["--memory", self.mem])
        self.assertEqual((a.live_price, a.symbol, a.tick), (False, "BTCUSDT", main.DEFAULT_TICK_SIMULATED))
        a = main.parse_args(["--live-price", "--symbol", "ETHUSDT", "--population-cap", "50"])
        self.assertEqual((a.live_price, a.symbol, a.tick, a.max_agents), (True, "ETHUSDT", main.DEFAULT_TICK_LIVE, 50))
        self.assertEqual(main.parse_args(["--live-price", "--tick", "0"]).tick, 0.0)

    def test_khoi_dong_that_bai_thi_chuyen_sang_gia_lap(self) -> None:
        args = main.parse_args(["--live-price", "--memory", self.mem, "--seed", "1", "--tick", "0"])
        boom = mock.Mock(side_effect=LivePriceUnavailable("không lấy được giá"))
        with mock.patch.object(main, "LiveMarketFeed", boom), self.assertLogs("he_sinh_thai", level="WARNING"):
            eco = main.Ecosystem(args)
        self.assertIsInstance(eco.market, MarketSimulator)
        self.assertIn("dự phòng", eco.price_source)

    def test_he_sinh_thai_chay_tren_gia_that(self) -> None:
        prices = [80000.0 * (1 + 0.004 * i) for i in range(40)]
        feed = lambda symbol, rng: LiveMarketFeed(symbol, fetch=scripted(prices), rng=rng)  # noqa: E731
        args = main.parse_args(
            ["--live-price", "--memory", self.mem, "--seed", "2", "--tick", "0", "--steps", "30", "-q"]
        )
        with mock.patch.object(main, "LiveMarketFeed", side_effect=feed):
            eco = main.Ecosystem(args)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(eco.run(), 0)
        self.assertIsInstance(eco.market, LiveMarketFeed)
        self.assertEqual(eco.market.tick, 30)
        self.assertEqual(eco.market.failed_fetches, 0)
        self.assertGreater(sum(a.stats.trades for a in eco.agents + eco.graveyard), 0)
        self.assertIn("BÁO CÁO CUỐI CÙNG", out.getvalue())


if __name__ == "__main__":
    unittest.main()
