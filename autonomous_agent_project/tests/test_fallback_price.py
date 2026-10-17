"""Giá khởi điểm giả lập "sát thực tế" khi --live-price lỗi, và tính độc lập với thang giá (không cần mạng)."""

from __future__ import annotations

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main
from core import dashboard as dash
from core.price_feed import (
    DEFAULT_FALLBACK_PRICE,
    REALISTIC_FALLBACK_PRICES,
    format_price,
    get_smart_fallback_price,
)
from core.strategy import MarketSimulator


class TestBangGia(unittest.TestCase):
    def test_tra_cuu(self) -> None:
        self.assertEqual(get_smart_fallback_price("BTCUSDT"), 82000.0)
        self.assertEqual(get_smart_fallback_price("ethusdt"), 2400.0)
        self.assertEqual(get_smart_fallback_price("XRPUSDT"), 0.6)
        self.assertEqual(get_smart_fallback_price("KHONGCO"), DEFAULT_FALLBACK_PRICE)
        self.assertEqual(set(REALISTIC_FALLBACK_PRICES), {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"})

    def test_dinh_dang_gia(self) -> None:
        self.assertEqual(format_price(82000.0), "82,000.00")
        self.assertEqual(format_price(2400.5), "2,400.50")
        self.assertEqual(format_price(1.0), "1.00")
        self.assertEqual(format_price(0.6), "0.6000")
        self.assertEqual(format_price(0.61237), "0.6124")
        self.assertEqual(format_price(0.00001234), "0.00001234")
        self.assertEqual(format_price(0.0), "0.00")


class TestKhoiDongDuPhong(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def eco(self, *argv: str, memory: str = "m.json") -> main.Ecosystem:
        args = main.parse_args(["--tick", "0", "--no-fsync", "-q", "--memory", str(self.dir / memory), *argv])
        return main.Ecosystem(args)

    def offline_eco(self, symbol: str, *argv: str, memory: str = "m.json") -> tuple[main.Ecosystem, str]:
        with mock.patch("core.price_feed.fetch_price", return_value=(None, None)), \
                self.assertLogs("he_sinh_thai", level="WARNING") as logs:
            eco = self.eco("--live-price", "--symbol", symbol, *argv, memory=memory)
        return eco, "\n".join(logs.output)

    def test_btc_khoi_dong_quanh_82000(self) -> None:
        eco, text = self.offline_eco("btcusdt", "--seed", "1")
        self.assertIsInstance(eco.market, MarketSimulator)
        self.assertIn("[Cảnh báo] Không thể kết nối Binance API. Kích hoạt chế độ giả lập với giá khởi điểm "
                      "thực tế: 82,000.00 cho BTCUSDT", text)
        self.assertTrue(60000 < eco.market.snapshot().price < 110000)
        self.assertEqual(eco.symbol, "BTCUSDT")

    def test_xrp_gia_nho_khong_mat_chinh_xac(self) -> None:
        eco, text = self.offline_eco("XRPUSDT", "--seed", "1")
        self.assertIn("0.6000 cho XRPUSDT", text)
        self.assertTrue(0.4 < eco.market.snapshot().price < 0.8)

    def test_cap_la_dung_mac_dinh_va_ghi_chu(self) -> None:
        eco, text = self.offline_eco("DOGEUSDT", "--seed", "1")
        self.assertIn("100.00 cho DOGEUSDT (chưa có mốc giá cho cặp này, dùng mặc định)", text)

    def test_thanh_cong_bao_gia_that(self) -> None:
        with mock.patch("core.price_feed.fetch_price", return_value=(2412.5, "https://api.binance.com")), \
                self.assertLogs("he_sinh_thai", level="INFO") as logs:
            eco = self.eco("--live-price", "--symbol", "ETHUSDT")
        self.assertIn("[Thành công] Đã lấy giá thật từ Binance: 2,412.50", "\n".join(logs.output))
        self.assertEqual(eco.price_source, "Binance ETHUSDT (thật)")

    def test_che_do_mac_dinh_van_bat_dau_tu_100(self) -> None:
        eco = self.eco("--seed", "1")
        reference = MarketSimulator(random.Random(random.Random(1).random()))
        self.assertEqual(eco.market.prices, reference.prices)
        self.assertTrue(80 < eco.market.snapshot().price < 125)
        self.assertIsNone(eco.symbol)

    def final_state(self, eco: main.Ecosystem) -> list[tuple]:
        with contextlib.redirect_stdout(io.StringIO()):
            eco.run()
        return [
            (a.agent_id, a.state.value, a.generation, a.parent_id, a.stats.trades, a.stats.wins, round(a.capital, 6))
            for a in sorted(eco.agents + eco.graveyard, key=lambda a: a.agent_id)
        ]

    def test_ket_qua_khong_phu_thuoc_thang_gia(self) -> None:
        with mock.patch("core.price_feed.fetch_price", return_value=(None, None)), \
                contextlib.redirect_stderr(io.StringIO()):
            btc = self.final_state(self.eco("--live-price", "--symbol", "BTCUSDT", "--seed", "9", "--steps", "300",
                                            memory="btc.json"))
            xrp = self.final_state(self.eco("--live-price", "--symbol", "XRPUSDT", "--seed", "9", "--steps", "300",
                                            memory="xrp.json"))
        plain = self.final_state(self.eco("--seed", "9", "--steps", "300", memory="plain.json"))
        self.assertGreater(sum(row[4] for row in btc), 100)
        self.assertEqual(btc, xrp)
        self.assertEqual(btc, plain)


@unittest.skipUnless(dash.RICH_AVAILABLE, "chưa cài rich")
class TestDashboardGiaNho(unittest.TestCase):
    def test_hien_thi_gia_xrp(self) -> None:
        view = dash.generate_dashboard_view(1, 0, "XRPUSDT", 0.61237, -0.4, 5, None)
        text = dash.render_to_text(view, width=140, height=40)
        self.assertIn("0.6124 USDT", text)
        self.assertIn("XRPUSDT", text)


if __name__ == "__main__":
    unittest.main()
