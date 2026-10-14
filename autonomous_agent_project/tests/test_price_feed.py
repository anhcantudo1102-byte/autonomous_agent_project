"""Kiểm thử lấy giá Binance (giả lập HTTP, không cần mạng)."""

from __future__ import annotations

import unittest
from unittest import mock

from core import price_feed


class FakeResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        return self._payload


@unittest.skipIf(price_feed.requests is None, "chưa cài requests")
class TestPriceFeed(unittest.TestCase):
    def test_lay_gia_thanh_cong(self) -> None:
        with mock.patch.object(price_feed.requests, "get", return_value=FakeResponse(200, {"price": "82024.01"})) as get:
            self.assertEqual(price_feed.get_latest_price("btcusdt"), 82024.01)
        self.assertEqual(get.call_args.kwargs["params"], {"symbol": "BTCUSDT"})

    def test_bi_chan_vung_thi_chuyen_endpoint_du_phong(self) -> None:
        blocked = FakeResponse(451, {"code": 0, "msg": "Service unavailable from a restricted location"})
        ok = FakeResponse(200, {"symbol": "BTCUSDT", "price": "100.5"})
        with mock.patch.object(price_feed.requests, "get", side_effect=[blocked, ok]) as get, \
                mock.patch("builtins.print"):
            self.assertEqual(price_feed.get_latest_price(), 100.5)
        self.assertIn("data-api.binance.vision", get.call_args_list[1].args[0])

    def test_moi_endpoint_loi_thi_tra_ve_none(self) -> None:
        err = price_feed.requests.ConnectionError("mất mạng")
        with mock.patch.object(price_feed.requests, "get", side_effect=err), mock.patch("builtins.print") as out:
            self.assertIsNone(price_feed.get_latest_price())
        self.assertEqual(out.call_count, len(price_feed.BINANCE_ENDPOINTS))


if __name__ == "__main__":
    unittest.main()
