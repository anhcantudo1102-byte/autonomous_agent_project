"""Kiểm thử chế độ backtest: tải nến (giả lập mạng), cache CSV, thị trường lịch sử, không nhìn trước tương lai.

Không test nào cần mạng: HTTP được thay bằng hàm giả, dữ liệu mẫu là 200 nến BTCUSDT 1h thật
trong ``tests/data/BTCUSDT-1h-mau.csv``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import tempfile
import unittest
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import main
from core import checkpoint as ckpt
from core import history as hist
from core.advanced_strategy import CandleBuilder, ConfluenceFeed
from core.memory import MemoryBank

SAMPLE = Path(__file__).resolve().parent / "data" / "BTCUSDT-1h-mau.csv"
HOUR = 3_600_000
T0 = hist.date_to_ms(date(2024, 1, 1))


def make_candles(n: int, start_ms: int = T0, step: int = HOUR, price: float = 100.0) -> list[list]:
    rows = []
    for i in range(n):
        o = price
        c = price * (1 + (0.01 if i % 7 < 4 else -0.012))
        rows.append([start_ms + i * step, o, max(o, c) * 1.002, min(o, c) * 0.998, c, 10.0 + i])
        price = c
    return rows


class FakeResponse:
    def __init__(self, status: int, payload=None, content: bytes = b"") -> None:
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeBinance:
    """Máy chủ klines giả: trả nến theo startTime/endTime/limit như Binance; endpoint 'chan' trả HTTP 451."""

    def __init__(self, candles: list[list], blocked=("https://chan",)) -> None:
        self.candles = [[r[0], *(str(x) for x in r[1:])] for r in candles]
        self.blocked = set(blocked)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, params=None, timeout=None):
        base = url.split("/api/")[0]
        self.calls.append((base, dict(params or {})))
        if base in self.blocked:
            return FakeResponse(451, {"code": 0, "msg": "Service unavailable from a restricted location"})
        p = params
        rows = [r for r in self.candles if p["startTime"] <= r[0] <= p["endTime"]][: p["limit"]]
        return FakeResponse(200, rows)


def monthly_zip(rows: list[list], microseconds: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        text = io.StringIO()
        writer = csv.writer(text)
        for r in rows:
            t = r[0] * 1000 if microseconds else r[0]
            writer.writerow([t, *r[1:6], t + HOUR - 1, 0, 0, 0, 0, 0])
        zf.writestr("data.csv", text.getvalue())
    return buf.getvalue()


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)


class TestTimeHelpers(unittest.TestCase):
    def test_khoang_ngay_tinh_ca_hai_dau_theo_utc(self) -> None:
        start, end = hist.date_range_ms(date(2024, 1, 1), date(2024, 1, 2))
        self.assertEqual(start, 1704067200000)
        self.assertEqual(end, 1704067200000 + 2 * 86_400_000 - 1)
        with self.assertRaises(ValueError):
            hist.date_range_ms(date(2024, 1, 2), date(2024, 1, 1))

    def test_ngay_va_khung_sai(self) -> None:
        self.assertEqual(hist.parse_date("2024-02-29"), date(2024, 2, 29))
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            hist.parse_date("01/02/2024")
        self.assertEqual(hist.interval_ms("4h"), 4 * HOUR)
        with self.assertRaisesRegex(ValueError, "không hỗ trợ"):
            hist.interval_ms("7h")

    def test_danh_sach_thang_qua_nam_moi(self) -> None:
        a = hist.date_to_ms(date(2023, 11, 15))
        b = hist.date_to_ms(date(2024, 2, 1))
        self.assertEqual(hist._months(a, b), [(2023, 11), (2023, 12), (2024, 1), (2024, 2)])
        self.assertEqual(hist.ms_to_text(T0), "2024-01-01 00:00")


class TestNormalize(unittest.TestCase):
    def test_micro_giay_doi_ve_mili_giay(self) -> None:
        row = hist.normalize_row([T0 * 1000, "1", "2", "0.5", "1.5", "7"])
        self.assertEqual(row, [T0, 1.0, 2.0, 0.5, 1.5, 7.0])

    def test_thieu_volume_thi_bang_0_va_loi_khi_thieu_cot(self) -> None:
        self.assertEqual(hist.normalize_row([T0, 1, 1, 1, 1])[5], 0.0)
        self.assertEqual(hist.normalize_row([T0, 1, 1, 1, 1, ""])[5], 0.0)
        with self.assertRaisesRegex(ValueError, "thiếu cột"):
            hist.normalize_row([T0, 1, 1, 1])

    def test_nen_vo_ly_bi_tu_choi(self) -> None:
        for bad in ([T0, 1, 0.9, 0.8, 1], [T0, 1, 2, 1.1, 1.5], [T0, 0, 1, 0, 1]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "không hợp lệ"):
                hist.normalize_row(bad)

    def test_loc_khoang_sap_xep_bo_trung(self) -> None:
        rows = make_candles(5)
        mixed = [rows[3], rows[0], rows[3], rows[4], rows[1], rows[2]]
        out = hist.clean_candles(mixed, rows[1][0], rows[3][0])
        self.assertEqual([r[0] for r in out], [rows[1][0], rows[2][0], rows[3][0]])


class TestCsv(TempDirCase):
    def test_doc_mau_that(self) -> None:
        candles = hist.read_history_csv(SAMPLE)
        self.assertEqual(len(candles), 200)
        self.assertEqual(candles[0][0], 1711929600000)
        self.assertTrue(all(b[0] - a[0] == HOUR for a, b in zip(candles, candles[1:])))

    def test_ghi_roi_doc_lai_y_nguyen(self) -> None:
        rows = make_candles(30)
        path = hist.write_history_csv(self.dir / "sub" / "x.csv", rows)
        self.assertEqual(hist.read_history_csv(path), hist.clean_candles(rows))
        self.assertFalse(path.with_suffix(".csv.tmp").exists())

    def test_khong_tieu_de_va_dinh_dang_goc_binance(self) -> None:
        rows = make_candles(4)
        plain = self.dir / "plain.csv"
        plain.write_text("\n".join(",".join(str(x) for x in r) for r in rows) + "\n\n", encoding="utf-8")
        self.assertEqual(len(hist.read_history_csv(plain)), 4)
        raw = self.dir / "raw.csv"
        with zipfile.ZipFile(io.BytesIO(monthly_zip(rows, microseconds=True))) as zf:
            raw.write_bytes(zf.read("data.csv"))
        self.assertEqual([r[0] for r in hist.read_history_csv(raw)], [r[0] for r in rows])

    def test_tep_loi(self) -> None:
        with self.assertRaisesRegex(hist.HistoryUnavailable, "Không đọc được"):
            hist.read_history_csv(self.dir / "khong_co.csv")
        empty = self.dir / "empty.csv"
        empty.write_text("open_time,open,high,low,close\n", encoding="utf-8")
        with self.assertRaisesRegex(hist.HistoryUnavailable, "không có nến"):
            hist.read_history_csv(empty)
        bad = self.dir / "bad.csv"
        bad.write_text("1,2,3\n", encoding="utf-8")
        with self.assertRaisesRegex(hist.HistoryUnavailable, "sai định dạng"):
            hist.read_history_csv(bad)

    def test_ten_tep_cache(self) -> None:
        path = hist.cache_path("btcusdt", "1h", date(2024, 1, 1), date(2024, 6, 30), self.dir)
        self.assertEqual(path, self.dir / "BTCUSDT-1h-20240101-20240630.csv")


class TestDownload(TempDirCase):
    def test_phan_trang_va_chuyen_endpoint_khi_bi_chan(self) -> None:
        server = FakeBinance(make_candles(2500))
        start, end = T0, T0 + 2499 * HOUR
        got = hist.fetch_klines_range("btcusdt", "1h", start, end, endpoints=("https://chan", "https://ok"),
                                      get=server, report=lambda _m: None)
        self.assertEqual(len(got), 2500)
        self.assertEqual([r[0] for r in got], [T0 + i * HOUR for i in range(2500)])
        pages = [p for base, p in server.calls if base == "https://ok"]
        self.assertEqual([p["startTime"] for p in pages], [T0, T0 + 1000 * HOUR, T0 + 2000 * HOUR])
        self.assertEqual(pages[0]["symbol"], "BTCUSDT")
        # Endpoint bị chặn chỉ bị thử ở trang đầu; các trang sau đi thẳng tới endpoint đã thành công.
        self.assertEqual([base for base, _ in server.calls].count("https://chan"), 1)

    def test_trang_day_dung_limit_roi_het_du_lieu(self) -> None:
        server = FakeBinance(make_candles(10), blocked=())
        got = hist.fetch_klines_range("BTCUSDT", "1h", T0, T0 + 50 * HOUR, endpoints=("https://ok",),
                                      get=server, limit=5)
        self.assertEqual(len(got), 10)
        self.assertEqual(len(server.calls), 3)

    def test_moi_endpoint_loi_tra_ve_none(self) -> None:
        msgs: list[str] = []

        def broken(url, params=None, timeout=None):
            if "a.example" in url:
                raise OSError("mất mạng")
            return FakeResponse(500, ValueError("không phải JSON")) if "b.example" in url else FakeResponse(503, "x")

        got = hist.fetch_klines_range("BTCUSDT", "1h", T0, T0 + HOUR, get=broken, report=msgs.append,
                                      endpoints=("https://a.example", "https://b.example", "https://c.example"))
        self.assertIsNone(got)
        self.assertEqual(len(msgs), 3)
        self.assertIn("mất mạng", msgs[0])

    def test_zip_hang_thang_micro_giay_va_thang_thieu(self) -> None:
        jan = make_candles(24 * 31, start_ms=hist.date_to_ms(date(2025, 1, 1)))
        urls: list[str] = []

        def vision(url, params=None, timeout=None):
            urls.append(url)
            if url.endswith("2025-01.zip"):
                return FakeResponse(200, content=monthly_zip(jan, microseconds=True))
            if url.endswith("2025-02.zip"):
                return FakeResponse(200, content=b"khong phai zip")
            return FakeResponse(404)

        msgs: list[str] = []
        start, end = hist.date_range_ms(date(2025, 1, 10), date(2025, 3, 5))
        got = hist.fetch_monthly_zips("ethusdt", "1h", start, end, base_url="https://z", get=vision,
                                      report=msgs.append)
        self.assertEqual(urls[0], "https://z/ETHUSDT/1h/ETHUSDT-1h-2025-01.zip")
        self.assertEqual(len(urls), 3)
        self.assertEqual(got[0][0], hist.date_to_ms(date(2025, 1, 10)))
        self.assertEqual(len(got), 22 * 24)
        self.assertEqual(len(msgs), 2)
        self.assertIsNone(hist.fetch_monthly_zips("X", "1h", start, end, get=lambda *a, **k: FakeResponse(404),
                                                  report=lambda _m: None))

    def test_tai_ve_luu_cache_lan_sau_chay_offline(self) -> None:
        server = FakeBinance(make_candles(48, start_ms=hist.date_to_ms(date(2024, 3, 1))))
        candles, source = hist.load_history("BTCUSDT", "1h", date(2024, 3, 1), date(2024, 3, 2), cache_dir=self.dir,
                                            get=server, endpoints=("https://chan", "https://ok"),
                                            report=lambda _m: None)
        self.assertEqual(len(candles), 48)
        self.assertIn("/api/v3/klines", source)
        cache = self.dir / "BTCUSDT-1h-20240301-20240302.csv"
        self.assertTrue(cache.exists())

        def offline(*_a, **_k):
            raise AssertionError("không được gọi mạng khi đã có cache")

        again, source2 = hist.load_history("BTCUSDT", "1h", date(2024, 3, 1), date(2024, 3, 2), cache_dir=self.dir,
                                           get=offline)
        self.assertEqual(again, candles)
        self.assertTrue(source2.startswith("cache"))

    def test_rest_bi_chan_thi_dung_zip(self) -> None:
        rows = make_candles(24 * 29, start_ms=hist.date_to_ms(date(2024, 2, 1)))

        def blocked_then_zip(url, params=None, timeout=None):
            if "/api/v3/klines" in url:
                return FakeResponse(451, {"msg": "restricted"})
            return FakeResponse(200, content=monthly_zip(rows))

        candles, source = hist.load_history("SOLUSDT", "1h", date(2024, 2, 1), date(2024, 2, 3), cache_dir=self.dir,
                                            get=blocked_then_zip, report=lambda _m: None)
        self.assertEqual(len(candles), 72)
        self.assertIn("zip", source)

    def test_khong_co_du_lieu_nao(self) -> None:
        with self.assertRaisesRegex(hist.HistoryUnavailable, "Không tải được"):
            hist.load_history("BTCUSDT", "1h", date(2024, 3, 1), date(2024, 3, 1), cache_dir=self.dir,
                              get=lambda *a, **k: FakeResponse(404, []), report=lambda _m: None)

    def test_thieu_requests_va_khoang_rong(self) -> None:
        with mock.patch.object(hist.price_feed, "requests", None):
            with self.assertRaisesRegex(hist.HistoryUnavailable, "requests"):
                hist.fetch_klines_range("BTCUSDT", "1h", T0, T0 + HOUR)
        self.assertEqual(hist.fetch_klines_range("BTCUSDT", "1h", T0 + HOUR, T0, get=FakeBinance([])), [])
        empty = self.dir / "trong.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(hist.HistoryUnavailable, "không có nến"):
            hist.read_history_csv(empty)

    def test_rest_tra_rong_thi_thu_zip(self) -> None:
        server = FakeBinance([], blocked=())
        self.assertEqual(hist.fetch_klines_range("BTCUSDT", "1h", T0, T0 + HOUR, get=server,
                                                 endpoints=("https://ok",)), [])


class TestHistoricalMarket(unittest.TestCase):
    def setUp(self) -> None:
        self.candles = make_candles(40)
        self.m = hist.HistoricalMarket(self.candles, symbol="btcusdt", interval="1h")

    def test_khoi_dong_va_anh_xa_tick_nen(self) -> None:
        m = self.m
        self.assertEqual(m.warmup, 13)
        self.assertEqual(m.total_ticks, 27)
        self.assertEqual(m.tick, 0)
        self.assertEqual(m.start_price, self.candles[12][4])
        self.assertEqual((len(m.prices), len(m.returns)), (13, 12))
        self.assertEqual(m.regime, "lich_su")
        self.assertEqual(m.snapshot().regime, "lich_su")
        snap = m.step()
        self.assertEqual(snap.tick, 1)
        self.assertEqual(snap.price, self.candles[13][4])
        self.assertEqual(m.current_candle, self.candles[13])
        self.assertAlmostEqual(snap.ret_1, self.candles[13][4] / self.candles[12][4] - 1)
        self.assertEqual(m.remaining(), 26)
        self.assertEqual(m.time_range(), (hist.ms_to_text(self.candles[13][0]), hist.ms_to_text(self.candles[-1][0])))

    def test_seek_trung_khop_voi_chay_tuan_tu(self) -> None:
        other = hist.HistoricalMarket(self.candles)
        for t in range(1, 28):
            snap = self.m.step()
            self.assertEqual(other.seek(t), snap)
            self.assertEqual((other.prices, other.returns), (self.m.prices, self.m.returns))

    def test_het_nen_va_thieu_nen(self) -> None:
        for _ in range(27):
            self.m.step()
        with self.assertRaises(hist.HistoryExhausted):
            self.m.step()
        with self.assertRaises(hist.HistoryExhausted):
            self.m.seek(28)
        with self.assertRaisesRegex(hist.HistoryUnavailable, "Cần hơn 13 nến"):
            hist.HistoricalMarket(make_candles(13))

    def test_buy_hold_va_max_drawdown(self) -> None:
        r = hist.buy_and_hold([100, 120, 90, 150], 50)
        self.assertEqual(r["von_cuoi"], 75)
        self.assertAlmostEqual(r["loi_nhuan"], 0.5)
        self.assertAlmostEqual(r["max_dd"], 0.25)
        self.assertEqual(hist.max_drawdown([1, 2, 3]), 0.0)
        with self.assertRaises(ValueError):
            hist.buy_and_hold([], 1)


class TestNoLookAhead(unittest.TestCase):
    def test_candle_builder_dung_bien_do_nen_that(self) -> None:
        b = CandleBuilder(2)
        b.add(1, 10.5, (10, 11, 9.5, 10.5))
        b.add(2, 10.2, (10.5, 12, 10.1, 10.2))
        self.assertEqual(b.klines(), [[1.0, 10, 12, 9.5, 10.2, 0.0]])
        b.add(3, 10.0)
        self.assertEqual(b.klines()[-1], [3.0, 10.0, 10.0, 10.0, 10.0, 0.0])

    def test_phan_tich_tai_tick_t_khong_doi_khi_cat_bo_tuong_lai(self) -> None:
        candles = hist.read_history_csv(SAMPLE)
        full, cut = ConfluenceFeed(htf_ticks=8, ltf_ticks=2), ConfluenceFeed(htf_ticks=8, ltf_ticks=2)
        short = candles[:120]
        for t, c in enumerate(candles, 1):
            a = full.update(t, c[4], c[1:5])
            if t <= len(short):
                self.assertEqual(cut.update(t, short[t - 1][4], short[t - 1][1:5]), a)
        self.assertEqual(full.source, "nến lịch sử gộp")


class TestBacktestCli(TempDirCase):
    def args(self, *argv: str, memory: str = "m.json") -> list[str]:
        return ["--backtest", "--history-file", str(SAMPLE), "-q", "--memory", str(self.dir / memory),
                "--checkpoint-file", str(self.dir / "c.json"), *argv]

    def eco(self, *argv: str, memory: str = "m.json") -> main.Ecosystem:
        return main.Ecosystem(main.parse_args(self.args(*argv, memory=memory)))

    def final(self, eco: main.Ecosystem) -> tuple[list[dict], str]:
        _, out = quiet(eco.run)
        return [ag.to_state() for ag in sorted(eco.agents + eco.graveyard, key=lambda a: a.agent_id)], out

    def test_mac_dinh_cu_khong_doi(self) -> None:
        args = main.parse_args([])
        self.assertFalse(args.backtest)
        self.assertEqual((args.steps, args.tick), (500, main.DEFAULT_TICK_SIMULATED))
        self.assertIsInstance(main.Ecosystem(main.parse_args(["--memory", str(self.dir / "x.json")])).market,
                              main.MarketSimulator)

    def test_tham_so_backtest(self) -> None:
        a = main.parse_args(["--backtest", "--start", "2024-01-01", "--end", "2024-01-31", "--interval", "4h"])
        self.assertEqual((a.start, a.end, a.tick, a.steps), (date(2024, 1, 1), date(2024, 1, 31), 0.0, None))
        d = main.parse_args(["--backtest", "--end", "2024-01-31", "--days", "10"])
        self.assertEqual(d.start, date(2024, 1, 22))
        self.assertEqual(main.parse_args(["--backtest", "--end", "2024-12-31"]).start,
                         date(2024, 12, 31) - timedelta(days=main.DEFAULT_BACKTEST_DAYS - 1))
        bad = (["--backtest", "--live-price"], ["--backtest", "--start", "2024-01-01", "--days", "3"],
               ["--backtest", "--days", "0"], ["--backtest", "--start", "2024-02-01", "--end", "2024-01-01"],
               ["--backtest", "--start", "2024/01/01"], ["--backtest", "--interval", "7h"],
               ["--start", "2024-01-01"], ["--history-file", "x.csv"], ["--backtest", "--steps", "0"])
        for argv in bad:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main.parse_args(argv)

    def test_chay_het_nen_bao_cao_buy_hold_va_ghi_tep(self) -> None:
        eco = self.eco("--seed", "3", "--equity-csv", str(self.dir / "e.csv"),
                       "--summary-json", str(self.dir / "s.json"))
        self.assertIsInstance(eco.market, hist.HistoricalMarket)
        self.assertTrue(eco.memory.compact)
        self.assertFalse(eco.memory.durable)
        self.assertIn("lịch sử BTCUSDT 1h", eco.price_source)
        _, out = self.final(eco)
        self.assertEqual(eco.current_tick, 187)
        self.assertEqual(eco.args.steps, 187)
        self.assertIn("BACKTEST BTCUSDT 1h", out)
        self.assertIn("Buy & Hold", out)
        summary = json.loads((self.dir / "s.json").read_text(encoding="utf-8"))
        candles = hist.read_history_csv(SAMPLE)
        self.assertEqual(summary["so_tick"], 187)
        self.assertEqual(summary["von_dau"], 100.0)
        self.assertAlmostEqual(summary["buy_hold_von_cuoi"], 100 * candles[-1][4] / candles[12][4], places=4)
        self.assertEqual(summary["song"] + summary["chet"], len(eco.agents) + len(eco.graveyard))
        with (self.dir / "e.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 188)
        self.assertEqual(rows[0]["thoi_gian_utc"], "khởi động")
        self.assertAlmostEqual(float(rows[-1]["von_he_sinh_thai"]), summary["von_cuoi"], places=4)
        self.assertAlmostEqual(hist.max_drawdown([float(r["von_he_sinh_thai"]) for r in rows]), summary["max_dd"],
                               places=5)
        # Bộ nhớ ghi dạng JSON gọn (không thụt lề) nhưng vẫn đọc được.
        text = (self.dir / "m.json").read_text(encoding="utf-8")
        self.assertNotIn("\n  ", text)
        self.assertIsInstance(json.loads(text), dict)

    def test_steps_bi_gioi_han_boi_so_nen(self) -> None:
        eco = self.eco("--seed", "3", "--steps", "40")
        self.final(eco)
        self.assertEqual(eco.current_tick, 40)
        self.assertEqual(eco.backtest_summary()["so_tick"], 40)
        many = self.eco("--seed", "3", "--steps", "9999", memory="b.json")
        self.final(many)
        self.assertEqual(many.current_tick, 187)

    def test_diet_vong_thi_von_dung_yen_toi_cuoi_giai_doan(self) -> None:
        eco = self.eco("--seed", "3", "--operating-cost", "3")
        _, out = self.final(eco)
        s = eco.backtest_summary()
        self.assertIsNotNone(s["diet_vong_tai_tick"])
        self.assertEqual(len(eco.equity_curve), 188)
        self.assertEqual(s["so_tick"], 187)
        self.assertEqual(eco.equity_curve[-1][2], eco.equity_curve[s["diet_vong_tai_tick"]][2])
        self.assertIn("diệt vong ở tick", out)

    def test_phan_tich_da_khung_dung_nen_lich_su_khong_nhin_truoc(self) -> None:
        cut = self.dir / "cut.csv"
        hist.write_history_csv(cut, hist.read_history_csv(SAMPLE)[:100])
        full = self.eco("--seed", "4", "--advanced-analysis", "--steps", "87")
        short = main.Ecosystem(main.parse_args([
            "--backtest", "--history-file", str(cut), "-q", "--memory", str(self.dir / "b.json"),
            "--seed", "4", "--advanced-analysis"]))
        a, _ = self.final(full)
        b, out = self.final(short)
        self.assertEqual(short.current_tick, 87)
        self.assertEqual(a, b)
        self.assertEqual(full.analysis, short.analysis)
        self.assertIn("nến lịch sử gộp", out)

    def test_resume_tat_dinh(self) -> None:
        flags = ("--seed", "5", "--advanced-analysis")
        straight, _ = self.final(self.eco(*flags, memory="a.json"))
        self.final(self.eco(*flags, "--steps", "90", "--checkpoint-every", "90", memory="b.json"))
        data = ckpt.read_checkpoint(self.dir / "c.json", ckpt.KIND_ECOSYSTEM)
        self.assertEqual(data["market"]["type"], "history")
        self.assertEqual(data["market"]["interval"], "1h")
        resumed_eco = self.eco(*flags, "--resume", memory="b.json")
        self.assertEqual((resumed_eco.start_tick, resumed_eco.market.tick), (90, 90))
        resumed, _ = self.final(resumed_eco)
        self.assertEqual(resumed, straight)
        self.assertEqual(resumed_eco.current_tick, 187)

    def test_resume_khac_chuoi_nen_thi_khoi_tao_moi(self) -> None:
        self.final(self.eco("--seed", "5", "--steps", "30", "--checkpoint-every", "30"))
        other = self.dir / "other.csv"
        hist.write_history_csv(other, make_candles(60))
        eco = main.Ecosystem(main.parse_args([
            "--backtest", "--history-file", str(other), "-q", "--memory", str(self.dir / "m.json"),
            "--checkpoint-file", str(self.dir / "c.json"), "--seed", "5", "--resume"]))
        self.assertEqual((eco.start_tick, eco.market.tick), (0, 0))
        sim = self.eco("--seed", "5", "--resume")  # cùng chuỗi → khôi phục
        self.assertEqual(sim.start_tick, 30)
        plain = main.Ecosystem(main.parse_args(["-q", "--memory", str(self.dir / "m.json"), "--checkpoint-file",
                                                str(self.dir / "c.json"), "--resume", "--steps", "5"]))
        self.assertEqual(plain.start_tick, 30)  # giả lập vẫn đọc được checkpoint backtest (giữ chuỗi giá hiện tại)

    def test_resume_o_nen_cuoi_khong_con_tick(self) -> None:
        self.final(self.eco("--seed", "5", "--checkpoint-every", "187"))
        eco = self.eco("--seed", "5", "--resume")
        _, out = self.final(eco)
        self.assertEqual(eco.current_tick, 187)
        self.assertIn("BACKTEST", out)

    def test_loi_tai_du_lieu_tra_ma_2(self) -> None:
        code = main.main(["--backtest", "--history-file", str(self.dir / "khong_co.csv"), "-q",
                          "--memory", str(self.dir / "m.json")])
        self.assertEqual(code, 2)

    def test_tai_qua_cache_khi_khong_co_history_file(self) -> None:
        hist.write_history_csv(hist.cache_path("BTCUSDT", "1h", date(2024, 4, 1), date(2024, 4, 9), self.dir),
                               hist.read_history_csv(SAMPLE))
        eco = main.Ecosystem(main.parse_args([
            "--backtest", "--start", "2024-04-01", "--end", "2024-04-09", "--history-dir", str(self.dir),
            "-q", "--memory", str(self.dir / "m.json"), "--seed", "1"]))
        self.assertIn("cache", eco.price_source)
        self.assertEqual(eco.market.total_ticks, 187)


class TestCompactMemory(TempDirCase):
    def test_compact_va_mac_dinh(self) -> None:
        pretty = MemoryBank(self.dir / "a.json")
        compact = MemoryBank(self.dir / "b.json", compact=True)
        self.assertIn("\n  ", (self.dir / "a.json").read_text(encoding="utf-8"))
        self.assertNotIn("\n", (self.dir / "b.json").read_text(encoding="utf-8"))
        self.assertEqual(pretty.summary(), compact.summary())


if __name__ == "__main__":
    unittest.main()
