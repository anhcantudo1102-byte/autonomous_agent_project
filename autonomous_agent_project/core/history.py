"""Dữ liệu nến lịch sử thật cho chế độ backtest (``main.py --backtest``).

Nguồn dữ liệu, thử lần lượt:

1. ``/api/v3/klines`` của ``api.binance.com`` rồi ``data-api.binance.vision`` (phân trang theo
   ``startTime``, tối đa 1000 nến mỗi lần gọi).
2. Nếu mọi endpoint REST bị chặn: tệp zip hàng tháng của ``data.binance.vision``
   (``data/spot/monthly/klines/...``). Từ năm 2025 các tệp này ghi thời gian theo micro giây,
   được chuẩn hoá về mili giây.

Kết quả được lưu cache CSV trong ``data/history/`` nên lần chạy sau không cần mạng.
``HistoricalMarket`` có cùng giao diện với ``MarketSimulator``: một nến = một tick, giá của tick
là giá đóng cửa của nến; chỉ báo của tick t chỉ dùng các nến đã đóng tới t.
"""

from __future__ import annotations

import csv
import io
import logging
import random
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import price_feed
from .strategy import MarketSnapshot, build_snapshot

log = logging.getLogger("lich_su")

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"
VISION_ZIP_BASE = "https://data.binance.vision/data/spot/monthly/klines"
PAGE_LIMIT = 1000
CSV_HEADER = ("open_time", "open", "high", "low", "close", "volume")
MICROSECOND_THRESHOLD = 10**14  # open_time lớn hơn mức này là micro giây (mili giây hiện tại ~1.7e12)

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}

Candle = list  # [open_time_ms (int), open, high, low, close, volume]
HttpGet = Callable[..., Any]  # cùng chữ ký requests.get(url, params=..., timeout=...)


class HistoryUnavailable(RuntimeError):
    """Không tải được (hoặc không đọc được) dữ liệu nến lịch sử."""


class HistoryExhausted(RuntimeError):
    """Đã chạy hết chuỗi nến lịch sử."""


# ---------------------------------------------------------------------- Tiện ích thời gian
def parse_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Ngày '{text}' không đúng định dạng YYYY-MM-DD") from e


def date_to_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def ms_to_text(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def date_range_ms(start: date, end: date) -> tuple[int, int]:
    """Khoảng [đầu ngày ``start``, cuối ngày ``end``] theo UTC, đơn vị mili giây (cả hai đầu tính)."""
    if end < start:
        raise ValueError(f"Ngày kết thúc {end} trước ngày bắt đầu {start}")
    return date_to_ms(start), date_to_ms(end + timedelta(days=1)) - 1


def interval_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(f"Khung nến '{interval}' không hỗ trợ (chọn: {', '.join(INTERVAL_MS)})") from None


# ---------------------------------------------------------------------- Chuẩn hoá nến
def normalize_row(row: Sequence[Any]) -> Candle:
    """Một dòng kline (REST, zip hoặc CSV) → ``[open_time_ms, open, high, low, close, volume]``."""
    if len(row) < 5:
        raise ValueError(f"Dòng nến thiếu cột (cần ≥ 5): {list(row)!r}")
    open_time = int(float(row[0]))
    if open_time > MICROSECOND_THRESHOLD:
        open_time //= 1000
    o, h, l, c = (float(x) for x in row[1:5])
    volume = float(row[5]) if len(row) > 5 and str(row[5]).strip() != "" else 0.0
    if min(o, h, l, c) <= 0 or h < max(o, c, l) or l > min(o, c):
        raise ValueError(f"Nến không hợp lệ tại {open_time}: open={o}, high={h}, low={l}, close={c}")
    return [open_time, o, h, l, c, volume]


def clean_candles(rows: Iterable[Sequence[Any]], start_ms: int | None = None, end_ms: int | None = None) -> list[Candle]:
    """Chuẩn hoá, lọc theo khoảng thời gian, sắp xếp và bỏ nến trùng ``open_time``."""
    unique: dict[int, Candle] = {}
    for row in rows:
        candle = normalize_row(row)
        if start_ms is not None and candle[0] < start_ms:
            continue
        if end_ms is not None and candle[0] > end_ms:
            continue
        unique[candle[0]] = candle
    return [unique[t] for t in sorted(unique)]


# ---------------------------------------------------------------------- CSV cache
def read_history_csv(path: str | Path) -> list[Candle]:
    """Đọc CSV nến: có hoặc không có dòng tiêu đề; nhận cả định dạng 12 cột gốc của Binance."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.reader(fh) if r and any(cell.strip() for cell in r)]
    except OSError as e:
        raise HistoryUnavailable(f"Không đọc được tệp lịch sử {path}: {e}") from e
    if rows:
        try:
            float(rows[0][0])
        except ValueError:
            rows = rows[1:]
    try:
        candles = clean_candles(rows)
    except ValueError as e:
        raise HistoryUnavailable(f"Tệp lịch sử {path} sai định dạng: {e}") from e
    if not candles:
        raise HistoryUnavailable(f"Tệp lịch sử {path} không có nến nào")
    return candles


def write_history_csv(path: str | Path, candles: Sequence[Candle]) -> Path:
    """Ghi atomic (tệp tạm rồi đổi tên) để cache không bao giờ bị ghi dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for c in candles:
            writer.writerow([int(c[0]), *(repr(float(x)) for x in c[1:6])])
    tmp.replace(path)
    return path


def cache_path(symbol: str, interval: str, start: date, end: date, cache_dir: str | Path = HISTORY_DIR) -> Path:
    return Path(cache_dir) / f"{symbol.upper()}-{interval}-{start:%Y%m%d}-{end:%Y%m%d}.csv"


# ---------------------------------------------------------------------- Tải dữ liệu
def _default_get() -> HttpGet:
    if price_feed.requests is None:
        raise HistoryUnavailable("Chưa cài thư viện requests — chạy `pip install requests` để tải dữ liệu lịch sử")
    return price_feed.requests.get


def _request_errors() -> tuple[type[BaseException], ...]:
    req = price_feed.requests
    return (OSError, ValueError) + ((req.RequestException,) if req is not None else ())


def fetch_klines_range(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    endpoints: Sequence[str] = price_feed.BINANCE_ENDPOINTS,
    get: HttpGet | None = None,
    timeout: float = 10.0,
    limit: int = PAGE_LIMIT,
    report: Callable[[str], object] = log.warning,
) -> list[Candle] | None:
    """Tải mọi nến trong [start_ms, end_ms] qua ``/api/v3/klines``, phân trang theo ``startTime``.

    Endpoint vừa thành công được ưu tiên cho trang sau. Một trang mà mọi endpoint đều lỗi → ``None``
    (người gọi chuyển sang tệp zip). Không có nến nào trong khoảng → danh sách rỗng.
    """
    get = get or _default_get()
    errors = _request_errors()
    step = interval_ms(interval)
    order = list(endpoints)
    rows: list[Sequence[Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {"symbol": symbol.upper(), "interval": interval, "startTime": cursor, "endTime": end_ms,
                  "limit": limit}
        page = None
        for base in list(order):
            try:
                response = get(f"{base}/api/v3/klines", params=params, timeout=timeout)
                data = response.json()
            except errors as e:
                report(f"Lỗi khi tải nến lịch sử từ {base}: {e}")
                continue
            if response.status_code != 200 or not isinstance(data, list):
                msg = data.get("msg", data) if isinstance(data, dict) else data
                report(f"Lỗi từ {base} (HTTP {response.status_code}) khi tải nến lịch sử: {msg}")
                continue
            if order[0] != base:
                order.remove(base)
                order.insert(0, base)
                log.info("🌐 Dùng %s cho các trang nến tiếp theo", base)
            page = data
            break
        if page is None:
            return None
        if not page:
            break
        rows.extend(page)
        last_open = int(float(page[-1][0]))
        if len(page) < limit:
            break
        cursor = last_open + step
    return clean_candles(rows, start_ms, end_ms)


def _months(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    first = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    last = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    year, month, months = first.year, first.month, []
    while (year, month) <= (last.year, last.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def fetch_monthly_zips(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = VISION_ZIP_BASE,
    get: HttpGet | None = None,
    timeout: float = 30.0,
    report: Callable[[str], object] = log.warning,
) -> list[Candle] | None:
    """Tải tệp zip hàng tháng từ data.binance.vision; tháng thiếu (ví dụ tháng hiện tại) chỉ cảnh báo."""
    get = get or _default_get()
    errors = _request_errors() + (zipfile.BadZipFile,)
    symbol = symbol.upper()
    rows: list[list[str]] = []
    for year, month in _months(start_ms, end_ms):
        name = f"{symbol}-{interval}-{year:04d}-{month:02d}"
        url = f"{base_url}/{symbol}/{interval}/{name}.zip"
        try:
            response = get(url, timeout=timeout)
            if response.status_code != 200:
                report(f"Không có tệp {name}.zip trên data.binance.vision (HTTP {response.status_code})")
                continue
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for member in zf.namelist():
                    text = zf.read(member).decode("utf-8")
                    rows.extend(r for r in csv.reader(io.StringIO(text)) if r and r[0][:1].isdigit())
        except errors as e:
            report(f"Lỗi khi tải {name}.zip: {e}")
    if not rows:
        return None
    return clean_candles(rows, start_ms, end_ms)


def load_history(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    *,
    cache_dir: str | Path = HISTORY_DIR,
    get: HttpGet | None = None,
    endpoints: Sequence[str] = price_feed.BINANCE_ENDPOINTS,
    zip_base_url: str = VISION_ZIP_BASE,
    report: Callable[[str], object] = log.warning,
) -> tuple[list[Candle], str]:
    """Trả về (nến, mô tả nguồn). Có cache thì đọc cache (không cần mạng); không thì tải rồi ghi cache."""
    interval_ms(interval)
    path = cache_path(symbol, interval, start, end, cache_dir)
    if path.exists():
        return read_history_csv(path), f"cache {path}"
    start_ms, end_ms = date_range_ms(start, end)
    candles = fetch_klines_range(symbol, interval, start_ms, end_ms, endpoints=endpoints, get=get, report=report)
    source = "Binance /api/v3/klines"
    if not candles:
        report("REST klines không dùng được — thử tệp zip hàng tháng của data.binance.vision")
        candles = fetch_monthly_zips(symbol, interval, start_ms, end_ms, base_url=zip_base_url, get=get,
                                     report=report)
        source = "data.binance.vision (zip hàng tháng)"
    if not candles:
        raise HistoryUnavailable(f"Không tải được nến {symbol.upper()} {interval} từ {start} đến {end}")
    write_history_csv(path, candles)
    return candles, f"{source}, đã lưu cache {path}"


# ---------------------------------------------------------------------- Thị trường lịch sử
class HistoricalMarket:
    """Phát lại chuỗi nến lịch sử với giao diện của ``MarketSimulator``.

    ``warmup`` nến đầu (mặc định ``max(lookback, vol_window) + 1``, bằng số giá khởi động của
    ``MarketSimulator``) chỉ dùng để tính chỉ báo cho tick 0; tick t ứng với nến ``warmup + t - 1``.
    """

    regime = "lich_su"

    def __init__(
        self,
        candles: Sequence[Sequence[Any]],
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        lookback: int = 12,
        vol_window: int = 10,
        warmup: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.candles = [normalize_row(c) for c in candles]
        self.symbol = symbol.upper()
        self.interval = interval
        self.lookback = lookback
        self.vol_window = vol_window
        self.warmup = max(lookback, vol_window) + 1 if warmup is None else warmup
        if self.warmup < 1 or len(self.candles) <= self.warmup:
            raise HistoryUnavailable(
                f"Cần hơn {self.warmup} nến để backtest (nến khởi động chỉ báo), hiện có {len(self.candles)}")
        self.rng = rng or random.Random()
        self.seek(0)
        self.start_price = self.prices[-1]

    @property
    def total_ticks(self) -> int:
        return len(self.candles) - self.warmup

    def remaining(self) -> int:
        return self.total_ticks - self.tick

    def candle_at(self, tick: int) -> Candle:
        return self.candles[self.warmup + tick - 1]

    @property
    def current_candle(self) -> Candle:
        return self.candle_at(self.tick)

    def seek(self, tick: int) -> MarketSnapshot:
        """Đặt thị trường về đúng trạng thái sau ``tick`` (dùng khi khởi tạo và khôi phục checkpoint)."""
        if not 0 <= tick <= self.total_ticks:
            raise HistoryExhausted(f"Tick {tick} nằm ngoài chuỗi lịch sử (0..{self.total_ticks})")
        self.tick = tick
        end = self.warmup + tick
        keep = max(self.lookback, self.vol_window) + 2
        closes = [c[4] for c in self.candles[max(0, end - keep - 1):end]]
        self.prices = closes[-keep:]
        self.returns = [b / a - 1 for a, b in zip(closes, closes[1:])][-keep:]
        self._snapshot = self._build_snapshot()
        return self._snapshot

    def _build_snapshot(self) -> MarketSnapshot:
        return build_snapshot(self.tick, self.prices, self.returns, lookback=self.lookback,
                              vol_window=self.vol_window, regime=self.regime)

    def snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def step(self) -> MarketSnapshot:
        if self.remaining() <= 0:
            raise HistoryExhausted(f"Đã hết {self.total_ticks} nến lịch sử của {self.symbol}")
        self.tick += 1
        price = self.current_candle[4]
        self.returns.append(price / self.prices[-1] - 1)
        self.prices.append(price)
        keep = max(self.lookback, self.vol_window) + 2
        del self.prices[:-keep]
        del self.returns[:-keep]
        self._snapshot = self._build_snapshot()
        return self._snapshot

    def time_range(self) -> tuple[str, str]:
        """Thời điểm mở nến của tick 1 và của nến cuối cùng (UTC)."""
        return ms_to_text(self.candle_at(1)[0]), ms_to_text(self.candles[-1][0])


def buy_and_hold(prices: Sequence[float], capital: float) -> dict[str, float]:
    """Mốc so sánh: mua ``capital`` ở giá đầu, giữ tới giá cuối. Trả về vốn cuối, lợi nhuận % và Max DD."""
    if not prices or prices[0] <= 0:
        raise ValueError("Cần chuỗi giá dương để tính Buy & Hold")
    equity = [capital * p / prices[0] for p in prices]
    return {"von_cuoi": equity[-1], "loi_nhuan": equity[-1] / capital - 1, "max_dd": max_drawdown(equity)}


def max_drawdown(equity: Sequence[float]) -> float:
    """Sụt giảm lớn nhất từ đỉnh (0..1) của một đường vốn."""
    peak, worst = float("-inf"), 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst
