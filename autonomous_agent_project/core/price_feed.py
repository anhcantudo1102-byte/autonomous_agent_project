"""Lấy giá thị trường thật từ API công khai của Binance (không cần API key).

Chạy thử trong Terminal (đứng ở thư mục ``autonomous_agent_project/``)::

    python -m core.price_feed            # giá BTCUSDT
    python -m core.price_feed ETHUSDT

``api.binance.com`` trả HTTP 451 ở các vùng bị Binance hạn chế; khi đó hàm tự chuyển sang
endpoint dữ liệu thị trường công khai ``data-api.binance.vision`` (cùng định dạng phản hồi).

``LiveMarketFeed`` bọc nguồn giá này thành một "thị trường" có cùng giao diện với
``MarketSimulator`` (``snapshot()`` / ``step()``), để ``main.py --live-price`` cho agent giao
dịch (mô phỏng) trên giá thật.
"""

from __future__ import annotations

import logging
import random
import statistics
import sys
from typing import Callable, Sequence

from .strategy import MarketSnapshot, build_snapshot

try:
    import requests
except ImportError:  # requests là phụ thuộc tuỳ chọn; phần mô phỏng không cần đến
    requests = None  # type: ignore[assignment]

BINANCE_ENDPOINTS: tuple[str, ...] = (
    "https://api.binance.com",
    "https://data-api.binance.vision",
)

log = logging.getLogger("gia_that")

# Giá khởi điểm "sát thực tế" cho thị trường giả lập khi không gọi được Binance lúc khởi động.
# Chỉ là mốc tham khảo (không cập nhật tự động); mô phỏng chỉ dùng lợi suất % nên mốc giá
# không ảnh hưởng tới lãi/lỗ, phí hay Fitness.
REALISTIC_FALLBACK_PRICES: dict[str, float] = {
    "BTCUSDT": 82000.0,
    "ETHUSDT": 2400.0,
    "BNBUSDT": 600.0,
    "SOLUSDT": 180.0,
    "XRPUSDT": 0.6,
}
DEFAULT_FALLBACK_PRICE = 100.0


def get_smart_fallback_price(symbol: str) -> float:
    """Giá khởi điểm giả lập theo cặp; cặp chưa có trong bảng → ``DEFAULT_FALLBACK_PRICE``."""
    return REALISTIC_FALLBACK_PRICES.get(symbol.upper(), DEFAULT_FALLBACK_PRICE)


def format_price(price: float) -> str:
    """Định dạng giá có dấu phân cách nghìn; giá nhỏ giữ thêm chữ số (0.6 → ``0.6000``, không thành ``0.60``)."""
    magnitude = abs(price)
    if magnitude >= 1 or magnitude == 0:
        decimals = 2
    elif magnitude >= 0.01:
        decimals = 4
    else:
        decimals = 8
    return f"{price:,.{decimals}f}"


class LivePriceUnavailable(RuntimeError):
    """Không lấy được giá khởi điểm nên không thể bắt đầu chế độ giá thật."""


def fetch_price(
    symbol: str,
    *,
    timeout: float = 5.0,
    endpoints: Sequence[str] = BINANCE_ENDPOINTS,
    report: Callable[[str], object] | None = None,
) -> tuple[float | None, str | None]:
    """Thử lần lượt các endpoint; trả về (giá, endpoint đã thành công) hoặc (None, None).

    ``report`` nhận thông báo lỗi; mặc định là ``print``.
    """
    report = report or print
    if requests is None:
        report("Lỗi: chưa cài thư viện requests — chạy `pip install requests`.")
        return None, None
    symbol = symbol.upper()
    for base in endpoints:
        url = f"{base}/api/v3/ticker/price"
        try:
            response = requests.get(url, params={"symbol": symbol}, timeout=timeout)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            report(f"Lỗi khi kết nối API Binance ({base}): {e}")
            continue
        if response.status_code != 200 or not isinstance(data, dict) or "price" not in data:
            msg = data.get("msg", "phản hồi không có trường 'price'") if isinstance(data, dict) else data
            report(f"Lỗi từ API Binance ({base}, HTTP {response.status_code}): {msg}")
            continue
        return float(data["price"]), base
    return None, None


def get_latest_price(
    symbol: str = "BTCUSDT",
    *,
    timeout: float = 5.0,
    endpoints: Sequence[str] = BINANCE_ENDPOINTS,
) -> float | None:
    """Trả về giá khớp gần nhất của ``symbol``, hoặc ``None`` nếu mọi endpoint đều lỗi."""
    price, _ = fetch_price(symbol, timeout=timeout, endpoints=endpoints)
    return price


def fetch_klines(
    symbol: str,
    interval: str = "4h",
    limit: int = 100,
    *,
    timeout: float = 5.0,
    endpoints: Sequence[str] = BINANCE_ENDPOINTS,
    report: Callable[[str], object] | None = None,
) -> list[list] | None:
    """Lấy nến từ ``/api/v3/klines`` (thử lần lượt các endpoint); ``None`` nếu mọi endpoint lỗi.

    Mỗi nến có dạng ``[open_time, open, high, low, close, volume, ...]`` (giá là chuỗi số).
    """
    report = report or print
    if requests is None:
        report("Lỗi: chưa cài thư viện requests — chạy `pip install requests`.")
        return None
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    for base in endpoints:
        try:
            response = requests.get(f"{base}/api/v3/klines", params=params, timeout=timeout)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            report(f"Lỗi khi kết nối API Binance ({base}): {e}")
            continue
        if response.status_code != 200 or not isinstance(data, list) or not data:
            msg = data.get("msg", data) if isinstance(data, dict) else "không có dữ liệu nến"
            report(f"Lỗi từ API Binance klines ({base}, HTTP {response.status_code}): {msg}")
            continue
        return data
    return None


class LiveMarketFeed:
    """Thị trường dựa trên giá thật: mỗi ``step()`` lấy một giá mới từ Binance.

    Nếu một lần gọi API thất bại, tick đó dùng **giá giả lập**: một bước ngẫu nhiên từ giá cuối
    với độ lệch chuẩn bằng biến động gần đây (chế độ ``gia_lap_du_phong``). Khi API hoạt động
    lại, giá thật được dùng tiếp. Endpoint vừa thành công được ưu tiên cho lần gọi sau để không
    phải thử lại endpoint bị chặn ở mỗi tick.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        *,
        fetch: Callable[[], float | None] | None = None,
        lookback: int = 12,
        vol_window: int = 10,
        timeout: float = 5.0,
        rng: random.Random | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.rng = rng or random.Random()
        self.lookback = lookback
        self.vol_window = vol_window
        self.timeout = timeout
        self.regime = "thuc_te"
        self.tick = 0
        self.failed_fetches = 0
        self.stale_streak = 0
        self._endpoints = list(BINANCE_ENDPOINTS)
        self._fetch = fetch or self._fetch_binance
        first = self._fetch()
        if first is None:
            raise LivePriceUnavailable(f"Không lấy được giá {self.symbol} từ Binance để khởi động chế độ giá thật")
        self.start_price = first
        self.prices: list[float] = [first]
        self.returns: list[float] = []
        self._snapshot = self._build_snapshot()

    def _fetch_binance(self) -> float | None:
        price, used = fetch_price(self.symbol, timeout=self.timeout, endpoints=self._endpoints, report=log.warning)
        if used is not None and self._endpoints[0] != used:
            self._endpoints.remove(used)
            self._endpoints.insert(0, used)
            log.info("🌐 Dùng endpoint %s cho các lần lấy giá tiếp theo", used)
        return price

    def _build_snapshot(self) -> MarketSnapshot:
        return build_snapshot(
            self.tick,
            self.prices,
            self.returns,
            lookback=self.lookback,
            vol_window=self.vol_window,
            regime=self.regime,
        )

    def _simulated_price(self, last: float) -> float:
        recent = self.returns[-self.vol_window :]
        sigma = statistics.pstdev(recent) if len(recent) > 1 else 0.0
        sigma = max(sigma, 1e-4)
        return last * (1 + max(-0.5, self.rng.gauss(0.0, sigma)))

    def snapshot(self) -> MarketSnapshot:
        return self._snapshot

    def step(self) -> MarketSnapshot:
        self.tick += 1
        price = self._fetch()
        last = self.prices[-1]
        if price is None or price <= 0:
            self.failed_fetches += 1
            self.stale_streak += 1
            price = self._simulated_price(last)
            self.regime = "gia_lap_du_phong"
            log.warning(
                "⚠️  Tick %d: không lấy được giá %s, dùng giá giả lập %s (lỗi liên tiếp %d lần)",
                self.tick, self.symbol, format_price(price), self.stale_streak,
            )
        else:
            if self.stale_streak:
                log.info("✅ Tick %d: API Binance hoạt động lại sau %d tick giả lập", self.tick, self.stale_streak)
            self.stale_streak = 0
            self.regime = "thuc_te"
        self.prices.append(price)
        self.returns.append(price / last - 1)
        keep = max(self.lookback, self.vol_window) + 2
        del self.prices[:-keep]
        del self.returns[:-keep]
        self._snapshot = self._build_snapshot()
        log.debug("Tick %d: giá %s = %s", self.tick, self.symbol, format_price(price))
        return self._snapshot


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    symbol = args[0] if args else "BTCUSDT"
    price = get_latest_price(symbol)
    if price is None:
        print(f"Không lấy được giá {symbol.upper()}.")
        return 1
    print(f"Giá {symbol.upper()} hiện tại: {format_price(price)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
