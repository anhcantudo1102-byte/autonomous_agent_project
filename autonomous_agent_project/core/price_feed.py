"""Lấy giá thị trường thật từ API công khai của Binance (không cần API key).

Chạy thử trong Terminal (đứng ở thư mục ``autonomous_agent_project/``)::

    python -m core.price_feed            # giá BTCUSDT
    python -m core.price_feed ETHUSDT

``api.binance.com`` trả HTTP 451 ở các vùng bị Binance hạn chế; khi đó hàm tự chuyển sang
endpoint dữ liệu thị trường công khai ``data-api.binance.vision`` (cùng định dạng phản hồi).
"""

from __future__ import annotations

import sys
from typing import Sequence

try:
    import requests
except ImportError:  # requests là phụ thuộc tuỳ chọn; phần mô phỏng không cần đến
    requests = None  # type: ignore[assignment]

BINANCE_ENDPOINTS: tuple[str, ...] = (
    "https://api.binance.com",
    "https://data-api.binance.vision",
)


def get_latest_price(
    symbol: str = "BTCUSDT",
    *,
    timeout: float = 5.0,
    endpoints: Sequence[str] = BINANCE_ENDPOINTS,
) -> float | None:
    """Trả về giá khớp gần nhất của ``symbol``, hoặc ``None`` nếu mọi endpoint đều lỗi."""
    if requests is None:
        print("Lỗi: chưa cài thư viện requests — chạy `pip install requests`.")
        return None
    symbol = symbol.upper()
    for base in endpoints:
        url = f"{base}/api/v3/ticker/price"
        try:
            response = requests.get(url, params={"symbol": symbol}, timeout=timeout)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Lỗi khi kết nối API Binance ({base}): {e}")
            continue
        if response.status_code != 200 or "price" not in data:
            msg = data.get("msg", "phản hồi không có trường 'price'") if isinstance(data, dict) else data
            print(f"Lỗi từ API Binance ({base}, HTTP {response.status_code}): {msg}")
            continue
        return float(data["price"])
    return None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    symbol = args[0] if args else "BTCUSDT"
    price = get_latest_price(symbol)
    if price is None:
        print(f"Không lấy được giá {symbol.upper()}.")
        return 1
    print(f"Giá {symbol.upper()} hiện tại: {price:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
