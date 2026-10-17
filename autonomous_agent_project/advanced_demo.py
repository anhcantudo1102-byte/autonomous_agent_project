"""Demo chiến lược đa chiều (SMC + Elliott heuristic + Hidden Base) trên nến HTF/LTF.

Ví dụ:
    python advanced_demo.py                         # nến giả lập xu hướng tăng
    python advanced_demo.py --scenario giam --seed 3
    python advanced_demo.py --live --symbol ETHUSDT --htf 4h --ltf 5m
    python advanced_demo.py --dummy-goc             # tái hiện dữ liệu dummy ban đầu (luôn WAIT)
"""

from __future__ import annotations

import argparse
import random
import sys

from core.advanced_strategy import MultiDimensionalAnalyzer, decide, generate_synthetic_klines
from core.price_feed import fetch_klines

SCENARIOS = {"tang": 0.004, "giam": -0.004, "di_ngang": 0.0}
DECISION_TEXT = {
    "LONG": "MỞ VỊ THẾ MUA (LONG) theo dòng tiền thông minh.",
    "SHORT": "MỞ VỊ THẾ BÁN (SHORT) theo xu hướng.",
    "WAIT": "KIÊN NHẪN CHỜ ĐỢI (WAIT FOR SETUP).",
}


def synthetic_data(scenario: str, rng: random.Random) -> tuple[list, list]:
    drift = SCENARIOS[scenario]
    htf = generate_synthetic_klines(60, drift=drift, volatility=0.006, rng=rng)
    last = htf[-1][4]
    ltf = generate_synthetic_klines(30, start_price=last, drift=drift / 8, volatility=0.0015, rng=rng,
                                    interval_ms=5 * 60 * 1000)
    return htf, ltf


def load_data(args: argparse.Namespace) -> tuple[list, list, str]:
    if args.dummy_goc:
        return [[0, 80000, 82000, 79000, 81500]] * 15, [[0, 81000, 81600, 80900, 81400]] * 10, "dummy gốc (nến giống hệt nhau)"
    rng = random.Random(args.seed)
    if args.live:
        htf = fetch_klines(args.symbol, args.htf, args.htf_limit)
        ltf = fetch_klines(args.symbol, args.ltf, args.ltf_limit) if htf else None
        if htf and ltf:
            return htf, ltf, f"Binance {args.symbol.upper()} thật ({args.htf}/{args.ltf})"
        print("⚠️  Không lấy được nến thật từ Binance — dùng nến GIẢ LẬP để tiếp tục.")
    htf, ltf = synthetic_data(args.scenario, rng)
    return htf, ltf, f"giả lập kịch bản '{args.scenario}' (seed={args.seed})"


def run_advanced_agent_simulation(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Demo phân tích đa khung SMC + Elliott + Hidden Base")
    p.add_argument("--live", action="store_true", help="Lấy nến thật từ Binance (lỗi thì dùng giả lập)")
    p.add_argument("--symbol", default="BTCUSDT", help="Cặp giao dịch")
    p.add_argument("--htf", default="4h", help="Khung lớn (xu hướng)")
    p.add_argument("--ltf", default="5m", help="Khung nhỏ (điểm vào lệnh)")
    p.add_argument("--htf-limit", type=int, default=100, help="Số nến HTF")
    p.add_argument("--ltf-limit", type=int, default=50, help="Số nến LTF")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="tang", help="Kịch bản nến giả lập")
    p.add_argument("--seed", type=int, default=7, help="Hạt giống cho nến giả lập")
    p.add_argument("--dummy-goc", action="store_true", help="Dùng dữ liệu dummy ban đầu để so sánh")
    args = p.parse_args(argv)

    analyzer = MultiDimensionalAnalyzer(symbol=args.symbol)
    print("=== KHỞI CHẠY MÔ PHỎNG AGENT VỚI CHIẾN LƯỢC ĐA CHIỀU (SMC + Elliott + Hidden Base) ===")
    htf, ltf, source = load_data(args)
    print(f"📥 Nguồn nến: {source} — HTF {len(htf)} nến, LTF {len(ltf)} nến, close cuối {float(ltf[-1][4]):,.2f}")

    result = analyzer.analyze_market_structure(htf, ltf)
    print("📊 Kết quả phân tích đa khung thời gian:")
    print(f"   - Xu hướng HTF (SMC): {result['trend']}")
    print(f"   - Tín hiệu LTF (Price Action/Hidden Base): {result['signal']}")
    print(f"   - Trạng thái sóng Elliott (heuristic động lượng): {result['wave']} ({result['wave_direction']})")
    print(f"   - Điểm hợp lưu (Confluence Score): {result['confluence_score']}")
    print("🔎 Lý do:")
    for reason in result["reasons"]:
        print(f"   • {reason}")
    decision = decide(result["confluence_score"])
    print(f"🤖 Quyết định của Agent: {DECISION_TEXT[decision]}")
    result["decision"] = decision
    return result


if __name__ == "__main__":
    run_advanced_agent_simulation()
    sys.exit(0)
