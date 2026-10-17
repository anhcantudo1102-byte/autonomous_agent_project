"""Vẽ đường vốn hệ sinh thái so với Buy & Hold từ các tệp ``--equity-csv`` của ``backtest_compare.py``.

Cần matplotlib (tuỳ chọn, không nằm trong requirements.txt): ``pip install matplotlib``.

    python3 scripts/plot_backtest.py --results results/backtest-no-cost --symbol BTCUSDT --seed 1 \\
        --output backtest-equity.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PERIOD_TITLES = {"tang": "Tăng", "giam": "Giảm", "ngang": "Đi ngang / điều chỉnh"}
CONFIG_LABELS = {"mac_dinh": "Mặc định", "adv_agent": "--advanced-analysis, tinh hoa agent",
                 "adv_production": "--advanced-analysis (production)"}


def read_curve(path: Path) -> tuple[list[int], list[float], list[float]]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return ([int(r["tick"]) for r in rows], [float(r["von_he_sinh_thai"]) for r in rows],
            [float(r["von_buy_hold"]) for r in rows])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Biểu đồ đường vốn backtest.")
    p.add_argument("--results", type=Path, required=True, help="Thư mục --out của backtest_compare.py")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--periods", nargs="+", default=list(PERIOD_TITLES))
    p.add_argument("--configs", nargs="+", default=["mac_dinh", "adv_production"])
    p.add_argument("--title", default="")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib — chạy `pip install matplotlib` để vẽ biểu đồ.", file=sys.stderr)
        return 1
    fig, axes = plt.subplots(1, len(args.periods), figsize=(5.2 * len(args.periods), 4.2), squeeze=False)
    for ax, period in zip(axes[0], args.periods):
        hold = None
        for config in args.configs:
            path = args.results / "equity" / f"{args.symbol}_{period}_{config}_s{args.seed}.csv"
            ticks, equity, hold_curve = read_curve(path)
            ax.plot(ticks, equity, linewidth=1.3, label=CONFIG_LABELS.get(config, config))
            hold = (ticks, hold_curve)
        ax.plot(*hold, color="black", linewidth=1.1, linestyle="--", label="Buy & Hold")
        ax.axhline(100, color="grey", linewidth=0.6)
        ax.set_title(f"{args.symbol} — {PERIOD_TITLES.get(period, period)}")
        ax.set_xlabel("Tick (1 nến)")
        ax.set_ylabel("Tổng vốn ($)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper left")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)
    print(f"Đã lưu biểu đồ: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
