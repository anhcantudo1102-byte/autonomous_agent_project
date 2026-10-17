"""Chạy ma trận backtest (cặp × giai đoạn × cấu hình × seed) song song rồi tổng hợp kết quả.

Ví dụ (đứng ở thư mục ``autonomous_agent_project/``)::

    python3 scripts/backtest_compare.py --out results/backtest --seeds 1 2 3 4 5
    python3 scripts/backtest_compare.py --out results/backtest-no-cost --operating-cost 0

Mỗi lần chạy là một tiến trình ``main.py --backtest`` riêng với bộ nhớ chung riêng (``--reset-memory``),
nên kết quả không phụ thuộc thứ tự chạy. Lần đầu cần mạng để tải nến (cache ở ``data/history/``).
Kết quả: ``<out>/runs/*.json`` (một tệp mỗi lần chạy), ``<out>/tong_hop.csv`` và bảng in ra màn hình.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

PERIODS = {
    "tang": ("2023-10-01", "2024-03-31"),
    "giam": ("2022-04-01", "2022-09-30"),
    "ngang": ("2024-04-01", "2024-09-30"),
}
CONFIGS = {
    "mac_dinh": [],
    "adv_agent": ["--advanced-analysis", "--elite-fitness", "agent"],
    "adv_production": ["--advanced-analysis"],
}


def run_one(job: dict, out: Path, operating_cost: float, interval: str, equity_seeds: set[int]) -> dict:
    name = f"{job['symbol']}_{job['period']}_{job['config']}_s{job['seed']}"
    summary = out / "runs" / f"{name}.json"
    if summary.exists():
        return json.loads(summary.read_text(encoding="utf-8")) | job
    start, end = PERIODS[job["period"]]
    cmd = [sys.executable, str(PROJECT_DIR / "main.py"), "--backtest", "--symbol", job["symbol"],
           "--interval", interval, "--start", start, "--end", end, "--seed", str(job["seed"]), "-q",
           "--reset-memory", "--memory", str(out / "memory" / f"{name}.json"), "--summary-json", str(summary),
           "--operating-cost", str(operating_cost), *CONFIGS[job["config"]]]
    if job["seed"] in equity_seeds:
        cmd += ["--equity-csv", str(out / "equity" / f"{name}.csv")]
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    if result.returncode != 0 or not summary.exists():
        raise RuntimeError(f"{name} lỗi (mã {result.returncode}): {result.stderr[-2000:]}")
    return json.loads(summary.read_text(encoding="utf-8")) | job


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["symbol"], r["period"], r["config"]), []).append(r)
    table = []
    for (symbol, period, config), items in sorted(groups.items()):
        table.append({
            "cap": symbol, "giai_doan": period, "cau_hinh": config, "so_seed": len(items),
            "von_cuoi_tb": statistics.fmean(r["von_cuoi"] for r in items),
            "von_cuoi_min": min(r["von_cuoi"] for r in items),
            "von_cuoi_max": max(r["von_cuoi"] for r in items),
            "max_dd_tb": statistics.fmean(r["max_dd"] for r in items),
            "song_tb": statistics.fmean(r["song"] for r in items),
            "chet_tb": statistics.fmean(r["chet"] for r in items),
            "diet_vong": sum(r["diet_vong_tai_tick"] is not None for r in items),
            "thang_buy_hold": sum(r["von_cuoi"] > r["buy_hold_von_cuoi"] for r in items),
            "buy_hold_von_cuoi": items[0]["buy_hold_von_cuoi"],
            "buy_hold_max_dd": items[0]["buy_hold_max_dd"],
        })
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ma trận backtest so sánh cấu hình hệ sinh thái với Buy & Hold.")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "results" / "backtest")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--periods", nargs="+", default=list(PERIODS), choices=list(PERIODS))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--interval", default="1h")
    p.add_argument("--operating-cost", type=float, default=0.02)
    p.add_argument("--equity-seeds", nargs="*", type=int, default=[1], help="Seed được ghi đường vốn CSV")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    for sub in ("runs", "memory", "equity"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    jobs = [{"symbol": s, "period": per, "config": c, "seed": seed}
            for s in args.symbols for per in args.periods for c in args.configs for seed in args.seeds]
    print(f"Chạy {len(jobs)} backtest với {args.workers} tiến trình song song → {args.out}", flush=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, j, args.out, args.operating_cost, args.interval, set(args.equity_seeds))
                   for j in jobs]
        for i, f in enumerate(futures, 1):
            rows.append(f.result())
            if i % 10 == 0 or i == len(jobs):
                print(f"  xong {i}/{len(jobs)}", flush=True)
    table = aggregate(rows)
    with (args.out / "tong_hop.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"{'Cặp':<8} {'Giai đoạn':<6} {'Cấu hình':<15} {'Vốn cuối TB':>11} {'min–max':>17} {'MaxDD TB':>8} "
          f"{'Sống':>5} {'Chết':>5} {'Diệt vong':>9} {'>B&H':>5} {'B&H':>8} {'B&H DD':>7}")
    for t in table:
        print(f"{t['cap']:<8} {t['giai_doan']:<6} {t['cau_hinh']:<15} ${t['von_cuoi_tb']:>10.2f} "
              f"{t['von_cuoi_min']:>7.2f}–{t['von_cuoi_max']:<9.2f} {t['max_dd_tb']:>8.1%} {t['song_tb']:>5.1f} "
              f"{t['chet_tb']:>5.1f} {t['diet_vong']:>4}/{t['so_seed']:<4} {t['thang_buy_hold']:>2}/{t['so_seed']:<2} "
              f"${t['buy_hold_von_cuoi']:>7.2f} {t['buy_hold_max_dd']:>7.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
