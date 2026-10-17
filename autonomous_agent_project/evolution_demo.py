"""Demo tiến hoá có checkpoint: chạy lại lệnh là tiếp tục từ thế hệ đã lưu.

    python evolution_demo.py            # lần 1: thế hệ 1–5, lưu checkpoints/latest_checkpoint.json
    python evolution_demo.py            # lần 2: khôi phục, chạy tiếp thế hệ 6–10
    python evolution_demo.py --reset    # xoá checkpoint, bắt đầu lại từ thế hệ 1

Đây là bản đã sửa của ``run_evolution_with_checkpoint`` gốc:

* ``best_agent`` là **bản sao** (``copy.deepcopy``). Bản gốc gán tham chiếu tới phần tử của quần thể
  nên fitness của "agent tốt nhất" cứ tăng theo quần thể mỗi thế hệ, không còn là kỷ lục đã ghi nhận;
* mức tăng fitness mỗi thế hệ phụ thuộc vào gen (có nhiễu ngẫu nhiên tái lập được), gen đột biến nhẹ
  sau mỗi thế hệ, nên agent tốt nhất thực sự thay đổi thay vì luôn là ID 0;
* trạng thái bộ sinh số ngẫu nhiên được lưu trong checkpoint: chạy 5 + 5 thế hệ cho kết quả y hệt
  chạy liền 10 thế hệ;
* checkpoint ghi atomic, có ``schema_version``, đặt trong thư mục dự án (``core/checkpoint.py``).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

from core.checkpoint import (
    CHECKPOINT_DIR,
    DEFAULT_FILENAME,
    load_checkpoint,
    resolve_path,
    restore_rng,
    rng_state,
    save_checkpoint,
)

POPULATION_SIZE = 10
BASE_GAIN = 1.5
GENE_MUTATION = 0.05


def initial_population() -> list[dict[str, Any]]:
    return [{"id": i, "genes": [0.5, 0.8], "fitness": 0.0} for i in range(POPULATION_SIZE)]


def evaluate(agent: dict[str, Any], rng: random.Random) -> float:
    """Điểm sống sót thế hệ này: 1.5 × chất lượng gen × nhiễu thị trường (giả lập)."""
    quality = sum(agent["genes"]) / len(agent["genes"])
    return BASE_GAIN * quality * rng.uniform(0.8, 1.2)


def mutate(agent: dict[str, Any], rng: random.Random) -> None:
    agent["genes"] = [round(min(1.0, max(0.0, g + rng.gauss(0, GENE_MUTATION))), 4) for g in agent["genes"]]


def run_evolution_with_checkpoint(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="Demo tiến hoá quần thể có lưu/khôi phục checkpoint.")
    p.add_argument("--generations", type=int, default=5, help="Số thế hệ chạy trong lần này")
    p.add_argument("--file", default=DEFAULT_FILENAME, help="Tên tệp checkpoint (hoặc đường dẫn tuyệt đối)")
    p.add_argument("--dir", type=Path, default=CHECKPOINT_DIR, help="Thư mục checkpoint")
    p.add_argument("--seed", type=int, default=42, help="Hạt giống khi khởi tạo mới")
    p.add_argument("--reset", action="store_true", help="Xoá checkpoint cũ, bắt đầu lại từ thế hệ 1")
    args = p.parse_args(argv)
    if args.generations <= 0:
        p.error("--generations phải > 0")

    print("=== KHỞI ĐỘNG HỆ THỐNG TIẾN HÓA AGENT ===")
    path = resolve_path(args.file, args.dir)
    if args.reset and path.exists():
        path.unlink()
        print(f"[Checkpoint] Đã xoá checkpoint cũ {path}")

    # 1. Thử khôi phục trạng thái từ lần chạy trước
    checkpoint_data = load_checkpoint(args.file, directory=args.dir)
    rng = None
    if checkpoint_data:
        start_generation = checkpoint_data["generation"] + 1
        population = checkpoint_data["population"]
        best_agent = checkpoint_data["best_agent"]
        if checkpoint_data.get("rng"):
            rng = restore_rng(checkpoint_data["rng"])
        print(f"-> Tiếp tục tiến hóa từ thế hệ {start_generation}")
    else:
        start_generation = 1
        population = initial_population()
        best_agent = None
        print("-> Khởi tạo quần thể gen mới hoàn toàn.")
    rng = rng or random.Random(args.seed)

    # 2. Vòng lặp tiến hóa giả lập qua các thế hệ
    gen = start_generation - 1
    for gen in range(start_generation, start_generation + args.generations):
        print(f"\n--- Đang chạy Thế hệ (Generation) {gen} ---")
        for agent in population:
            agent["fitness"] = round(agent["fitness"] + evaluate(agent, rng), 4)

        current_best = max(population, key=lambda x: x["fitness"])
        if best_agent is None or current_best["fitness"] > best_agent["fitness"]:
            best_agent = copy.deepcopy(current_best) | {"generation": gen}
        print(f"🌟 Gen tốt nhất hiện tại (ID: {best_agent['id']}, ghi nhận ở thế hệ {best_agent['generation']}): "
              f"Fitness = {best_agent['fitness']}, gen = {best_agent['genes']}")

        for agent in population:
            mutate(agent, rng)

        # 3. Lưu checkpoint tự động sau mỗi thế hệ (kèm trạng thái ngẫu nhiên để chạy tiếp y hệt)
        save_checkpoint(gen, population, best_agent, args.file, directory=args.dir, extra={"rng": rng_state(rng)})

    return {"generation": gen, "population": population, "best_agent": best_agent}


if __name__ == "__main__":
    run_evolution_with_checkpoint(sys.argv[1:])
