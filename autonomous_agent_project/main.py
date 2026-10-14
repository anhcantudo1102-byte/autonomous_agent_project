"""Điểm khởi chạy: dựng hệ sinh thái agent và điều phối vòng lặp mô phỏng thời gian thực.

Ví dụ:
    python main.py                         # 300 bước, 3 agent, tick 0.05 giây
    python main.py --steps 600 --tick 0 --seed 7 --reset-memory
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import signal
import sys
import time
from pathlib import Path

from core.agent import INITIAL_CAPITAL, Agent
from core.memory import MemoryBank
from core.strategy import TEN_CHE_DO, MarketSimulator, SimulatedExecutionEngine, StrategyParams

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY = PROJECT_DIR / "data" / "memory_bank.json"

log = logging.getLogger("he_sinh_thai")


class Ecosystem:
    """Bộ điều phối: một thị trường chung, nhiều agent, một bộ nhớ chung."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.memory = MemoryBank(args.memory, durable=not args.no_fsync)
        if args.reset_memory:
            self.memory.reset()
        self.market = MarketSimulator(random.Random(self.rng.random()))
        self.engine = SimulatedExecutionEngine(random.Random(self.rng.random()))
        self._ids = itertools.count(1)
        self.agents: list[Agent] = []
        self.graveyard: list[Agent] = []
        self._stop = False
        for _ in range(args.agents):
            self.agents.append(self._spawn_genesis())

    def next_id(self) -> str:
        return f"A-{next(self._ids):03d}"

    def _spawn_genesis(self) -> Agent:
        agent = Agent(
            self.next_id(),
            self.memory,
            StrategyParams.random(self.rng),
            capital=INITIAL_CAPITAL,
            engine=self.engine,
            rng=random.Random(self.rng.random()),
            operating_cost=self.args.operating_cost,
            mutation_rate=self.args.mutation_rate,
        )
        p = agent.params
        log.info(
            "🌱 Khởi sinh %s với $%.2f | ngưỡng động lượng %.2f%%, tỷ trọng %.0f%%, đòn bẩy %.1fx, "
            "chịu biến động %.1f%%%s",
            agent.agent_id,
            agent.capital,
            p.momentum_threshold * 100,
            p.position_fraction * 100,
            p.leverage,
            p.vol_tolerance * 100,
            ", đánh ĐẢO CHIỀU" if p.contrarian else "",
        )
        return agent

    def request_stop(self, *_: object) -> None:
        if not self._stop:
            log.warning("⏹️  Nhận tín hiệu dừng — sẽ kết thúc sau tick hiện tại.")
        self._stop = True

    def run(self) -> int:
        a = self.args
        mem = self.memory.summary()
        log.info(
            "🚀 Bắt đầu mô phỏng: %d bước, tick %.2fs, seed=%s, tối đa %d agent. Bộ nhớ: %s "
            "(%d quy tắc, %d cái chết đã biết)",
            a.steps, a.tick, a.seed, a.max_agents, a.memory, mem["quy_tac"], mem["cai_chet"],
        )
        step = 0
        for step in range(1, a.steps + 1):
            if self._stop:
                break
            snap = self.market.step()
            newborns: list[Agent] = []
            for agent in list(self.agents):
                room = len(self.agents) + len(newborns) < a.max_agents
                newborns += agent.step(snap, id_factory=self.next_id, can_replicate=room)
                if not agent.is_alive:
                    # Agent đã tự hủy: rút khỏi bộ lập lịch, "tiến trình" của nó kết thúc.
                    self.agents.remove(agent)
                    self.graveyard.append(agent)
            self.agents.extend(newborns)

            if step % a.summary_every == 0 or step == a.steps:
                self._print_status(step, snap)
            if not self.agents:
                log.warning("☠️  Toàn bộ hệ sinh thái đã diệt vong tại tick %d.", step)
                break
            if step % 20 == 0:
                self.memory.flush()
            if a.tick > 0:
                time.sleep(a.tick)
        self.memory.flush()
        self._print_report(step)
        return 0

    def _print_status(self, step: int, snap) -> None:
        total = sum(ag.total_assets(snap) for ag in self.agents)
        mem = self.memory.summary()
        gens = max((ag.generation for ag in self.agents), default=0)
        log.info(
            "📊 Tick %4d | giá %8.2f | chế độ %-11s | sống %2d, chết %2d | tổng vốn $%7.2f | "
            "thế hệ cao nhất %d | quy tắc %d (%d hiệu lực)",
            step, snap.price, TEN_CHE_DO[snap.regime], len(self.agents), len(self.graveyard),
            total, gens, mem["quy_tac"], mem["quy_tac_hieu_luc"],
        )

    def _print_report(self, step: int) -> None:
        everyone = sorted(self.agents + self.graveyard, key=lambda ag: ag.agent_id)
        print()
        print("=" * 96)
        print(f" BÁO CÁO CUỐI CÙNG sau {step} tick")
        print("=" * 96)
        print(f" {'ID':<6} {'Thế hệ':>6} {'Cha':<6} {'Trạng thái':<11} {'Vốn':>9} {'Lệnh':>5} "
              f"{'Thắng':>5} {'Thua':>5} {'Bị chặn':>7} {'Điều chỉnh':>10} {'Con':>4}")
        print("-" * 96)
        for ag in everyone:
            s = ag.stats
            print(f" {ag.agent_id:<6} {ag.generation:>6} {ag.parent_id or '-':<6} {ag.state.value:<11} "
                  f"${ag.capital:>8.2f} {s.trades:>5} {s.wins:>5} {s.losses:>5} {s.blocked:>7} "
                  f"{s.adjusted:>10} {s.children:>4}")
        print("-" * 96)
        mem = self.memory.summary()
        print(f" Sống: {len(self.agents)} | Đã chết: {len(self.graveyard)} | "
              f"Tổng vốn còn lại: ${sum(ag.capital for ag in self.agents):.2f}")
        print(f" Bộ nhớ chung ({self.args.memory}): {mem['quy_tac']} quy tắc "
              f"({mem['quy_tac_hieu_luc']} hiệu lực), {mem['that_bai']} hồ sơ thất bại, "
              f"{mem['cai_chet']} hồ sơ cái chết, {mem['pha_he']} sự kiện sinh sản.")
        rules = [r for r in self.memory.rules() if r.get("hieu_luc", True)]
        if rules:
            print(" Một số quy tắc sinh tồn đang hiệu lực:")
            for r in rules[-5:]:
                print(f"   • {r['id']} [{r['hanh_dong']}] {r['mo_ta']}")
        print("=" * 96)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mô phỏng hệ sinh thái Autonomous Adaptive AI Agent (giao dịch giả lập).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=500, help="Số bước (tick) mô phỏng")
    p.add_argument("--tick", type=float, default=0.05, help="Số giây nghỉ giữa hai tick (0 = chạy nhanh nhất)")
    p.add_argument("--seed", type=int, default=None, help="Hạt giống ngẫu nhiên để tái lập kết quả")
    p.add_argument("--agents", type=int, default=5, help="Số agent khởi sinh (thế hệ 0)")
    p.add_argument("--max-agents", type=int, default=30, help="Giới hạn số agent sống cùng lúc")
    p.add_argument("--operating-cost", type=float, default=0.02, help="Chi phí vận hành mỗi tick ($)")
    p.add_argument("--mutation-rate", type=float, default=0.1, help="Độ lớn đột biến gen khi nhân bản")
    p.add_argument("--summary-every", type=int, default=25, help="In tóm tắt sau mỗi N tick")
    p.add_argument("--memory", type=Path, default=DEFAULT_MEMORY, help="Đường dẫn tệp bộ nhớ chung")
    p.add_argument("--reset-memory", action="store_true", help="Xoá sạch bộ nhớ trước khi chạy")
    p.add_argument("--no-fsync", action="store_true", help="Bỏ fsync khi ghi (nhanh hơn, kém bền hơn)")
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="In chi tiết từng lệnh/Post-Mortem")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Chỉ in cảnh báo và báo cáo cuối")
    args = p.parse_args(argv)
    if args.steps <= 0 or args.agents <= 0 or args.max_agents < args.agents or args.summary_every <= 0:
        p.error("Tham số không hợp lệ: cần steps > 0, agents > 0, summary-every > 0 và max-agents ≥ agents")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)-14s %(message)s", datefmt="%H:%M:%S")
    eco = Ecosystem(args)
    signal.signal(signal.SIGINT, eco.request_stop)
    return eco.run()


if __name__ == "__main__":
    sys.exit(main())
