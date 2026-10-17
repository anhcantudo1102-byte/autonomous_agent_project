"""Điểm khởi chạy: dựng hệ sinh thái agent và điều phối vòng lặp mô phỏng thời gian thực.

Ví dụ:
    python main.py                         # 500 bước, 5 agent, tick 0.05 giây (giá giả lập)
    python main.py --steps 600 --tick 0 --seed 7 --reset-memory
    python main.py --live-price --symbol ETHUSDT --steps 300   # giá thật từ Binance, tick 2 giây
    python main.py --steps 200 --checkpoint-every 50           # lưu checkpoint mỗi 50 tick và khi Ctrl+C
    python main.py --steps 200 --resume                        # chạy tiếp 200 tick từ checkpoint
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from core import checkpoint as ckpt
from core.agent import FITNESS_THRESHOLD, INITIAL_CAPITAL, Agent, GeneticConfig, GeneticController
from core.memory import MemoryBank
from core.price_feed import LiveMarketFeed, LivePriceUnavailable
from core.strategy import TEN_CHE_DO, MarketSimulator, SimulatedExecutionEngine, StrategyParams

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY = PROJECT_DIR / "data" / "memory_bank.json"
DEFAULT_CHECKPOINT = ckpt.CHECKPOINT_DIR / "ecosystem_checkpoint.json"
CHECKPOINT_FIELDS = ("tick", "next_id", "rng", "market", "agents", "graveyard")
DEFAULT_TICK_SIMULATED = 0.05
DEFAULT_TICK_LIVE = 2.0

log = logging.getLogger("he_sinh_thai")


class Ecosystem:
    """Bộ điều phối: một thị trường chung, nhiều agent, một bộ nhớ chung."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.memory = MemoryBank(args.memory, durable=not args.no_fsync)
        if args.reset_memory:
            self.memory.reset()
        market_rng = random.Random(self.rng.random())
        self.market: MarketSimulator | LiveMarketFeed
        self.price_source = "giả lập"
        if args.live_price:
            try:
                self.market = LiveMarketFeed(args.symbol, rng=market_rng)
            except LivePriceUnavailable as e:
                log.warning("⚠️  %s — chuyển sang thị trường GIẢ LẬP để tiếp tục chạy.", e)
                self.market = MarketSimulator(market_rng)
                self.price_source = f"giả lập (dự phòng vì không lấy được giá {args.symbol.upper()})"
            else:
                self.price_source = f"Binance {self.market.symbol} (thật)"
                log.info("🌐 Chế độ giá thật: %s từ Binance, giá khởi điểm %.2f",
                         self.market.symbol, self.market.start_price)
        else:
            self.market = MarketSimulator(market_rng)
        self.engine = SimulatedExecutionEngine(random.Random(self.rng.random()))
        self._next_id = 1
        self.agents: list[Agent] = []
        self.graveyard: list[Agent] = []
        self._newborns: list[Agent] = []
        self._stop = False
        self.start_tick = 0
        self.current_tick = 0
        self._saved_tick: int | None = None
        self.best_agent: dict[str, Any] | None = None
        self.checkpointing = args.checkpoint_every > 0 or args.resume
        self.genetics = GeneticController(
            self.memory,
            lambda: [*self.agents, *self._newborns],
            random.Random(self.rng.random()),
            GeneticConfig(
                elite_size=args.elite_size,
                crossover_prob=args.crossover_prob,
                base_mutation_rate=args.mutation_rate,
                min_genome_distance=args.min_diversity,
            ),
        )
        if args.resume and self._resume():
            return
        for _ in range(args.agents):
            self.agents.append(self._spawn_genesis())

    def next_id(self) -> str:
        agent_id = f"A-{self._next_id:03d}"
        self._next_id += 1
        return agent_id

    # ------------------------------------------------------------ Checkpoint
    def _update_best(self) -> None:
        """Giữ BẢN SAO trạng thái agent có Fitness cao nhất từng ghi nhận (không phải tham chiếu)."""
        for ag in self.agents:
            score = ag.fitness().score
            if self.best_agent is None or score > self.best_agent["fitness"]["score"]:
                self.best_agent = ag.describe() | {"tick_ghi_nhan": self.current_tick}

    def _fitness_stats(self) -> dict[str, Any]:
        scores = [ag.fitness().score for ag in self.agents]
        return {
            "so_agent_song": len(scores),
            "tot_nhat": round(max(scores), 6) if scores else None,
            "trung_binh": round(statistics.fmean(scores), 6) if scores else None,
            "trung_vi": round(statistics.median(scores), 6) if scores else None,
            "the_he_cao_nhat": max((ag.generation for ag in self.agents), default=0),
            "da_dang_gen": round(self.genetics.diversity(), 6),
        }

    def _market_state(self) -> dict[str, Any]:
        m = self.market
        state: dict[str, Any] = {
            "type": "live" if isinstance(m, LiveMarketFeed) else "simulator",
            "tick": m.tick,
            "regime": m.regime,
            "prices": list(m.prices),
            "returns": list(m.returns),
        }
        if isinstance(m, LiveMarketFeed):
            state["symbol"] = m.symbol
        return state

    def save_checkpoint(self) -> Path | None:
        """Ghi atomic trạng thái quần thể ra ``--checkpoint-file``. Lỗi ghi chỉ cảnh báo, không dừng mô phỏng."""
        self.memory.flush()
        self._update_best()
        data = {
            "tick": self.current_tick,
            "next_id": self._next_id,
            "seed": self.args.seed,
            "price_source": self.price_source,
            "memory_file": str(self.args.memory),
            "rng": {
                "ecosystem": ckpt.rng_state(self.rng),
                "market": ckpt.rng_state(self.market.rng),
                "engine": ckpt.rng_state(self.engine.rng),
                "genetics": ckpt.rng_state(self.genetics.rng),
            },
            "market": self._market_state(),
            "fitness_stats": self._fitness_stats(),
            "best_agent": self.best_agent,
            "agents": [ag.to_state() for ag in self.agents],
            "graveyard": [ag.to_state() for ag in self.graveyard],
        }
        try:
            path = ckpt.write_checkpoint(self.args.checkpoint_file, ckpt.KIND_ECOSYSTEM, data,
                                         durable=not self.args.no_fsync)
        except ckpt.CheckpointError as e:
            log.warning("⚠️  %s", e)
            return None
        self._saved_tick = self.current_tick
        log.info("💾 Checkpoint tick %d: %d agent sống, %d đã chết → %s",
                 self.current_tick, len(self.agents), len(self.graveyard), path)
        return path

    def _resume(self) -> bool:
        """Khôi phục từ ``--checkpoint-file``. Trả về False (khởi tạo mới) nếu không có tệp hoặc tệp hỏng."""
        path = self.args.checkpoint_file
        if not Path(path).exists():
            log.warning("⚠️  Không có checkpoint tại %s — khởi tạo quần thể mới.", path)
            return False
        data = ckpt.load_or_quarantine(path, ckpt.KIND_ECOSYSTEM, CHECKPOINT_FIELDS, report=log.warning)
        if data is None:
            return False
        try:
            agents = [Agent.from_state(s, self.memory, self.engine) for s in data["agents"]]
            graveyard = [Agent.from_state(s, self.memory, self.engine) for s in data["graveyard"]]
            rngs = data["rng"]
            restored = {name: ckpt.restore_rng(rngs[name]) for name in ("ecosystem", "market", "engine", "genetics")}
            market = data["market"]
            tick, next_id = int(data["tick"]), int(data["next_id"])
        except (KeyError, TypeError, ValueError) as e:
            backup = ckpt.quarantine(path)
            log.warning("⚠️  Checkpoint %s sai cấu trúc (%s: %s), đã chuyển sang %s — khởi tạo quần thể mới.",
                        path, type(e).__name__, e, backup)
            return False
        if not any(ag.is_alive for ag in agents):
            log.warning("⚠️  Quần thể trong checkpoint %s đã diệt vong — khởi tạo quần thể mới.", path)
            return False

        for name, target in (("ecosystem", self.rng), ("engine", self.engine.rng), ("genetics", self.genetics.rng)):
            target.setstate(restored[name].getstate())
        m = self.market
        m.tick = int(market["tick"])
        same_kind = (market["type"] == "live") == isinstance(m, LiveMarketFeed)
        if same_kind and not isinstance(m, LiveMarketFeed):
            m.rng.setstate(restored["market"].getstate())
            m.regime = market["regime"]
            m.prices = [float(p) for p in market["prices"]]
            m.returns = [float(r) for r in market["returns"]]
        m._snapshot = m._build_snapshot()
        self.agents = [ag for ag in agents if ag.is_alive]
        self.graveyard = graveyard + [ag for ag in agents if not ag.is_alive]
        self._next_id = next_id
        self.start_tick = self.current_tick = tick
        self.best_agent = data.get("best_agent")
        stats = data.get("fitness_stats") or {}
        log.info(
            "♻️  Khôi phục checkpoint %s (lưu lúc %s): tick %d, %d agent sống, %d đã chết, thế hệ cao nhất %s, "
            "Fitness tốt nhất %s. Trạng thái ngẫu nhiên lấy từ checkpoint (bỏ qua --seed).",
            path, data.get("timestamp", "?"), tick, len(self.agents), len(self.graveyard),
            stats.get("the_he_cao_nhat", "?"), stats.get("tot_nhat", "?"),
        )
        if not same_kind:
            log.warning("⚠️  Checkpoint dùng nguồn giá '%s', lần này nguồn giá khác — giữ chuỗi giá hiện tại.",
                        market["type"])
        return True

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
            fitness_threshold=self.args.fitness_threshold,
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
            "🚀 Bắt đầu mô phỏng: %d bước, tick %.2fs, seed=%s, tối đa %d agent, nguồn giá: %s. Bộ nhớ: %s "
            "(%d quy tắc, %d cái chết đã biết)",
            a.steps, a.tick, a.seed, a.max_agents,
            self.price_source,
            a.memory, mem["quy_tac"], mem["cai_chet"],
        )
        if self.checkpointing:
            log.info("💾 Checkpoint: %s (mỗi %s tick và khi kết thúc/Ctrl+C)%s",
                     a.checkpoint_file, a.checkpoint_every or "—",
                     f", tiếp tục từ tick {self.start_tick}" if self.start_tick else "")
        last = self.start_tick + a.steps
        for step in range(self.start_tick + 1, last + 1):
            if self._stop:
                break
            snap = self.market.step()
            newborns = self._newborns = []
            for agent in list(self.agents):
                room = len(self.agents) + len(newborns) < a.max_agents
                newborns += agent.step(snap, id_factory=self.next_id, can_replicate=room, genetics=self.genetics)
                if not agent.is_alive:
                    # Agent đã tự hủy: rút khỏi bộ lập lịch, "tiến trình" của nó kết thúc.
                    self.agents.remove(agent)
                    self.graveyard.append(agent)
            self.agents.extend(newborns)
            self._newborns = []
            if a.diversity_every > 0 and step % a.diversity_every == 0:
                self.genetics.enforce_diversity(step)

            self.current_tick = step
            if step % a.summary_every == 0 or step == last:
                self._print_status(step, snap)
            if not self.agents:
                log.warning("☠️  Toàn bộ hệ sinh thái đã diệt vong tại tick %d.", step)
                break
            if a.checkpoint_every > 0 and step % a.checkpoint_every == 0:
                self.save_checkpoint()
            elif step % 20 == 0:
                self.memory.flush()
            if a.tick > 0 and not self._stop:
                time.sleep(a.tick)
        self.memory.flush()
        if self.checkpointing and self._saved_tick != self.current_tick:
            self.save_checkpoint()
        self._print_report(self.current_tick)
        return 0

    def _print_status(self, step: int, snap) -> None:
        total = sum(ag.total_assets(snap) for ag in self.agents)
        mem = self.memory.summary()
        gens = max((ag.generation for ag in self.agents), default=0)
        best = max((ag.fitness().score for ag in self.agents), default=0.0)
        log.info(
            "📊 Tick %4d | giá %8.2f | chế độ %-11s | sống %2d, chết %2d | tổng vốn $%7.2f | "
            "thế hệ cao nhất %d | quy tắc %d (%d hiệu lực) | Fitness tốt nhất %.2f | đa dạng gen %.3f | "
            "biên độ đột biến %.2f",
            step, snap.price, TEN_CHE_DO[snap.regime], len(self.agents), len(self.graveyard),
            total, gens, mem["quy_tac"], mem["quy_tac_hieu_luc"], best, self.genetics.diversity(),
            self.genetics.mutation_rate(step),
        )

    def _print_report(self, step: int) -> None:
        everyone = sorted(self.agents + self.graveyard, key=lambda ag: ag.agent_id)
        print()
        width = 112
        print("=" * width)
        print(f" BÁO CÁO CUỐI CÙNG sau {step} tick")
        print("=" * width)
        print(f" {'ID':<6} {'Thế hệ':>6} {'Cha':<6} {'Trạng thái':<11} {'Vốn':>9} {'Fitness':>8} "
              f"{'Max DD':>7} {'Lệnh':>5} {'Thắng':>5} {'Thua':>5} {'Bị chặn':>7} {'Điều chỉnh':>10} {'Con':>4}")
        print("-" * width)
        for ag in everyone:
            s = ag.stats
            print(f" {ag.agent_id:<6} {ag.generation:>6} {ag.parent_id or '-':<6} {ag.state.value:<11} "
                  f"${ag.capital:>8.2f} {ag.fitness().score:>8.2f} {s.max_drawdown:>7.0%} {s.trades:>5} "
                  f"{s.wins:>5} {s.losses:>5} {s.blocked:>7} {s.adjusted:>10} {s.children:>4}")
        print("-" * width)
        mem = self.memory.summary()
        print(f" Sống: {len(self.agents)} | Đã chết: {len(self.graveyard)} | "
              f"Tổng vốn còn lại: ${sum(ag.capital for ag in self.agents):.2f} | "
              f"Đa dạng gen: {self.genetics.diversity():.3f}")
        print(f" Bộ nhớ chung ({self.args.memory}): {mem['quy_tac']} quy tắc "
              f"({mem['quy_tac_hieu_luc']} hiệu lực), {mem['that_bai']} hồ sơ thất bại, "
              f"{mem['cai_chet']} hồ sơ cái chết, {mem['pha_he']} sự kiện sinh sản.")
        rules = [r for r in self.memory.rules() if r.get("hieu_luc", True)]
        if rules:
            print(" Một số quy tắc sinh tồn đang hiệu lực:")
            for r in rules[-5:]:
                print(f"   • {r['id']} [{r['hanh_dong']}] {r['mo_ta']}")
        print("=" * width)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mô phỏng hệ sinh thái Autonomous Adaptive AI Agent (giao dịch giả lập).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=500, help="Số bước (tick) mô phỏng")
    p.add_argument("--tick", type=float, default=None,
                   help=f"Số giây nghỉ giữa hai tick, 0 = nhanh nhất (mặc định {DEFAULT_TICK_SIMULATED} khi giả lập, "
                        f"{DEFAULT_TICK_LIVE} khi --live-price)")
    p.add_argument("--seed", type=int, default=None, help="Hạt giống ngẫu nhiên để tái lập kết quả")
    p.add_argument("--agents", type=int, default=5, help="Số agent khởi sinh (thế hệ 0)")
    p.add_argument("--max-agents", "--population-cap", dest="max_agents", type=int, default=30,
                   help="Giới hạn số agent sống cùng lúc")
    p.add_argument("--operating-cost", type=float, default=0.02, help="Chi phí vận hành mỗi tick ($)")
    p.add_argument("--mutation-rate", type=float, default=0.1,
                   help="Biên độ đột biến cơ sở; thực tế tự co giãn theo áp lực thất bại trong bộ nhớ")
    p.add_argument("--fitness-threshold", type=float, default=FITNESS_THRESHOLD,
                   help="Điểm Fitness tối thiểu để được phép tự nhân bản")
    p.add_argument("--elite-size", type=int, default=3, help="Số agent Fitness cao nhất tham gia lai ghép")
    p.add_argument("--crossover-prob", type=float, default=0.7, help="Xác suất lai ghép thay vì tự đột biến")
    p.add_argument("--min-diversity", type=float, default=0.05,
                   help="Khoảng cách gen tối thiểu giữa hai agent (chống đồng huyết thống)")
    p.add_argument("--diversity-every", type=int, default=25,
                   help="Rà soát trùng lặp gen sau mỗi N tick (0 = tắt)")
    p.add_argument("--summary-every", type=int, default=25, help="In tóm tắt sau mỗi N tick")
    p.add_argument("--memory", type=Path, default=DEFAULT_MEMORY, help="Đường dẫn tệp bộ nhớ chung")
    p.add_argument("--reset-memory", action="store_true", help="Xoá sạch bộ nhớ trước khi chạy")
    p.add_argument("--live-price", action="store_true",
                   help="Dùng giá thị trường thật từ Binance thay cho thị trường giả lập (lệnh vẫn là mô phỏng)")
    p.add_argument("--symbol", type=str, default="BTCUSDT",
                   help="Cặp giao dịch khi bật --live-price (ví dụ: BTCUSDT, ETHUSDT)")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Lưu checkpoint quần thể sau mỗi N tick, kèm một lần khi kết thúc/Ctrl+C (0 = tắt)")
    p.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Tệp checkpoint quần thể (tách biệt với bộ nhớ chung)")
    p.add_argument("--resume", action="store_true",
                   help="Khôi phục quần thể từ --checkpoint-file rồi chạy tiếp --steps tick; "
                        "không có tệp hoặc tệp hỏng thì khởi tạo mới")
    p.add_argument("--no-fsync", action="store_true", help="Bỏ fsync khi ghi (nhanh hơn, kém bền hơn)")
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="In chi tiết từng lệnh/Post-Mortem")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Chỉ in cảnh báo và báo cáo cuối")
    args = p.parse_args(argv)
    if args.tick is None:
        args.tick = DEFAULT_TICK_LIVE if args.live_price else DEFAULT_TICK_SIMULATED
    if args.steps <= 0 or args.agents <= 0 or args.max_agents < args.agents or args.summary_every <= 0:
        p.error("Tham số không hợp lệ: cần steps > 0, agents > 0, summary-every > 0 và max-agents ≥ agents")
    if args.tick < 0:
        p.error("Tham số không hợp lệ: --tick phải ≥ 0")
    if args.elite_size < 1 or not 0 <= args.crossover_prob <= 1 or args.min_diversity < 0 or args.diversity_every < 0:
        p.error("Tham số di truyền không hợp lệ: elite-size ≥ 1, 0 ≤ crossover-prob ≤ 1, "
                "min-diversity ≥ 0, diversity-every ≥ 0")
    if args.checkpoint_every < 0:
        p.error("Tham số không hợp lệ: --checkpoint-every phải ≥ 0")
    if args.resume and args.reset_memory:
        p.error("Không dùng --resume cùng --reset-memory: quần thể khôi phục sẽ lệch với bộ nhớ chung đã bị xoá")
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
