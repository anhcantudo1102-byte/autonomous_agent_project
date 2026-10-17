"""Demo dashboard độc lập (dữ liệu giả, không chạy mô phỏng thật).

    python dashboard_demo.py                     # 20 bước, mỗi bước 0.5 giây
    python dashboard_demo.py --steps 60 --delay 0.2

Bản đã sửa của ``run_dashboard_demo`` ban đầu: ``best_agent`` có thể là None ở các bước đầu (hiển
thị "N/A"), ``population_stats`` được dùng thật (số agent sống/chết, đa dạng gen, bảng quần thể),
có bảng sự kiện sinh/chết/quy tắc CẤM. Dashboard trong mô phỏng thật: ``python main.py --dashboard``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from core.dashboard import (
    RICH_AVAILABLE,
    AgentRow,
    DashboardEvent,
    EcosystemSnapshot,
    generate_dashboard_view,
)


class FakePopulation:
    """Quần thể giả: vốn đi ngẫu nhiên, đủ $40 thì sinh con, cạn $0.5 thì chết."""

    def __init__(self, rng: random.Random, size: int = 6) -> None:
        self.rng = rng
        self.next_id = 1
        self.agents: list[AgentRow] = []
        self.dead = 0
        self.events: list[DashboardEvent] = []
        self.best: dict | None = None
        for _ in range(size):
            self.agents.append(self._new(0, None, 20.0))

    def _new(self, generation: int, parent: str | None, capital: float) -> AgentRow:
        agent_id = f"A-{self.next_id:03d}"
        self.next_id += 1
        return AgentRow(agent_id, generation, capital, 0.0, "ACTIVE", parent)

    def step(self, tick: int, market_return: float) -> None:
        survivors: list[AgentRow] = []
        for a in self.agents:
            capital = max(0.0, a.capital * (1 + self.rng.gauss(market_return * 3, 0.12)) - 0.02)
            fitness = (capital - 20.0) / 20.0 / 0.1
            if capital <= 0.5:
                self.dead += 1
                self.events.append(DashboardEvent(tick, "chet", f"{a.agent_id} tự hủy: vốn cạn ${capital:.2f}"))
                continue
            if capital >= 40.0 and fitness >= 1.5:
                capital -= 20.0
                child = self._new(a.generation + 1, a.agent_id, 20.0)
                survivors.append(child)
                self.events.append(DashboardEvent(tick, "sinh", f"{child.agent_id} ra đời từ {a.agent_id} "
                                                               f"(thế hệ {child.generation})"))
            survivors.append(AgentRow(a.agent_id, a.generation, capital, fitness, "ACTIVE", a.parent_id))
        self.agents = survivors
        if self.rng.random() < 0.06:
            side, vol = self.rng.choice(("MUA (LONG)", "BÁN (SHORT)")), self.rng.choice(("0.5%–1.0%", "1.0%–2.0%", "> 2.0%"))
            self.events.append(DashboardEvent(tick, "cam", f"Quy tắc CẤM mới QT-{tick:04d}: cấm {side} khi biến động {vol}"))
        if tick >= 3 and self.agents:
            top = max(self.agents, key=lambda r: r.fitness)
            if self.best is None or top.fitness > self.best["fitness"]:
                self.best = {"id": top.agent_id, "fitness": top.fitness}

    def stats(self, total_steps: int, diversity: float) -> EcosystemSnapshot:
        return EcosystemSnapshot(
            alive=len(self.agents),
            dead=self.dead,
            diversity=diversity,
            mutation_rate=0.1 + 0.02 * min(self.dead, 10),
            price_source="giả lập (demo)",
            regime="Demo",
            total_steps=total_steps,
            total_capital=sum(a.capital for a in self.agents),
            agents=list(self.agents),
            events=self.events[-20:],
        )


def run_dashboard_demo(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Demo dashboard rich với dữ liệu giả.")
    p.add_argument("--steps", type=int, default=20, help="Số bước demo")
    p.add_argument("--delay", type=float, default=0.5, help="Số giây giữa hai bước")
    p.add_argument("--symbol", default="BTCUSDT", help="Cặp hiển thị trên tiêu đề")
    p.add_argument("--seed", type=int, default=7, help="Hạt giống ngẫu nhiên")
    args = p.parse_args(argv)
    if not RICH_AVAILABLE:
        print("⚠️  Chưa cài thư viện rich — chạy `pip install rich` rồi thử lại.", file=sys.stderr)
        return 1

    from rich.console import Console
    from rich.live import Live

    console = Console()
    console.print("[bold green]Đang khởi động Dashboard...[/bold green]")
    rng = random.Random(args.seed)
    population = FakePopulation(rng)
    price = 82000.0
    try:
        with Live(console=console, refresh_per_second=4) as live:
            for step in range(1, args.steps + 1):
                change = rng.gauss(0.0005, 0.004)
                price *= 1 + change
                population.step(step, change)
                generation = max((a.generation for a in population.agents), default=0)
                live.update(generate_dashboard_view(
                    step=step,
                    generation=generation,
                    symbol=args.symbol,
                    current_price=price,
                    price_change=change * 100,
                    population_stats=population.stats(args.steps, rng.uniform(0.2, 0.4)),
                    best_agent=population.best,
                    footer="Nhấn Ctrl+C để thoát demo.",
                ))
                time.sleep(args.delay)
    except KeyboardInterrupt:
        console.print("[yellow]Đã dừng demo.[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(run_dashboard_demo(sys.argv[1:]))
