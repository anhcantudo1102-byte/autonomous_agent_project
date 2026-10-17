"""Dashboard trực quan trên terminal cho mô phỏng, dựng bằng thư viện ``rich`` (tuỳ chọn).

* ``generate_dashboard_view(...)`` dựng một ``Layout`` từ dữ liệu thuần (không cần hệ sinh thái), giữ
  chữ ký của bản phác thảo ban đầu và nhận thêm bảng quần thể, sự kiện, thông số di truyền.
* ``EcosystemDashboard`` gắn vào ``main.Ecosystem``: theo dõi agent sinh/chết và quy tắc CẤM mới
  giữa các lần cập nhật, rồi hiển thị bằng ``rich.live.Live``.

Không cài ``rich`` thì module vẫn import được (``RICH_AVAILABLE = False``); ``main.py`` sẽ cảnh báo
và quay về log thường.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

try:
    from rich import box
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - phụ thuộc môi trường
    RICH_AVAILABLE = False

from .price_feed import format_price

if TYPE_CHECKING:
    from rich.console import Console as ConsoleType

NOT_AVAILABLE = "N/A"
MAX_AGENT_ROWS = 12
MAX_EVENTS = 8
CAUSE_PREVIEW = 70

STATE_STYLE = {"ACTIVE": "green", "TERMINATED": "red"}
EVENT_STYLE = {"sinh": "green", "chet": "red", "cam": "yellow"}
EVENT_ICON = {"sinh": "🧬", "chet": "💀", "cam": "🚫"}


class DashboardUnavailable(RuntimeError):
    """Không bật được dashboard (chưa cài rich, hoặc terminal không hỗ trợ)."""


@dataclass(frozen=True)
class AgentRow:
    agent_id: str
    generation: int
    capital: float
    fitness: float
    state: str
    parent_id: str | None = None


@dataclass(frozen=True)
class DashboardEvent:
    tick: int
    kind: str  # "sinh" | "chet" | "cam"
    text: str


@dataclass
class EcosystemSnapshot:
    """Mọi thứ dashboard cần cho một khung hình."""

    alive: int = 0
    dead: int = 0
    diversity: float | None = None
    mutation_rate: float | None = None
    price_source: str = "giả lập"
    regime: str | None = None
    total_steps: int | None = None
    total_capital: float | None = None
    rules_active: int | None = None
    checkpoint: str | None = None
    agents: list[AgentRow] = field(default_factory=list)
    events: list[DashboardEvent] = field(default_factory=list)


def _best_fields(best_agent: dict[str, Any] | None) -> tuple[str, str]:
    """(ID, fitness) của agent tốt nhất; chấp nhận ``fitness`` là số hoặc dict có ``score``."""
    if not best_agent:
        return NOT_AVAILABLE, "—"
    fitness = best_agent.get("fitness", 0.0)
    if isinstance(fitness, dict):
        fitness = fitness.get("score", 0.0)
    try:
        return str(best_agent.get("id", NOT_AVAILABLE)), f"{float(fitness):.2f}"
    except (TypeError, ValueError):
        return str(best_agent.get("id", NOT_AVAILABLE)), "—"


def _stats(population_stats: EcosystemSnapshot | dict[str, Any] | int | None) -> EcosystemSnapshot:
    """Chuẩn hoá ``population_stats``: số nguyên (bản gốc truyền 100) = số agent sống."""
    if isinstance(population_stats, EcosystemSnapshot):
        return population_stats
    if isinstance(population_stats, dict):
        known = EcosystemSnapshot.__dataclass_fields__
        return EcosystemSnapshot(**{k: v for k, v in population_stats.items() if k in known})
    if isinstance(population_stats, int):
        return EcosystemSnapshot(alive=population_stats)
    return EcosystemSnapshot()


def _require_rich() -> None:
    if not RICH_AVAILABLE:
        raise DashboardUnavailable("Chưa cài thư viện rich (pip install rich)")


def _agents_table(rows: list[AgentRow], alive: int = 0, limit: int = MAX_AGENT_ROWS) -> "Table":
    table = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold magenta", pad_edge=False)
    table.add_column("ID", no_wrap=True)
    table.add_column("Thế hệ", justify="right", no_wrap=True)
    table.add_column("Cha", no_wrap=True, style="dim")
    table.add_column("Vốn", justify="right", no_wrap=True)
    table.add_column("Fitness", justify="right", no_wrap=True)
    table.add_column("Trạng thái", no_wrap=True)
    ranked = sorted(rows, key=lambda r: r.fitness, reverse=True)
    for r in ranked[:limit]:
        cap_style = "green" if r.capital >= 20 else "yellow" if r.capital >= 5 else "red"
        table.add_row(
            r.agent_id,
            str(r.generation),
            r.parent_id or "-",
            f"[{cap_style}]${r.capital:,.2f}[/]",
            f"{r.fitness:+.2f}",
            f"[{STATE_STYLE.get(r.state, 'white')}]{r.state}[/]",
        )
    if not ranked:
        table.caption = ("Không có dữ liệu chi tiết từng agent" if alive else "Chưa có agent nào đang sống")
    elif len(ranked) > limit:
        table.add_row(f"[dim]… và {len(ranked) - limit} agent khác[/dim]", "", "", "", "", "")
    return table


def _events_panel(events: list[DashboardEvent]) -> "Panel":
    if not events:
        body: Any = Text("Chưa có sự kiện sinh, chết hay quy tắc CẤM mới.", style="dim")
    else:
        lines = []
        for ev in reversed(events[-MAX_EVENTS:]):
            line = Text(f"t{ev.tick:>5} {EVENT_ICON.get(ev.kind, '•')} ", style="dim")
            line.append(ev.text, style=EVENT_STYLE.get(ev.kind, "white"))
            lines.append(line)
        body = Group(*lines)
    return Panel(body, title="Sự kiện gần nhất", border_style="cyan")


def generate_dashboard_view(
    step: int,
    generation: int,
    symbol: str,
    current_price: float,
    price_change: float,
    population_stats: EcosystemSnapshot | dict[str, Any] | int | None = None,
    best_agent: dict[str, Any] | None = None,
    *,
    footer: str = "Nhấn Ctrl+C để dừng mô phỏng (checkpoint được lưu nếu đã bật).",
) -> "Layout":
    """Tạo Layout cho một khung hình. ``price_change`` tính bằng %; ``best_agent`` có thể là None."""
    _require_rich()
    stats = _stats(population_stats)
    best_id, best_fitness = _best_fields(best_agent)

    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
    layout["left"].split(Layout(name="market"), Layout(name="events", minimum_size=6))

    source_style = "green" if "thật" in stats.price_source else "yellow"
    layout["header"].update(Panel(
        f"[bold cyan]AUTONOMOUS SURVIVAL AGENT DASHBOARD[/bold cyan] | Cặp: [green]{symbol}[/green] | "
        f"Thế hệ: [yellow]{generation}[/yellow] | Nguồn giá: [{source_style}]{stats.price_source}[/]",
        style="blue",
    ))

    info = Table(show_header=True, header_style="bold magenta", expand=True, box=box.SIMPLE_HEAD)
    info.add_column("Chỉ số", style="dim", no_wrap=True)
    info.add_column("Giá trị", justify="right", no_wrap=True)
    step_text = f"{step}" if stats.total_steps is None else f"{step} / {stats.total_steps}"
    change_style = "green" if price_change > 0 else "red" if price_change < 0 else "white"
    info.add_row("Bước mô phỏng (Step)", step_text)
    info.add_row("Giá hiện tại", f"{format_price(current_price)} USDT")
    info.add_row("Biến động", f"[{change_style}]{price_change:+.2f}%[/]")
    if stats.regime:
        info.add_row("Chế độ thị trường", stats.regime)
    info.add_row("Agent sống / đã chết", f"[green]{stats.alive}[/] / [red]{stats.dead}[/]")
    if stats.total_capital is not None:
        info.add_row("Tổng vốn đang sống", f"${stats.total_capital:,.2f}")
    info.add_row("Đa dạng gen", NOT_AVAILABLE if stats.diversity is None else f"{stats.diversity:.3f}")
    info.add_row("Biên độ đột biến", NOT_AVAILABLE if stats.mutation_rate is None else f"{stats.mutation_rate:.2f}")
    if stats.rules_active is not None:
        info.add_row("Quy tắc hiệu lực", str(stats.rules_active))
    info.add_row("Agent xuất sắc nhất (ID)", f"ID: {best_id}")
    info.add_row("Fitness tối đa", best_fitness)
    layout["market"].size = info.row_count + 6  # viền panel + tiêu đề bảng + đường kẻ + lề box
    layout["market"].update(Panel(info, title="Trạng thái Thị trường & Agent"))
    layout["events"].update(_events_panel(stats.events))
    layout["right"].update(Panel(
        _agents_table(stats.agents, stats.alive), title=f"Quần thể đang sống ({stats.alive})", border_style="green"
    ))

    note = footer if stats.checkpoint is None else f"{footer} {stats.checkpoint}"
    layout["footer"].update(Panel(f"[dim]{note}[/dim]", style="white"))
    return layout


# ------------------------------------------------------------- Gắn vào main.py
class _SkipAgentLogs(logging.Filter):
    """Ẩn log từng agent khi dashboard chạy — sinh/chết đã có trong bảng sự kiện."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("agent.")


class EcosystemDashboard:
    """Dashboard ``rich.live.Live`` theo dõi một ``main.Ecosystem``.

    Dùng: ``start()`` → ``update(step, snap)`` sau mỗi tick → ``stop()`` (gọi trong ``finally``).
    Khung hình được dựng tối đa ``refresh_per_second`` lần mỗi giây để không làm chậm mô phỏng.
    """

    def __init__(
        self,
        eco: Any,
        *,
        console: "ConsoleType | None" = None,
        refresh_per_second: float = 4.0,
        require_terminal: bool = True,
    ) -> None:
        _require_rich()
        self.eco = eco
        self.console = console or Console()
        if require_terminal and (not self.console.is_terminal or self.console.is_dumb_terminal):
            raise DashboardUnavailable("Terminal hiện tại không hỗ trợ giao diện tương tác (không phải TTY)")
        self.refresh_per_second = refresh_per_second
        self.events: deque[DashboardEvent] = deque(maxlen=MAX_EVENTS * 4)
        self._known_alive: set[str] = {a.agent_id for a in eco.agents}
        self._known_dead: set[str] = {a.agent_id for a in eco.graveyard}
        self._known_rules: set[str] = {r["id"] for r in eco.memory.rules()}
        self._last_render = 0.0
        self._live: "Live | None" = None
        self._log_state: list[tuple[logging.Handler, int]] = []
        self._log_handler: logging.Handler | None = None
        self.frames = 0

    # ---------------------------------------------------------- Vòng đời
    def start(self) -> None:
        self._redirect_logging()
        self._live = Live(self.render(self.eco.current_tick, None), console=self.console,
                          refresh_per_second=self.refresh_per_second, transient=False)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.update(self.render(self.eco.current_tick, None), refresh=True)
            self._live.stop()
            self._live = None
        self._restore_logging()

    def __enter__(self) -> "EcosystemDashboard":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _redirect_logging(self) -> None:
        """Log cảnh báo đi qua console của dashboard (in phía trên khung Live, không làm vỡ giao diện)."""
        root = logging.getLogger()
        for h in root.handlers:
            self._log_state.append((h, h.level))
            h.setLevel(logging.CRITICAL + 1)
        handler = RichHandler(console=self.console, level=logging.WARNING, show_path=False, markup=False)
        handler.addFilter(_SkipAgentLogs())
        root.addHandler(handler)
        self._log_handler = handler

    def _restore_logging(self) -> None:
        root = logging.getLogger()
        if self._log_handler is not None:
            root.removeHandler(self._log_handler)
            self._log_handler = None
        for h, level in self._log_state:
            h.setLevel(level)
        self._log_state = []

    # ---------------------------------------------------------- Cập nhật
    def update(self, step: int, snap: Any = None, *, force: bool = False) -> None:
        self.collect_events(step)
        now = time.monotonic()
        if self._live is None or (not force and now - self._last_render < 1.0 / self.refresh_per_second):
            return
        self._last_render = now
        self._live.update(self.render(step, snap))

    def collect_events(self, step: int) -> list[DashboardEvent]:
        """So với lần trước: agent mới sinh, agent vừa chết, quy tắc CẤM mới trong bộ nhớ chung."""
        eco = self.eco
        new: list[DashboardEvent] = []
        for a in eco.agents:
            if a.agent_id not in self._known_alive and a.agent_id not in self._known_dead:
                self._known_alive.add(a.agent_id)
                parent = f" từ {a.parent_id}" if a.parent_id else ""
                new.append(DashboardEvent(step, "sinh", f"{a.agent_id} ra đời{parent} (thế hệ {a.generation})"))
        for a in eco.graveyard:
            if a.agent_id not in self._known_dead:
                self._known_dead.add(a.agent_id)
                self._known_alive.discard(a.agent_id)
                # Câu đầu của death_cause là "Vốn cạn còn $x … tại tick n", câu thứ hai mới là nguyên nhân.
                sentences = (a.death_cause or "không rõ nguyên nhân").split(". ")
                cause = sentences[1] if len(sentences) > 1 else sentences[0]
                if len(cause) > CAUSE_PREVIEW:
                    cause = cause[: CAUSE_PREVIEW - 1] + "…"
                new.append(DashboardEvent(a.death_tick or step, "chet", f"{a.agent_id} tự hủy: {cause}"))
        for r in eco.memory.rules():
            if r["id"] in self._known_rules:
                continue
            self._known_rules.add(r["id"])
            if r.get("hanh_dong") == "CAM":
                desc = r.get("mo_ta", "")
                desc = desc if len(desc) <= CAUSE_PREVIEW + 30 else desc[: CAUSE_PREVIEW + 29] + "…"
                new.append(DashboardEvent(step, "cam", f"Quy tắc CẤM mới {r['id']}: {desc}"))
        self.events.extend(new)
        return new

    def snapshot(self, step: int) -> EcosystemSnapshot:
        eco = self.eco
        rows = [
            AgentRow(a.agent_id, a.generation, a.capital, a.fitness().score, a.state.value, a.parent_id)
            for a in eco.agents
        ]
        cp = None
        if getattr(eco, "checkpointing", False):
            saved = getattr(eco, "_saved_tick", None)
            cp = f"Checkpoint: {eco.args.checkpoint_file.name}" + (f", lần lưu cuối tick {saved}." if saved else ".")
        return EcosystemSnapshot(
            alive=len(eco.agents),
            dead=len(eco.graveyard),
            diversity=eco.genetics.diversity(),
            mutation_rate=eco.genetics.mutation_rate(step),
            price_source=eco.price_source,
            total_steps=eco.start_tick + eco.args.steps,
            total_capital=sum(a.capital for a in eco.agents),
            rules_active=eco.memory.summary()["quy_tac_hieu_luc"],
            checkpoint=cp,
            agents=rows,
            events=list(self.events),
        )

    def render(self, step: int, snap: Any = None) -> "Layout":
        from .strategy import TEN_CHE_DO

        eco = self.eco
        eco._update_best()
        snap = snap or eco.market.snapshot()
        stats = self.snapshot(step)
        stats.regime = TEN_CHE_DO.get(snap.regime, snap.regime)
        symbol = getattr(eco.market, "symbol", None) or getattr(eco, "symbol", None) or "GIẢ LẬP"
        self.frames += 1
        return generate_dashboard_view(
            step=step,
            generation=max((a.generation for a in eco.agents), default=0),
            symbol=symbol,
            current_price=snap.price,
            price_change=snap.ret_1 * 100,
            population_stats=stats,
            best_agent=eco.best_agent,
        )


def render_to_text(renderable: Any, *, width: int = 140, height: int = 40) -> str:
    """Render ra chuỗi (không cần TTY) — dùng cho test và xuất ảnh chụp."""
    _require_rich()
    console = Console(width=width, height=height, record=True, force_terminal=False, color_system=None,
                      file=_NullFile())
    console.print(renderable)
    return console.export_text()


class _NullFile:
    def write(self, _: str) -> int:
        return 0

    def flush(self) -> None:
        pass


def sample_rows(rows: Iterable[tuple[str, int, float, float]]) -> list[AgentRow]:
    """Tiện ích dựng ``AgentRow`` cho demo: (ID, thế hệ, vốn, fitness)."""
    return [AgentRow(i, g, c, f, "ACTIVE") for i, g, c, f in rows]
