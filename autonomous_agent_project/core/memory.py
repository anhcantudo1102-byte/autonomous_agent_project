"""Bộ nhớ Reflexion dùng chung cho toàn bộ hệ sinh thái agent.

Mọi agent (cha, con, cháu...) cùng đọc/ghi một tệp ``data/memory_bank.json``:

* ``that_bai``   – nhật ký các lệnh lỗ kèm ngữ cảnh và bài học (Post-Mortem).
* ``quy_tac``    – quy tắc sinh tồn: mô tả bằng tiếng Việt + điều kiện máy đọc được.
* ``cai_chet``   – hồ sơ các agent đã TERMINATED và nguyên nhân cái chết.
* ``pha_he``     – sự kiện sinh sản (cha → con, tham số trước/sau đột biến).
* ``thong_ke_tin_hieu`` – thắng/thua theo "chữ ký tín hiệu" để đánh giá quy tắc.

Ghi tệp luôn theo kiểu atomic: ghi ra tệp tạm cùng thư mục, fsync, rồi
``os.replace``. Thao tác đọc-sửa-ghi được bảo vệ bởi khoá luồng và khoá tệp
(``fcntl`` nếu hệ điều hành hỗ trợ) để nhiều tiến trình có thể dùng chung.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # Windows
    fcntl = None

log = logging.getLogger("memory")

PHIEN_BAN_DINH_DANG = 1

# Biên phân nhóm (bucket) dùng để khái quát hoá ngữ cảnh thất bại thành quy tắc.
VOL_EDGES = (0.0, 0.005, 0.01, 0.02, 0.04, math.inf)
MOM_EDGES = (-math.inf, -0.03, -0.01, 0.0, 0.01, 0.03, math.inf)

# Ngưỡng đúc kết quy tắc.
MIN_LOSSES_CANH_BAO = 4
LOSS_RATE_CANH_BAO = 0.6
MIN_LOSSES_CAM = 6
LOSS_RATE_CAM = 0.65
SEVERITY_DON_BAY = 0.2
MIN_SAMPLES_GO_BO = 8
FAILURE_COUNTER_HORIZON = 200  # số tick giữ lại trong bộ đếm thất bại theo tick

HANH_DONG_CAM = "CAM"
HANH_DONG_GIAM_QUY_MO = "GIAM_QUY_MO"
HANH_DONG_GIOI_HAN_DON_BAY = "GIOI_HAN_DON_BAY"

TEN_HUONG = {"LONG": "MUA (LONG)", "SHORT": "BÁN (SHORT)"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bucket(value: float, edges: tuple[float, ...]) -> tuple[float, float]:
    """Trả về khoảng [lo, hi) chứa ``value``."""
    for lo, hi in zip(edges, edges[1:]):
        if lo <= value < hi:
            return lo, hi
    return edges[-2], edges[-1]


def _fmt_bound(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:.3f}"


def signature_of(context: dict[str, Any]) -> str:
    """Chữ ký tín hiệu: hướng lệnh + nhóm biến động + nhóm động lượng."""
    vlo, vhi = bucket(float(context["volatility"]), VOL_EDGES)
    mlo, mhi = bucket(float(context["momentum"]), MOM_EDGES)
    return (
        f"{context['direction']}|vol:{_fmt_bound(vlo)}..{_fmt_bound(vhi)}"
        f"|mom:{_fmt_bound(mlo)}..{_fmt_bound(mhi)}"
    )


def _range_conditions(field: str, lo: float, hi: float) -> list[dict[str, Any]]:
    conds: list[dict[str, Any]] = []
    if not math.isinf(lo):
        conds.append({"truong": field, "toan_tu": ">=", "gia_tri": round(lo, 6)})
    if not math.isinf(hi):
        conds.append({"truong": field, "toan_tu": "<", "gia_tri": round(hi, 6)})
    return conds


def _range_text(name: str, lo: float, hi: float) -> str:
    if math.isinf(lo) and math.isinf(hi):
        return f"{name} bất kỳ"
    if math.isinf(lo):
        return f"{name} dưới {hi:+.1%}" if name == "động lượng" else f"{name} dưới {hi:.1%}"
    if math.isinf(hi):
        return f"{name} từ {lo:+.1%} trở lên" if name == "động lượng" else f"{name} từ {lo:.1%} trở lên"
    if name == "động lượng":
        return f"{name} từ {lo:+.1%} đến {hi:+.1%}"
    return f"{name} từ {lo:.1%} đến {hi:.1%}"


def describe_context(context: dict[str, Any]) -> str:
    """Mô tả ngắn gọn một ngữ cảnh giao dịch bằng tiếng Việt."""
    return (
        f"{TEN_HUONG.get(context['direction'], context['direction'])} đòn bẩy "
        f"{context['leverage']:.1f}x, tỷ trọng {context['fraction']:.0%}, "
        f"biến động {context['volatility']:.2%}, động lượng {context['momentum']:+.2%}"
    )


@dataclass(frozen=True)
class MemoryView:
    """Ảnh chụp bộ nhớ vừa đọc từ đĩa — bằng chứng agent đã "tra cứu vết xe đổ"."""

    tick: int
    rules: tuple[dict[str, Any], ...]
    data_version: int

    def active_rules(self) -> list[dict[str, Any]]:
        return [r for r in self.rules if r.get("hieu_luc", True)]


@dataclass
class PostMortem:
    """Kết quả mổ xẻ một lệnh lỗ."""

    agent_id: str
    tick: int
    signature: str
    loss: float
    severity: float
    lesson: str
    rule_events: list[tuple[str, dict[str, Any]]]


class MemoryBank:
    """Kho ký ức Reflexion dùng chung, lưu bền vững trong một tệp JSON."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        durable: bool = True,
        max_failures: int = 300,
        max_deaths: int = 200,
        max_births: int = 500,
        compact: bool = False,
    ) -> None:
        self.path = Path(path)
        self.durable = durable
        self.compact = compact
        self.max_failures = max_failures
        self.max_deaths = max_deaths
        self.max_births = max_births
        self.read_count = 0
        self.write_count = 0
        self._lock = threading.RLock()
        # Kết quả thắng được gom lại trong RAM, ghi kèm lần ghi kế tiếp để tránh
        # phải ghi đĩa sau mỗi lệnh có lãi.
        self._pending: dict[str, dict[str, float]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._atomic_write(self._empty())

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "phien_ban_dinh_dang": PHIEN_BAN_DINH_DANG,
            "phien_ban_du_lieu": 0,
            "tao_luc": _now(),
            "cap_nhat_luc": _now(),
            "bo_dem_quy_tac": 0,
            "quy_tac": [],
            "that_bai": [],
            "cai_chet": [],
            "pha_he": [],
            "thong_ke_tin_hieu": {},
            "that_bai_theo_tick": {},
        }

    def reset(self) -> None:
        """Xoá sạch ký ức (dùng cho tham số ``--reset-memory``)."""
        with self._lock:
            self._pending.clear()
            self._atomic_write(self._empty())
        log.warning("Đã RESET bộ nhớ chung: %s", self.path)

    def load(self) -> dict[str, Any]:
        """Đọc toàn bộ tệp bộ nhớ; tệp hỏng sẽ được sao lưu và khởi tạo lại."""
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return self._empty()
        except json.JSONDecodeError:
            backup = self.path.with_suffix(".corrupt.json")
            os.replace(self.path, backup)
            log.error("Tệp bộ nhớ hỏng, đã sao lưu sang %s và khởi tạo lại", backup)
            return self._empty()
        base = self._empty()
        base.update(data)
        return base

    def _atomic_write(self, data: dict[str, Any]) -> None:
        data["cap_nhat_luc"] = _now()
        fd, tmp = tempfile.mkstemp(prefix=".memory_", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                if self.compact:
                    # json.dumps không thụt lề mới dùng bộ mã hoá C; json.dump luôn dùng bản Python chậm.
                    fh.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
                else:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                if self.durable:
                    os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.write_count += 1

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        if fcntl is None:
            yield
            return
        lock_path = self.path.with_name(self.path.name + ".lock")
        with open(lock_path, "a+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        """Đọc-sửa-ghi nguyên tử: luôn làm việc trên bản mới nhất trên đĩa."""
        with self._lock, self._file_lock():
            data = self.load()
            self._merge_pending(data)
            yield data
            data["phien_ban_du_lieu"] = int(data.get("phien_ban_du_lieu", 0)) + 1
            self._atomic_write(data)

    # ------------------------------------------------------------- Đọc
    def read_view(self, tick: int) -> MemoryView:
        """BẮT BUỘC gọi trước mỗi hành động kiếm tiền: đọc lại tệp từ đĩa."""
        with self._lock:
            data = self.load()
            self.read_count += 1
        return MemoryView(
            tick=tick,
            rules=tuple(data.get("quy_tac", [])),
            data_version=int(data.get("phien_ban_du_lieu", 0)),
        )

    def rules(self) -> list[dict[str, Any]]:
        return list(self.load().get("quy_tac", []))

    def recent_failures(self, tick: int, window: int) -> int:
        """Số lệnh lỗ toàn hệ sinh thái trong ``window`` tick gần nhất (tính cả tick hiện tại)."""
        counter = self.load().get("that_bai_theo_tick", {})
        return sum(int(n) for k, n in counter.items() if tick - window < int(k) <= tick)

    def summary(self) -> dict[str, int]:
        data = self.load()
        rules = data.get("quy_tac", [])
        return {
            "quy_tac": len(rules),
            "quy_tac_hieu_luc": sum(1 for r in rules if r.get("hieu_luc", True)),
            "that_bai": len(data.get("that_bai", [])),
            "cai_chet": len(data.get("cai_chet", [])),
            "pha_he": len(data.get("pha_he", [])),
        }

    # -------------------------------------------------------- Thống kê
    def record_win(self, context: dict[str, Any], pnl: float) -> None:
        """Ghi nhận lệnh có lãi (gom trong RAM, ghi đĩa ở lần ghi kế tiếp)."""
        sig = signature_of(context)
        with self._lock:
            p = self._pending.setdefault(sig, {"thang": 0, "thua": 0, "lai_lo_rong": 0.0})
            p["thang"] += 1
            p["lai_lo_rong"] += pnl

    def flush(self) -> None:
        """Ghi các thống kê đang chờ xuống đĩa."""
        with self._lock:
            if not self._pending:
                return
            with self._transaction():
                pass

    def _merge_pending(self, data: dict[str, Any]) -> None:
        if not self._pending:
            return
        stats = data.setdefault("thong_ke_tin_hieu", {})
        for sig, delta in self._pending.items():
            s = stats.setdefault(sig, {"thang": 0, "thua": 0, "lai_lo_rong": 0.0})
            s["thang"] += int(delta["thang"])
            s["thua"] += int(delta["thua"])
            s["lai_lo_rong"] = round(s["lai_lo_rong"] + delta["lai_lo_rong"], 6)
        self._pending.clear()
        self._reevaluate_soft_rules(data)

    @staticmethod
    def _reevaluate_soft_rules(data: dict[str, Any]) -> None:
        """Gỡ bỏ quy tắc "giảm quy mô" nếu bằng chứng mới cho thấy tín hiệu đã có lãi."""
        stats = data.get("thong_ke_tin_hieu", {})
        for rule in data.get("quy_tac", []):
            if rule.get("hanh_dong") != HANH_DONG_GIAM_QUY_MO or not rule.get("hieu_luc", True):
                continue
            s = stats.get(rule.get("chu_ky", ""))
            if not s:
                continue
            n = s["thang"] + s["thua"]
            if n >= MIN_SAMPLES_GO_BO and s["thua"] / n < 0.5 and s["lai_lo_rong"] > 0:
                rule["hieu_luc"] = False
                rule["ghi_chu"] = (
                    f"Tự gỡ bỏ: bằng chứng mới cho thấy tín hiệu thắng {s['thang']}/{n} lần, "
                    f"lãi ròng ${s['lai_lo_rong']:.2f}."
                )
                log.info("♻️  Gỡ bỏ quy tắc %s vì tín hiệu đã phục hồi", rule["id"])

    # ------------------------------------------------------ Quy tắc
    @staticmethod
    def _find_rule(data: dict[str, Any], key: str) -> dict[str, Any] | None:
        for rule in data["quy_tac"]:
            if rule.get("khoa") == key:
                return rule
        return None

    @staticmethod
    def _new_rule_id(data: dict[str, Any]) -> str:
        data["bo_dem_quy_tac"] = int(data.get("bo_dem_quy_tac", 0)) + 1
        return f"QT-{data['bo_dem_quy_tac']:04d}"

    def _signal_conditions(self, context: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
        vlo, vhi = bucket(float(context["volatility"]), VOL_EDGES)
        mlo, mhi = bucket(float(context["momentum"]), MOM_EDGES)
        conds = [{"truong": "direction", "toan_tu": "==", "gia_tri": context["direction"]}]
        conds += _range_conditions("volatility", vlo, vhi)
        conds += _range_conditions("momentum", mlo, mhi)
        return conds, _range_text("biến động", vlo, vhi), _range_text("động lượng", mlo, mhi)

    def _upsert_signal_rule(
        self,
        data: dict[str, Any],
        *,
        context: dict[str, Any],
        action: str,
        agent_id: str,
        tick: int,
        description: str,
        severity_label: str,
        evidence: dict[str, Any],
    ) -> tuple[str, dict[str, Any]] | None:
        """Tạo mới hoặc nâng cấp quy tắc cho một chữ ký tín hiệu."""
        sig = signature_of(context)
        key = f"tin_hieu::{sig}"
        conds, _, _ = self._signal_conditions(context)
        params = {"he_so": 0.5} if action == HANH_DONG_GIAM_QUY_MO else {}
        rule = self._find_rule(data, key)
        if rule is None:
            rule = {
                "id": self._new_rule_id(data),
                "khoa": key,
                "chu_ky": sig,
                "mo_ta": description,
                "dieu_kien": conds,
                "hanh_dong": action,
                "tham_so": params,
                "muc_do": severity_label,
                "bang_chung": evidence,
                "nguon_agent": agent_id,
                "tao_luc_tick": tick,
                "cap_nhat_tick": tick,
                "hieu_luc": True,
            }
            data["quy_tac"].append(rule)
            return "tao_moi", rule
        rank = {HANH_DONG_GIAM_QUY_MO: 1, HANH_DONG_CAM: 2}
        upgraded = rank.get(action, 0) > rank.get(rule["hanh_dong"], 0)
        reactivated = not rule.get("hieu_luc", True)
        if not (upgraded or reactivated or rule["hanh_dong"] == action):
            return None
        rule.update(
            mo_ta=description,
            bang_chung=evidence,
            cap_nhat_tick=tick,
            hieu_luc=True,
        )
        if upgraded:
            rule.update(hanh_dong=action, tham_so=params, muc_do=severity_label)
        rule.pop("ghi_chu", None)
        return ("nang_cap" if upgraded or reactivated else "cung_co"), rule

    def _upsert_leverage_rule(
        self, data: dict[str, Any], *, context: dict[str, Any], severity: float, agent_id: str, tick: int
    ) -> tuple[str, dict[str, Any]] | None:
        vlo, _ = bucket(float(context["volatility"]), VOL_EDGES)
        cap = max(1.0, round(float(context["leverage"]) * 0.6, 2))
        key = f"don_bay::vol>={vlo:.3f}"
        rule = self._find_rule(data, key)
        if rule is not None and rule["tham_so"]["don_bay_toi_da"] <= cap:
            return None
        description = (
            f"Không dùng đòn bẩy quá {cap:.1f}x khi biến động từ {vlo:.1%} trở lên — "
            f"một lệnh {describe_context(context)} đã cuốn mất {severity:.0%} vốn."
        )
        conds = [
            {"truong": "volatility", "toan_tu": ">=", "gia_tri": round(vlo, 6)},
            {"truong": "leverage", "toan_tu": ">", "gia_tri": cap},
        ]
        if rule is None:
            rule = {
                "id": self._new_rule_id(data),
                "khoa": key,
                "chu_ky": key,
                "mo_ta": description,
                "dieu_kien": conds,
                "hanh_dong": HANH_DONG_GIOI_HAN_DON_BAY,
                "tham_so": {"don_bay_toi_da": cap},
                "muc_do": "nghiem_trong",
                "bang_chung": {"ty_le_mat_von": round(severity, 4)},
                "nguon_agent": agent_id,
                "tao_luc_tick": tick,
                "cap_nhat_tick": tick,
                "hieu_luc": True,
            }
            data["quy_tac"].append(rule)
            return "tao_moi", rule
        rule.update(mo_ta=description, dieu_kien=conds, tham_so={"don_bay_toi_da": cap}, cap_nhat_tick=tick)
        return "nang_cap", rule

    # ---------------------------------------------------- Post-Mortem
    def post_mortem(
        self,
        *,
        agent_id: str,
        tick: int,
        context: dict[str, Any],
        pnl: float,
        capital_before: float,
    ) -> PostMortem:
        """Mổ xẻ một lệnh lỗ: ghi ngữ cảnh, rút bài học, cập nhật quy tắc chung."""
        loss = -pnl
        severity = loss / capital_before if capital_before > 0 else 1.0
        sig = signature_of(context)
        events: list[tuple[str, dict[str, Any]]] = []
        with self._transaction() as data:
            stats = data["thong_ke_tin_hieu"].setdefault(sig, {"thang": 0, "thua": 0, "lai_lo_rong": 0.0})
            stats["thua"] += 1
            stats["lai_lo_rong"] = round(stats["lai_lo_rong"] + pnl, 6)
            n = stats["thang"] + stats["thua"]
            loss_rate = stats["thua"] / n

            lesson = (
                f"Lỗ ${loss:.2f} ({severity:.1%} vốn) khi {describe_context(context)}. "
                f"Tín hiệu này hiện thua {stats['thua']}/{n} lần, lãi/lỗ ròng ${stats['lai_lo_rong']:.2f}."
            )
            _, vol_txt, mom_txt = self._signal_conditions(context)
            huong = TEN_HUONG[context["direction"]]
            evidence = {"thua": stats["thua"], "thang": stats["thang"], "lai_lo_rong": stats["lai_lo_rong"]}

            if stats["thua"] >= MIN_LOSSES_CAM and loss_rate >= LOSS_RATE_CAM and stats["lai_lo_rong"] < 0:
                ev = self._upsert_signal_rule(
                    data,
                    context=context,
                    action=HANH_DONG_CAM,
                    agent_id=agent_id,
                    tick=tick,
                    description=(
                        f"Cấm {huong} khi {vol_txt} và {mom_txt} — vết xe đổ: thua "
                        f"{stats['thua']}/{n} lần, lỗ ròng ${-stats['lai_lo_rong']:.2f}."
                    ),
                    severity_label="nghiem_trong",
                    evidence=evidence,
                )
                if ev:
                    events.append(ev)
            elif (
                stats["thua"] >= MIN_LOSSES_CANH_BAO
                and loss_rate >= LOSS_RATE_CANH_BAO
                and stats["lai_lo_rong"] < 0
            ):
                ev = self._upsert_signal_rule(
                    data,
                    context=context,
                    action=HANH_DONG_GIAM_QUY_MO,
                    agent_id=agent_id,
                    tick=tick,
                    description=(
                        f"Thận trọng: khi định {huong} lúc {vol_txt} và {mom_txt}, chỉ dùng 50% "
                        f"quy mô vị thế — tín hiệu đã thua {stats['thua']}/{n} lần."
                    ),
                    severity_label="canh_bao",
                    evidence=evidence,
                )
                if ev:
                    events.append(ev)

            if severity >= SEVERITY_DON_BAY and float(context["leverage"]) > 1.5:
                ev = self._upsert_leverage_rule(
                    data, context=context, severity=severity, agent_id=agent_id, tick=tick
                )
                if ev:
                    events.append(ev)

            data["that_bai"].append(
                {
                    "agent_id": agent_id,
                    "tick": tick,
                    "chu_ky": sig,
                    "ngu_canh": context,
                    "lai_lo": round(pnl, 6),
                    "von_truoc": round(capital_before, 6),
                    "muc_nghiem_trong": round(severity, 4),
                    "bai_hoc": lesson,
                    "quy_tac_lien_quan": [r["id"] for _, r in events],
                }
            )
            del data["that_bai"][: -self.max_failures]
            counter = data.setdefault("that_bai_theo_tick", {})
            counter[str(tick)] = int(counter.get(str(tick), 0)) + 1
            for old in [k for k in counter if int(k) <= tick - FAILURE_COUNTER_HORIZON]:
                del counter[old]
        return PostMortem(agent_id, tick, sig, loss, severity, lesson, events)

    def record_death(
        self,
        *,
        agent_id: str,
        tick: int,
        cause: str,
        final_capital: float,
        generation: int,
        parent_id: str | None,
        params: dict[str, Any],
        stats: dict[str, Any],
        fatal_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Ghi nguyên nhân cái chết; lệnh chí mạng trở thành quy tắc CẤM."""
        rule: dict[str, Any] | None = None
        with self._transaction() as data:
            if fatal_context is not None:
                _, vol_txt, mom_txt = self._signal_conditions(fatal_context)
                sig = signature_of(fatal_context)
                s = data["thong_ke_tin_hieu"].get(sig, {"thang": 0, "thua": 0, "lai_lo_rong": 0.0})
                ev = self._upsert_signal_rule(
                    data,
                    context=fatal_context,
                    action=HANH_DONG_CAM,
                    agent_id=agent_id,
                    tick=tick,
                    description=(
                        f"Bài học từ cái chết của {agent_id}: tuyệt đối không "
                        f"{TEN_HUONG[fatal_context['direction']]} khi {vol_txt} và {mom_txt} — "
                        f"lệnh như vậy đã góp phần xoá sổ một agent."
                    ),
                    severity_label="chet_nguoi",
                    evidence={"thua": s["thua"], "thang": s["thang"], "lai_lo_rong": s["lai_lo_rong"]},
                )
                if ev:
                    rule = ev[1]
            data["cai_chet"].append(
                {
                    "agent_id": agent_id,
                    "tick": tick,
                    "the_he": generation,
                    "cha": parent_id,
                    "von_cuoi": round(final_capital, 6),
                    "nguyen_nhan": cause,
                    "tham_so_chien_luoc": params,
                    "thong_ke": stats,
                    "quy_tac_rut_ra": rule["id"] if rule else None,
                }
            )
            del data["cai_chet"][: -self.max_deaths]
        return rule

    def record_birth(
        self,
        *,
        parent_id: str,
        child_id: str,
        tick: int,
        generation: int,
        parent_params: dict[str, Any],
        child_params: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._transaction() as data:
            data["pha_he"].append(
                {
                    "tick": tick,
                    "cha": parent_id,
                    "con": child_id,
                    "the_he": generation,
                    "tham_so_cha": parent_params,
                    "tham_so_con": child_params,
                    **(extra or {}),
                }
            )
            del data["pha_he"][: -self.max_births]
