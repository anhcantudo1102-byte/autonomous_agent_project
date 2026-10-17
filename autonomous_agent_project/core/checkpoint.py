"""Lưu / khôi phục checkpoint của quá trình tiến hoá ra tệp JSON.

So với bản phác thảo ban đầu, module này:

* ghi **atomic**: ghi ra tệp tạm cùng thư mục → ``fsync`` → ``os.replace``. Bị ngắt giữa chừng
  (Ctrl+C, mất điện) thì tệp checkpoint cũ vẫn nguyên vẹn, không bao giờ bị ghi dở;
* đặt thư mục mặc định theo **thư mục dự án** (``autonomous_agent_project/checkpoints``), không phụ
  thuộc thư mục đang đứng khi chạy lệnh;
* gắn trường ``schema_version`` để nhận ra checkpoint của phiên bản khác;
* lưu **bản sao sâu** của dữ liệu tại thời điểm gọi, nên sửa quần thể sau đó không ảnh hưởng tới
  checkpoint (và ``best_agent`` không bị "trôi" theo quần thể);
* tệp hỏng (JSON lỗi, thiếu trường, sai kiểu) được đổi tên thành ``*.corrupt.json`` để điều tra,
  rồi trả về ``None`` cho chương trình khởi tạo lại từ đầu thay vì sập.

``memory_bank.json`` (bộ nhớ Reflexion) là tệp riêng, module này không đụng tới.
"""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_FILENAME = "latest_checkpoint.json"
SCHEMA_VERSION = 1

KIND_EVOLUTION = "tien_hoa_demo"
KIND_ECOSYSTEM = "he_sinh_thai"

Report = Callable[[str], Any]


class CheckpointError(Exception):
    """Checkpoint không đọc/ghi được hoặc không đúng định dạng."""


def resolve_path(filename: str | os.PathLike[str] = DEFAULT_FILENAME, directory: str | os.PathLike[str] | None = None) -> Path:
    """Đường dẫn tuyệt đối của tệp checkpoint.

    ``filename`` tuyệt đối thì dùng nguyên; còn lại ghép vào ``directory``
    (mặc định ``CHECKPOINT_DIR`` trong thư mục dự án).
    """
    path = Path(filename)
    if path.is_absolute():
        return path
    return Path(directory if directory is not None else CHECKPOINT_DIR) / path


# ---------------------------------------------------------------- Đọc / ghi
def write_checkpoint(path: str | os.PathLike[str], kind: str, data: dict[str, Any], *, durable: bool = True) -> Path:
    """Ghi atomic ``data`` (kèm phiên bản schema, loại, thời gian) ra ``path``. Lỗi → ``CheckpointError``."""
    path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **copy.deepcopy(data),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                if durable:
                    os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except (OSError, TypeError, ValueError) as e:
        raise CheckpointError(f"Không ghi được checkpoint {path}: {e}") from e
    return path


def read_checkpoint(path: str | os.PathLike[str], kind: str, required: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Đọc checkpoint. Không có tệp → ``None``. Hỏng / sai định dạng → ``CheckpointError``."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CheckpointError(f"Tệp checkpoint {path} hỏng: {e}") from e
    if not isinstance(data, dict):
        raise CheckpointError(f"Tệp checkpoint {path} không phải đối tượng JSON")
    version = data.get("schema_version", 0)
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        raise CheckpointError(
            f"Checkpoint {path} có schema_version={version!r}, phiên bản này chỉ đọc được ≤ {SCHEMA_VERSION}"
        )
    found_kind = data.get("kind", KIND_EVOLUTION if version == 0 else None)
    if found_kind != kind:
        raise CheckpointError(f"Checkpoint {path} thuộc loại {found_kind!r}, cần loại {kind!r}")
    missing = [k for k in required if k not in data]
    if missing:
        raise CheckpointError(f"Checkpoint {path} thiếu trường: {', '.join(missing)}")
    return data


def quarantine(path: str | os.PathLike[str]) -> Path | None:
    """Đổi tên tệp hỏng thành ``<tên>.corrupt.json`` (giữ lại để điều tra). Trả về đường dẫn mới."""
    path = Path(path)
    target = path.with_name(f"{path.stem}.corrupt{path.suffix or '.json'}")
    try:
        os.replace(path, target)
    except OSError:
        return None
    return target


def load_or_quarantine(
    path: str | os.PathLike[str], kind: str, required: tuple[str, ...] = (), *, report: Report | None = None
) -> dict[str, Any] | None:
    """Như ``read_checkpoint`` nhưng không ném lỗi: tệp hỏng được cách ly và trả về ``None``."""
    report = report or print
    try:
        return read_checkpoint(path, kind, required)
    except CheckpointError as e:
        if "schema_version" in str(e) or "thuộc loại" in str(e):
            report(f"[Checkpoint] {e}. Bỏ qua, giữ nguyên tệp.")
            return None
        backup = quarantine(path)
        where = f", đã chuyển sang {backup}" if backup else ""
        report(f"[Checkpoint] {e}{where}. Khởi tạo mới từ đầu.")
        return None


# ------------------------------------------------- API cho demo tiến hoá
EVOLUTION_FIELDS = ("generation", "population", "best_agent")


def save_checkpoint(
    generation: int,
    population: list[dict[str, Any]],
    best_agent: dict[str, Any] | None,
    filename: str | os.PathLike[str] = DEFAULT_FILENAME,
    *,
    directory: str | os.PathLike[str] | None = None,
    extra: dict[str, Any] | None = None,
    durable: bool = True,
    report: Report | None = None,
) -> bool:
    """Lưu trạng thái quần thể và agent tốt nhất của thế hệ ``generation``. Trả về True nếu thành công.

    ``extra``: các trường bổ sung (ví dụ trạng thái RNG), không được trùng tên trường chính.
    """
    report = report or print
    path = resolve_path(filename, directory)
    data = {**(extra or {}), "generation": generation, "population": population, "best_agent": best_agent}
    try:
        write_checkpoint(path, KIND_EVOLUTION, data, durable=durable)
    except CheckpointError as e:
        report(f"[Checkpoint] Lỗi khi lưu checkpoint: {e}")
        return False
    report(f"[Checkpoint] Đã lưu thành công trạng thái thế hệ {generation} vào {path}")
    return True


def load_checkpoint(
    filename: str | os.PathLike[str] = DEFAULT_FILENAME,
    *,
    directory: str | os.PathLike[str] | None = None,
    report: Report | None = None,
) -> dict[str, Any] | None:
    """Khôi phục checkpoint đã lưu. Không có tệp hoặc tệp hỏng → ``None`` (khởi tạo mới)."""
    report = report or print
    path = resolve_path(filename, directory)
    if not path.exists():
        report(f"[Checkpoint] Không tìm thấy file checkpoint tại {path}. Khởi tạo mới từ đầu.")
        return None
    data = load_or_quarantine(path, KIND_EVOLUTION, EVOLUTION_FIELDS, report=report)
    if data is None:
        return None
    if not isinstance(data["generation"], int) or not isinstance(data["population"], list):
        backup = quarantine(path)
        report(f"[Checkpoint] Tệp {path} sai kiểu dữ liệu, đã chuyển sang {backup}. Khởi tạo mới từ đầu.")
        return None
    report(f"[Checkpoint] Khôi phục thành công thế hệ {data['generation']} từ {path}")
    return data


# ------------------------------------------- Tiện ích tuần tự hoá trạng thái
def rng_state(rng: random.Random) -> list[Any]:
    """Trạng thái ``random.Random`` dạng JSON được (tuple → list)."""
    version, internal, gauss_next = rng.getstate()
    return [version, list(internal), gauss_next]


def restore_rng(state: list[Any]) -> random.Random:
    rng = random.Random()
    set_rng_state(rng, state)
    return rng


def set_rng_state(rng: random.Random, state: list[Any]) -> None:
    version, internal, gauss_next = state
    rng.setstate((version, tuple(internal), gauss_next))
