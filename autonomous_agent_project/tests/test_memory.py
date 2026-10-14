"""Kiểm thử bộ nhớ Reflexion: ghi atomic, Post-Mortem, đúc kết quy tắc."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.memory import MemoryBank, signature_of

CTX = {"direction": "LONG", "volatility": 0.012, "momentum": 0.015, "leverage": 1.2, "fraction": 0.5, "price": 100.0}


class TestMemoryBank(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "data" / "memory_bank.json"
        self.memory = MemoryBank(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def lose(self, n: int, pnl: float = -0.5, ctx: dict | None = None) -> list:
        return [
            self.memory.post_mortem(agent_id="A-001", tick=i, context=ctx or CTX, pnl=pnl, capital_before=20.0)
            for i in range(n)
        ]

    def test_tao_tep_hop_le_va_khong_de_lai_tep_tam(self) -> None:
        self.lose(2)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["that_bai"]), 2)
        leftovers = [p.name for p in self.path.parent.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

    def test_tep_hong_duoc_sao_luu_va_khoi_phuc(self) -> None:
        self.path.write_text("{hỏng", encoding="utf-8")
        data = self.memory.load()
        self.assertEqual(data["quy_tac"], [])
        self.assertTrue(self.path.with_suffix(".corrupt.json").exists())

    def test_post_mortem_ghi_bai_hoc_tieng_viet(self) -> None:
        pm = self.lose(1)[0]
        self.assertIn("Lỗ $0.50", pm.lesson)
        self.assertIn("MUA (LONG)", pm.lesson)
        self.assertEqual(pm.signature, signature_of(CTX))
        failure = self.memory.load()["that_bai"][0]
        self.assertEqual(failure["ngu_canh"]["direction"], "LONG")

    def test_dong_ket_quy_tac_giam_quy_mo_roi_nang_len_cam(self) -> None:
        self.lose(3)
        self.assertEqual(self.memory.rules(), [])
        self.lose(1)
        rules = self.memory.rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["hanh_dong"], "GIAM_QUY_MO")
        self.assertIn("Thận trọng", rules[0]["mo_ta"])
        self.lose(2)
        rules = self.memory.rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["hanh_dong"], "CAM")
        self.assertIn("Cấm MUA (LONG)", rules[0]["mo_ta"])
        fields = {c["truong"] for c in rules[0]["dieu_kien"]}
        self.assertEqual(fields, {"direction", "volatility", "momentum"})

    def test_lo_nang_voi_don_bay_cao_sinh_quy_tac_don_bay(self) -> None:
        ctx = dict(CTX, leverage=4.0)
        pm = self.lose(1, pnl=-6.0, ctx=ctx)[0]
        self.assertEqual([r["hanh_dong"] for _, r in pm.rule_events], ["GIOI_HAN_DON_BAY"])
        self.assertAlmostEqual(pm.rule_events[0][1]["tham_so"]["don_bay_toi_da"], 2.4)

    def test_quy_tac_mem_tu_go_khi_tin_hieu_phuc_hoi(self) -> None:
        self.lose(4)
        self.assertTrue(self.memory.rules()[0]["hieu_luc"])
        for _ in range(6):
            self.memory.record_win(CTX, 1.0)
        self.memory.flush()
        rule = self.memory.rules()[0]
        self.assertFalse(rule["hieu_luc"])
        self.assertIn("Tự gỡ bỏ", rule["ghi_chu"])

    def test_bo_nho_dung_chung_giua_cac_the_hien(self) -> None:
        self.lose(4)
        other = MemoryBank(self.path)
        self.assertEqual(len(other.read_view(1).rules), 1)

    def test_reset_xoa_sach(self) -> None:
        self.lose(4)
        self.memory.reset()
        self.assertEqual(self.memory.summary()["quy_tac"], 0)
        self.assertEqual(self.memory.summary()["that_bai"], 0)


if __name__ == "__main__":
    unittest.main()
