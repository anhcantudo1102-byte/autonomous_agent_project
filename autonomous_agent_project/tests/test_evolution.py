import unittest
from core.evolution import calculate_advanced_fitness


class TestAdvancedFitness(unittest.TestCase):
    def setUp(self):
        self.dummy_agent = {"id": 1, "genes": [0.5, 0.8]}

    def test_fitness_bonus_for_confluence_alignment(self):
        """Test trường hợp Agent mua/bán thuận theo Confluence Score cao -> Nhận thưởng điểm Fitness"""
        trades = [
            {"pnl": 50.0, "action": "BUY", "confluence_score": 3.0},
            {"pnl": 30.0, "action": "SELL", "confluence_score": -2.5},
        ]
        fitness = calculate_advanced_fitness(self.dummy_agent, trades)
        
        # PnL (80) + Bonus Mua (3.0 * 10 = 30) + Bonus Bán (2.5 * 10 = 25) = 135.0
        self.assertEqual(fitness, 135.0)

    def test_fitness_penalty_for_counter_trend_trades(self):
        """Test trường hợp Agent đánh ngược dòng tiền thông minh (SMC/Sóng 3) -> Bị phạt kỷ luật"""
        trades = [
            {"pnl": 10.0, "action": "BUY", "confluence_score": -3.0},  # Đánh Mua khi Score báo Bán
        ]
        fitness = calculate_advanced_fitness(self.dummy_agent, trades)
        
        # PnL (10) - Penalty (20) = -10 -> Ngưỡng tối thiểu là 0.1
        self.assertEqual(fitness, 0.1)

    def test_fitness_penalty_for_overtrading_in_noise(self):
        """Test trường hợp Agent giao dịch vô kỷ luật khi thị trường không rõ xu hướng (Score ~ 0)"""
        trades = [
            {"pnl": 0.0, "action": "BUY", "confluence_score": 0.1},
            {"pnl": 0.0, "action": "SELL", "confluence_score": -0.2},
        ]
        fitness = calculate_advanced_fitness(self.dummy_agent, trades)
        
        # PnL (0) - Penalty (5 + 5 = 10) = -10 -> Ngưỡng tối thiểu là 0.1
        self.assertEqual(fitness, 0.1)

    def test_fitness_non_negative_boundary(self):
        """Đảm bảo điểm Fitness luôn không âm (tối thiểu 0.1) để thuật toán chọn lọc hoạt động chuẩn"""
        trades = [
            {"pnl": -100.0, "action": "BUY", "confluence_score": -3.0},
        ]
        fitness = calculate_advanced_fitness(self.dummy_agent, trades)
        self.assertGreaterEqual(fitness, 0.1)


class Lenh:
    """Lệnh dạng đối tượng (không phải dict)."""

    def __init__(self, pnl=None, action=None, confluence_score=None):
        self.pnl, self.action, self.confluence_score = pnl, action, confluence_score


class TestAdvancedFitnessBien(unittest.TestCase):
    """Các ca biên bổ sung (quy ước ghi trong docstring của calculate_advanced_fitness)."""

    def fit(self, trades, **kw):
        return calculate_advanced_fitness({"id": 1}, trades, **kw)

    def test_khong_co_lenh_tra_ve_san(self):
        self.assertEqual(self.fit([]), 0.1)
        self.assertEqual(self.fit(None), 0.1)
        self.assertEqual(self.fit(iter(())), 0.1)

    def test_chi_tinh_pnl_khi_khong_co_chieu_hoac_khong_co_diem(self):
        trades = [
            {"pnl": 7.0, "action": "HOLD", "confluence_score": 3.0},
            {"pnl": 5.0, "action": "WAIT", "confluence_score": -4.0},
            {"pnl": 4.0, "confluence_score": 2.0},
            {"pnl": 3.0, "action": "BUY", "confluence_score": None},
            {"pnl": 2.0, "action": "SELL"},
            {"action": "BUY", "confluence_score": 0.0, "pnl": None},
        ]
        # 7 + 5 + 4 + 3 + 2 + 0, chỉ lệnh cuối (BUY, score 0 → nhiễu) bị phạt 5.
        self.assertEqual(self.fit(trades), 16.0)

    def test_action_khong_phan_biet_hoa_thuong_va_dong_nghia(self):
        trades = [
            {"pnl": 0, "action": "buy", "confluence_score": 1},
            {"pnl": 0, "action": "Long", "confluence_score": 1},
            {"pnl": 0, "action": "STRONG_BUY", "confluence_score": 1},
            {"pnl": 0, "action": "short", "confluence_score": -1},
            {"pnl": 0, "action": "strong_sell", "confluence_score": -1},
        ]
        self.assertEqual(self.fit(trades), 50.0)

    def test_nhieu_duoc_xet_truoc_nguoc_chieu(self):
        # Ngược chiều nhưng |score| < 0.5 → chỉ phạt nhiễu 5, không phạt 20.
        self.assertEqual(self.fit([{"pnl": 30.0, "action": "BUY", "confluence_score": -0.4}]), 25.0)
        self.assertEqual(self.fit([{"pnl": 30.0, "action": "SELL", "confluence_score": 0.49}]), 25.0)

    def test_dung_nguong_la_tin_hieu_ro(self):
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "BUY", "confluence_score": 0.5}]), 5.0)
        self.assertEqual(self.fit([{"pnl": 30.0, "action": "SELL", "confluence_score": 0.5}]), 10.0)

    def test_ban_nguoc_chieu_bi_phat(self):
        self.assertEqual(self.fit([{"pnl": 50.0, "action": "SELL", "confluence_score": 3.0}]), 30.0)

    def test_lo_va_san(self):
        self.assertEqual(self.fit([{"pnl": -100.0, "action": "BUY", "confluence_score": -3.0}]), 0.1)
        self.assertEqual(self.fit([{"pnl": -1.0, "action": "BUY", "confluence_score": 0.2}], floor=-50), -6.0)
        self.assertEqual(self.fit([{"pnl": 0.05}]), 0.1)

    def test_tham_so_tuy_chinh(self):
        trades = [
            {"pnl": 10.0, "action": "BUY", "confluence_score": 2.0},
            {"pnl": 10.0, "action": "BUY", "confluence_score": -2.0},
            {"pnl": 10.0, "action": "SELL", "confluence_score": -0.8},
        ]
        f = self.fit(trades, bonus_per_point=1.0, counter_penalty=3.0, noise_penalty=2.0, noise_threshold=1.0)
        self.assertEqual(f, 30.0 + 2.0 - 3.0 - 2.0)
        # Ngưỡng 0: chỉ score đúng bằng 0 mới là nhiễu.
        self.assertEqual(self.fit([{"pnl": 0, "action": "BUY", "confluence_score": 0.01}], noise_threshold=0), 0.1)
        self.assertEqual(self.fit([{"pnl": 10, "action": "BUY", "confluence_score": 0}], noise_threshold=0), 5.0)

    def test_lenh_dang_doi_tuong_va_lam_tron(self):
        trades = [Lenh(0.1, "BUY", 0.7), Lenh(0.2), Lenh()]
        self.assertEqual(self.fit(trades), 7.3)

    def test_gia_tri_khong_hop_le(self):
        bad = [
            {"pnl": "50", "action": "BUY", "confluence_score": 3.0},
            {"pnl": True},
            {"pnl": float("nan")},
            {"pnl": 1.0, "action": "BUY", "confluence_score": float("inf")},
            {"pnl": 1.0, "action": "BUY", "confluence_score": "3"},
        ]
        for trade in bad:
            with self.subTest(trade=trade), self.assertRaises(ValueError):
                self.fit([trade])
        with self.assertRaisesRegex(ValueError, "#1"):
            self.fit([{"pnl": 1.0}, {"pnl": None, "confluence_score": False, "action": "BUY"}])

    def test_tham_so_am_bi_tu_choi(self):
        for name in ("bonus_per_point", "counter_penalty", "noise_penalty", "noise_threshold"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.fit([], **{name: -1})

    def test_agent_khong_anh_huong(self):
        trades = [{"pnl": 50.0, "action": "BUY", "confluence_score": 3.0}]
        self.assertEqual(calculate_advanced_fitness(None, trades), calculate_advanced_fitness({"genes": [9]}, trades))


if __name__ == "__main__":
    unittest.main()
