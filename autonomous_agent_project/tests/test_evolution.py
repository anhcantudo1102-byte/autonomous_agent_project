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


from core.evolution import calculate_advanced_fitness_scaled  # noqa: E402


class TestAdvancedFitnessScaled(unittest.TestCase):
    """Fitness theo ROI (vốn 20): giá trị = 20 × (1 + pnl/20 + thưởng − phạt), sàn 0.1."""

    def fit(self, trades, **kw):
        return calculate_advanced_fitness_scaled({"id": 1}, trades, **kw)

    def test_chua_giao_dich_tra_ve_von_goc(self):
        self.assertEqual(self.fit([]), 20.0)
        self.assertEqual(self.fit(None), 20.0)
        self.assertEqual(self.fit(iter(())), 20.0)
        self.assertEqual(self.fit([], initial_capital=50), 50.0)

    def test_thuong_khi_thuan_chieu_manh(self):
        # 20 × (1 + 2/20 + 3 × 0.002) = 22.12
        self.assertEqual(self.fit([{"pnl": 2.0, "action": "BUY", "confluence_score": 3.0}]), 22.12)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "SELL", "confluence_score": -2.5}]), 20.1)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "BUY", "confluence_score": 2.0}]), 20.08)  # đúng ngưỡng

    def test_phat_khi_nguoc_chieu_manh(self):
        # 20 × (1 + 1/20 − 0.005) = 20.9
        self.assertEqual(self.fit([{"pnl": 1.0, "action": "BUY", "confluence_score": -2.0}]), 20.9)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "SELL", "confluence_score": 3.0}]), 19.9)

    def test_phat_giao_dich_trong_nhieu(self):
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "BUY", "confluence_score": 0.2}]), 19.95)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "SELL", "confluence_score": 0.0}]), 19.95)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "SELL", "confluence_score": -0.49}]), 19.95)

    def test_vung_giua_khong_thuong_khong_phat(self):
        self.assertEqual(self.fit([{"pnl": 0.3, "action": "BUY", "confluence_score": 1.0}]), 20.3)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "SELL", "confluence_score": 1.99}]), 20.0)
        self.assertEqual(self.fit([{"pnl": 0.0, "action": "BUY", "confluence_score": 0.5}]), 20.0)

    def test_ket_hop_nhieu_lenh(self):
        trades = [
            {"pnl": 1.5, "action": "SELL", "confluence_score": -2.5},  # +0.5%
            {"pnl": 1.0, "action": "SELL", "confluence_score": 3.0},   # −0.5%
            {"pnl": -0.5, "action": "BUY", "confluence_score": 0.2},   # −0.25%
        ]
        # 20 × (1 + 2/20 + 0.005 − 0.005 − 0.0025) = 21.95
        self.assertEqual(self.fit(trades), 21.95)

    def test_lo_nang_ve_san(self):
        self.assertEqual(self.fit([{"pnl": -30.0, "action": "BUY", "confluence_score": -2.0}]), 0.1)
        self.assertEqual(self.fit([{"pnl": -20.0}]), 0.1)
        self.assertEqual(self.fit([{"pnl": -30.0}], floor=-100), -10.0)

    def test_von_ban_dau_khac(self):
        # 100 × (1 + 10/100 + 0.006) = 110.6
        self.assertEqual(self.fit([{"pnl": 10.0, "action": "BUY", "confluence_score": 3.0}], initial_capital=100), 110.6)

    def test_khong_chieu_hoac_khong_diem_chi_tinh_pnl(self):
        trades = [
            {"pnl": 1.0, "action": "HOLD", "confluence_score": 0.0},
            {"pnl": 1.0, "confluence_score": 3.0},
            {"pnl": 1.0, "action": "BUY"},  # bản gốc coi thiếu điểm là 0.0 → phạt nhiễu
            {"pnl": 1.0, "action": "SELL", "confluence_score": None},  # bản gốc: TypeError
            {"pnl": None, "action": "WAIT", "confluence_score": -3.0},  # bản gốc: TypeError
        ]
        self.assertEqual(self.fit(trades), 24.0)

    def test_action_khong_phan_biet_hoa_thuong_va_dong_nghia(self):
        trades = [
            {"pnl": 0, "action": "buy", "confluence_score": 2},
            {"pnl": 0, "action": "Long", "confluence_score": 2},
            {"pnl": 0, "action": "strong_buy", "confluence_score": 2},
            {"pnl": 0, "action": "SHORT", "confluence_score": -2},
            {"pnl": 0, "action": "Strong_Sell", "confluence_score": 2},
        ]
        # 4 lệnh thuận × 0.4% − 1 lệnh ngược 0.5% = +1.1% → 20.22
        self.assertEqual(self.fit(trades), 20.22)

    def test_lenh_dang_doi_tuong(self):
        self.assertEqual(self.fit([Lenh(2.0, "BUY", 3.0), Lenh(), Lenh(None, "SELL", 0.1)]), 22.07)

    def test_tham_so_tuy_chinh(self):
        trades = [
            {"pnl": 0, "action": "BUY", "confluence_score": 1.0},    # thuận, ≥ 1 → +1%
            {"pnl": 0, "action": "SELL", "confluence_score": 1.5},   # ngược → −10%
            {"pnl": 0, "action": "SELL", "confluence_score": -0.2},  # nhiễu < 0.3 → −5%
            {"pnl": 0, "action": "SELL", "confluence_score": -0.6},  # vùng giữa
        ]
        f = self.fit(trades, strong_threshold=1.0, bonus_rate_per_point=0.01, counter_penalty_rate=0.1,
                     noise_penalty_rate=0.05, noise_threshold=0.3, floor=5.0)
        self.assertEqual(f, 17.2)
        # Ngưỡng mạnh 0: điểm 0 không được thưởng/phạt ngược chiều mà tính là nhiễu.
        self.assertEqual(self.fit([{"pnl": 0, "action": "BUY", "confluence_score": 0}], strong_threshold=0), 19.95)
        self.assertEqual(self.fit([{"pnl": 0, "action": "BUY", "confluence_score": 0.3}], strong_threshold=0), 20.012)

    def test_von_ban_dau_khong_hop_le(self):
        for capital in (0, -5, 0.0, float("nan"), float("inf"), "20", True, None):
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                self.fit([], initial_capital=capital)

    def test_gia_tri_lenh_khong_hop_le(self):
        for trade in ({"pnl": float("nan")}, {"pnl": "1"}, {"pnl": 1, "action": "BUY", "confluence_score": float("-inf")},
                      {"pnl": 1, "action": "BUY", "confluence_score": False}):
            with self.subTest(trade=trade), self.assertRaises(ValueError):
                self.fit([trade])

    def test_tham_so_am_bi_tu_choi(self):
        for name in ("strong_threshold", "bonus_rate_per_point", "counter_penalty_rate", "noise_penalty_rate",
                     "noise_threshold"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.fit([], **{name: -0.1})

    def test_khop_ban_goc_cua_nguoi_dung_tren_du_lieu_hop_le(self):
        def goc(trade_history, initial_capital=20.0):
            if not trade_history:
                return initial_capital
            roi = sum(t.get("pnl", 0.0) for t in trade_history) / initial_capital
            bonus = violations = 0.0
            for t in trade_history:
                a, s = t.get("action"), t.get("confluence_score", 0.0)
                if (a == "BUY" and s >= 2.0) or (a == "SELL" and s <= -2.0):
                    bonus += abs(s) * 0.002
                elif (a == "BUY" and s <= -2.0) or (a == "SELL" and s >= 2.0):
                    violations += 1.0
                elif a in ["BUY", "SELL"] and abs(s) < 0.5:
                    violations += 0.5
            return max(0.1, initial_capital * (1.0 + roi + bonus - violations * 0.005))

        rng = __import__("random").Random(3)
        for _ in range(200):
            trades = [{"pnl": round(rng.uniform(-3, 3), 2), "action": rng.choice(["BUY", "SELL", "HOLD"]),
                       "confluence_score": round(rng.uniform(-7.5, 7.5), 1)} for _ in range(rng.randint(0, 6))]
            self.assertAlmostEqual(self.fit(trades), goc(trades), places=6)


from core.evolution import calculate_production_fitness, max_drawdown_from_trades  # noqa: E402


def T(pnl, action=None, score=None):
    return {"pnl": pnl, "action": action, "confluence_score": score}


class TestProductionFitness(unittest.TestCase):
    """Fitness thương mại (vốn 20): vốn cuối ± 0.1%/0.2% vốn mỗi lệnh hợp lưu mạnh − phạt Max DD > 20%."""

    def fit(self, trades, **kw):
        return calculate_production_fitness({"id": 1}, trades, **kw)

    def test_chua_giao_dich_tra_ve_von_goc(self):
        self.assertEqual(self.fit([]), 20.0)
        self.assertEqual(self.fit(None), 20.0)
        self.assertEqual(self.fit([], initial_capital=50), 50.0)

    def test_thuong_lenh_thuan_chieu_manh(self):
        self.assertEqual(self.fit([T(2.0, "BUY", 3.0)]), 22.02)  # 22 + 0.1% × 20
        self.assertEqual(self.fit([T(0.0, "SELL", -2.5)]), 20.02)
        self.assertEqual(self.fit([T(0.0, "BUY", 2.0)]), 20.02)  # đúng ngưỡng
        self.assertEqual(self.fit([T(0.0, "BUY", 7.5)]), 20.02)  # không nhân theo |score|

    def test_phat_lenh_nguoc_chieu_manh(self):
        self.assertEqual(self.fit([T(2.0, "BUY", -3.0)]), 21.96)  # 22 − 0.2% × 20
        self.assertEqual(self.fit([T(0.0, "SELL", 2.0)]), 19.96)

    def test_khong_phat_nhieu_va_vung_giua(self):
        for score in (0.0, 0.1, -0.49, 0.5, 1.0, -1.99):
            for action in ("BUY", "SELL"):
                with self.subTest(score=score, action=action):
                    self.assertEqual(self.fit([T(0.0, action, score)]), 20.0)

    def test_hoa_von_khong_vi_pham_bang_dung_ngoai(self):
        trades = [T(1.0, "BUY", 0.1), T(-1.0, "SELL", 1.0), T(0.0, "HOLD", -3.0), T(0.0)]
        self.assertEqual(self.fit(trades), self.fit([]))

    def test_hoa_von_van_thap_hon_von_goc_khi(self):
        self.assertEqual(self.fit([T(0.0, "BUY", -3.0)]), 19.96)  # lệnh ngược chiều mạnh
        self.assertEqual(self.fit([T(-10.0), T(10.0)]), 19.4)  # DD 50% → −0.1 × 20 × 0.3
        self.assertEqual(self.fit([T(1.0, "BUY", 3.0), T(-1.0, "BUY", -3.0)]), 19.98)  # thưởng 0.02 < phạt 0.04

    def test_phat_drawdown_chi_phan_vuot_nguong(self):
        self.assertEqual(self.fit([T(-4.0), T(4.0)]), 20.0)  # DD đúng 20%: chưa phạt
        self.assertEqual(self.fit([T(20.0), T(-16.0)]), 23.6)  # đỉnh 40 → 24: DD 40% → −0.4
        self.assertEqual(self.fit([T(-30.0), T(40.0)]), 28.4)  # vốn âm giữa chừng: DD kẹp 100% → −1.6

    def test_max_drawdown_tu_duong_von(self):
        self.assertEqual(max_drawdown_from_trades([], 20), 0.0)
        self.assertEqual(max_drawdown_from_trades([5, -10], 20), 0.4)
        self.assertEqual(max_drawdown_from_trades([-10, 5, -10], 20), 0.75)
        self.assertEqual(max_drawdown_from_trades([-30], 20), 1.0)

    def test_con_von_nhung_duoi_san(self):
        self.assertEqual(self.fit([T(-19.999)]), 0.01)

    def test_chay_tai_khoan_phan_hoa_theo_muc_lo(self):
        blown = [self.fit([T(pnl)]) for pnl in (-20.0, -25.0, -40.0, -1e6)]
        self.assertEqual(blown[:3], [0.005, 0.004, 0.0025])  # 0.01 × 0.5 / (1 + |vốn cuối| / 20)
        self.assertAlmostEqual(blown[3], 1e-7)
        self.assertTrue(all(a > b > 0 for a, b in zip(blown, blown[1:])))
        self.assertLess(blown[0], self.fit([T(-19.999)]))  # luôn dưới agent còn vốn
        self.assertEqual(self.fit([T(-20.0, "BUY", 3.0)]), 0.005)  # không cộng thưởng khi đã cháy

    def test_tham_so_tuy_chinh(self):
        trades = [T(2.0, "BUY", 1.0), T(-4.0, "SELL", 1.0)]
        # 18 + 0.01×20 − 0.05×20 − 1×20×(4/22 − 0.1) = 15.563636
        f = self.fit(trades, strong_threshold=1, aligned_bonus_rate=0.01, counter_penalty_rate=0.05,
                     drawdown_threshold=0.1, drawdown_penalty_rate=1, floor=1, blown_ratio=0.2)
        self.assertEqual(f, 15.563636)
        self.assertAlmostEqual(self.fit([T(-30.0)], floor=1, blown_ratio=0.2), 0.2 / 1.5)
        self.assertEqual(self.fit([T(0.0, "BUY", 0.0)], strong_threshold=0), 20.0)
        self.assertEqual(self.fit([T(10.0, "BUY", 3.0)], initial_capital=100), 110.1)

    def test_action_hoa_thuong_dong_nghia_va_doi_tuong(self):
        trades = [T(0, "long", 3), T(0, "Short", -3), T(0, "strong_sell", 3), T(None, "buy", None)]
        self.assertEqual(self.fit(trades), 20.0)  # +0.02 +0.02 −0.04
        self.assertEqual(self.fit([Lenh(2.0, "STRONG_BUY", 2.5), Lenh()]), 22.02)

    def test_gia_tri_khong_hop_le(self):
        for capital in (0, -1, float("nan"), float("inf"), "20", True, None):
            with self.subTest(capital=capital), self.assertRaises(ValueError):
                self.fit([], initial_capital=capital)
        for trade in (T(float("nan")), T("1"), T(1, "BUY", float("inf")), T(1, "BUY", True)):
            with self.subTest(trade=trade), self.assertRaises(ValueError):
                self.fit([trade])
        for name in ("strong_threshold", "aligned_bonus_rate", "counter_penalty_rate", "drawdown_threshold",
                     "drawdown_penalty_rate", "floor", "blown_ratio"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.fit([], **{name: -0.1})
        with self.assertRaises(ValueError):
            self.fit([], blown_ratio=1.0)

    def test_chi_phi_ngoai_lenh(self):
        self.assertEqual(self.fit([], other_costs=0.0), 20.0)
        self.assertEqual(self.fit([], other_costs=2.0), 18.0)  # đứng ngoài vẫn trả chi phí vận hành
        self.assertEqual(self.fit([T(1.0, "BUY", 3.0)], other_costs=0.5), 20.52)
        self.assertEqual(self.fit([], other_costs=25.0), 0.004)  # chi phí làm cháy tài khoản
        for bad in (-1, float("nan"), float("inf"), "1", True, None):
            with self.subTest(other_costs=bad), self.assertRaises(ValueError):
                self.fit([], other_costs=bad)

    def test_khop_ban_goc_khi_con_von_va_dd_nho(self):
        def goc(history, initial_capital=20.0):
            if not history:
                return initial_capital
            value = initial_capital + sum(t.get("pnl", 0.0) for t in history)
            if value <= 0:
                return 0.01
            adj = 0.0
            for t in history:
                a, s = t.get("action"), t.get("confluence_score", 0.0)
                if (a == "BUY" and s >= 2.0) or (a == "SELL" and s <= -2.0):
                    adj += 0.001 * initial_capital
                elif (a == "BUY" and s <= -2.0) or (a == "SELL" and s >= 2.0):
                    adj -= 0.002 * initial_capital
            return max(0.01, value + adj)

        rng = __import__("random").Random(5)
        for _ in range(200):
            trades = [T(round(rng.uniform(-0.5, 0.5), 2), rng.choice(["BUY", "SELL", "HOLD"]),
                        round(rng.uniform(-7.5, 7.5), 1)) for _ in range(rng.randint(1, 6))]
            self.assertAlmostEqual(self.fit(trades), goc(trades), places=6)


if __name__ == "__main__":
    unittest.main()
