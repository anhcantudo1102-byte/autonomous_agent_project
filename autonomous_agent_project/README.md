# autonomous_agent_project — Hệ sinh thái Autonomous Adaptive AI Agent

Mô phỏng một quần thể "agent kinh tế" tự trị: mỗi agent khởi đầu với **đúng $20.0**, tự giao dịch
trên một thị trường giả lập để kiếm tiền, **tự học từ thất bại** (Reflexion), **tự hủy** khi cạn vốn
và **tự nhân bản** khi đủ giàu. Mọi agent dùng chung một bộ nhớ `data/memory_bank.json`, nên bài học
xương máu của một agent (kể cả cái chết của nó) trở thành "luật sinh tồn" cho cả hệ sinh thái.

* Chỉ dùng **thư viện chuẩn Python 3.10+** — không cần cài gói, không cần API key.
* Giao dịch **hoàn toàn mô phỏng**; lớp thực thi được tách qua interface để có thể thay bằng sàn thật.

## Cấu trúc thư mục

```
autonomous_agent_project/
├── main.py              # Khởi tạo & điều phối vòng lặp mô phỏng thời gian thực (CLI)
├── core/
│   ├── agent.py         # Vòng đời agent: theo dõi vốn, Tự hủy, Tự nhân bản
│   ├── strategy.py      # Thị trường giả lập, chiến lược, interface thực thi, bộ lọc quy tắc
│   └── memory.py        # Reflexion: Post-Mortem, đúc kết quy tắc, ghi JSON atomic
├── data/
│   └── memory_bank.json # Bộ nhớ chung (tự tạo khi chạy, không commit)
├── tests/               # Unit test (unittest, tương thích pytest)
└── requirements.txt     # Rỗng — chỉ dùng thư viện chuẩn
```

## Cách chạy nhanh

Mở Terminal, đứng ở **thư mục `autonomous_agent_project/`** rồi chạy:

```bash
cd autonomous_agent_project
python main.py
```

(Trên một số hệ Linux/macOS, lệnh là `python3` thay cho `python`.)

Mặc định chương trình chạy 500 tick với 5 agent khởi sinh, mỗi tick nghỉ 0.05 giây để bạn xem diễn
biến "thời gian thực". Nhấn `Ctrl+C` bất kỳ lúc nào để dừng êm — bộ nhớ vẫn được ghi đầy đủ và báo
cáo cuối vẫn được in ra.

### Các ví dụ thường dùng

```bash
# Chạy nhanh nhất có thể (không nghỉ giữa các tick), tái lập được nhờ seed, xoá ký ức cũ
python main.py --tick 0 --seed 5 --reset-memory

# Mô phỏng dài 2000 tick với 8 agent ban đầu, tối đa 50 agent sống cùng lúc
python main.py --steps 2000 --agents 8 --max-agents 50 --tick 0

# Xem "chậm" như phim: 1 tick mỗi 0.3 giây, in tóm tắt sau mỗi 10 tick
python main.py --steps 200 --tick 0.3 --summary-every 10

# Xem chi tiết từng Post-Mortem, lệnh bị chặn/điều chỉnh
python main.py --steps 100 --tick 0 --verbose

# Chỉ in cảnh báo (cái chết) và báo cáo cuối
python main.py --tick 0 --quiet

# Chạy tiếp với bộ nhớ đã tích luỹ từ lần trước (KHÔNG truyền --reset-memory):
# thế hệ agent mới sẽ né những "vết xe đổ" mà thế hệ trước đã ghi lại.
python main.py --tick 0 --seed 6

# Dùng một tệp bộ nhớ riêng cho thí nghiệm
python main.py --memory /tmp/thi_nghiem.json --reset-memory --tick 0
```

### Tham số dòng lệnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--steps` | 500 | Số tick mô phỏng |
| `--tick` | 0.05 | Số giây nghỉ giữa hai tick (`0` = nhanh nhất) |
| `--seed` | ngẫu nhiên | Hạt giống để tái lập kết quả |
| `--agents` | 5 | Số agent thế hệ 0 |
| `--max-agents` | 30 | Trần số agent sống cùng lúc (hết chỗ thì hoãn sinh sản) |
| `--operating-cost` | 0.02 | Chi phí vận hành mỗi tick ($) — agent không làm gì cũng sẽ "chết đói" |
| `--mutation-rate` | 0.1 | Độ lớn đột biến gen khi nhân bản |
| `--summary-every` | 25 | In dòng tóm tắt sau mỗi N tick |
| `--memory` | `data/memory_bank.json` | Đường dẫn tệp bộ nhớ chung |
| `--reset-memory` | tắt | Xoá sạch bộ nhớ trước khi chạy |
| `--no-fsync` | tắt | Bỏ `fsync` khi ghi để chạy nhanh hơn (vẫn atomic nhờ `os.replace`) |
| `-v` / `-q` | | Chi tiết hơn / chỉ cảnh báo |

### Đoạn output mẫu

`python main.py --tick 0 --seed 5 --reset-memory --summary-every 100` (rút gọn):

```
🌱 Khởi sinh A-001 với $20.00 | ngưỡng động lượng 1.00%, tỷ trọng 77%, đòn bẩy 3.7x, chịu biến động 5.6%, đánh ĐẢO CHIỀU
...
🧠 Nâng cấp quy tắc QT-0008 [CAM]: Cấm BÁN (SHORT) khi biến động từ 2.0% đến 4.0% và động lượng từ +3.0% trở lên — vết xe đổ: thua 8/12 lần, lỗ ròng $2.46.
🧠 Đúc kết quy tắc mới QT-0009 [GIOI_HAN_DON_BAY]: Không dùng đòn bẩy quá 2.2x khi biến động từ 4.0% trở lên — một lệnh BÁN (SHORT) đòn bẩy 3.7x, tỷ trọng 77%, biến động 4.07%, động lượng +2.93% đã cuốn mất 24% vốn.
📊 Tick  300 | giá   157.87 | chế độ Tăng trưởng | sống  5, chết  0 | tổng vốn $  92.66 | thế hệ cao nhất 0 | quy tắc 14 (14 hiệu lực)
🧬 Tự nhân bản → A-006 (thế hệ 1). Cha còn $20.46, con nhận $20.00. Gen đột biến: momentum_threshold: 0.01143→0.01027, position_fraction: 0.6245→0.7344, ...
💀 TERMINATED — Vốn cạn còn $0.49 (≤ $0.50) tại tick 328. 90/156 lệnh thua, chuỗi thua cuối 6 lệnh, ... Lệnh chí mạng: BÁN (SHORT) đòn bẩy 3.7x, tỷ trọng 39%, biến động 1.10%, động lượng +1.80%.
🪦 Di chúc để lại quy tắc QT-0015: Bài học từ cái chết của A-001: tuyệt đối không BÁN (SHORT) khi biến động từ 1.0% đến 2.0% và động lượng từ +1.0% đến +3.0% — lệnh như vậy đã góp phần xoá sổ một agent.
♻️  Gỡ bỏ quy tắc QT-0014 vì tín hiệu đã phục hồi
📊 Tick  500 | giá   497.10 | chế độ Tăng trưởng | sống 13, chết  1 | tổng vốn $ 346.26 | thế hệ cao nhất 3 | quy tắc 17 (16 hiệu lực)

 BÁO CÁO CUỐI CÙNG sau 500 tick
 ID     Thế hệ Cha    Trạng thái        Vốn  Lệnh Thắng  Thua Bị chặn Điều chỉnh  Con
 A-001       0 -      TERMINATED  $    0.49   156    66    90      91         82    0
 A-002       0 -      ACTIVE      $   24.00   239   162    77       5          2    3
 A-006       1 A-002  ACTIVE      $   26.71   160   109    51       1          1    2
 A-011       3 A-007  ACTIVE      $   27.97    36    25    11       0          0    0
 ...
```

Kết quả cụ thể phụ thuộc phiên bản Python và trạng thái bộ nhớ trước đó; dùng `--seed` kèm
`--reset-memory` để tái lập.

## Cơ chế hoạt động

### Vòng đời mỗi tick (`core/agent.py`)

1. **Chốt vị thế** đang mở từ tick trước → lãi được cộng dồn vào thống kê; lỗ kích hoạt **Post-Mortem**.
2. **Trả chi phí vận hành** (`--operating-cost`).
3. **Tự hủy**: nếu vốn **≤ $0.5** → trạng thái `TERMINATED`, dừng mọi hoạt động, chẩn đoán nguyên nhân
   cái chết (chuỗi thua, lệnh chí mạng hay "chết đói") và ghi vào `cai_chet` trong bộ nhớ. Lệnh chí mạng
   được đúc thành quy tắc `CAM` mức `chet_nguoi`. Bộ điều phối rút agent khỏi lịch chạy — "tiến trình"
   của nó kết thúc.
4. **Tự nhân bản**: nếu tổng tài sản **≥ $40.0** → trích **$20** sinh agent con. Con thừa hưởng toàn bộ
   `memory_bank` (cùng tệp chung) và bộ gen chiến lược của cha có **đột biến nhẹ** (nhân nhiễu Gauss
   N(1, 0.1), kẹp trong biên, 5% khả năng lật gen "đánh đảo chiều"). Thế hệ và ID cha được ghi lại
   trong `pha_he`.
5. **Đọc bộ nhớ (bắt buộc)** từ đĩa → chiến lược đề xuất lệnh → **bộ lọc quy tắc** chặn/điều chỉnh →
   mở vị thế qua `ExecutionEngine`. `TradingStrategy.decide()` ném `MemoryNotConsultedError` nếu không
   nhận được ảnh chụp bộ nhớ đọc đúng ở tick hiện tại, nên không thể "quên" bước tra cứu vết xe đổ.

### Chiến lược & thực thi (`core/strategy.py`)

* `MarketSimulator`: giá chạy theo bước ngẫu nhiên với 4 chế độ ẩn (Tăng trưởng, Suy thoái, Đi ngang,
  Hỗn loạn) và các cú sập/bật giá bất ngờ. Agent chỉ thấy `momentum`, `volatility`, `ret_1`, `price`.
* `StrategyParams` (bộ gen): `momentum_threshold`, `position_fraction`, `leverage`, `vol_tolerance`,
  `contrarian`.
* `ExecutionEngine` (interface): `open_position()` / `close_position()`. Bản mặc định
  `SimulatedExecutionEngine` có phí và trượt giá. Để giao dịch thật, viết lớp con gọi API sàn rồi truyền
  vào `Agent(..., engine=MyExchangeEngine())`.
* `RuleFilter`: đánh giá điều kiện máy đọc được của từng quy tắc trên ngữ cảnh lệnh; ưu tiên `CAM` →
  `GIOI_HAN_DON_BAY` → `GIAM_QUY_MO`.

### Bộ nhớ Reflexion (`core/memory.py`)

Mỗi lệnh lỗ được mổ xẻ: ghi ngữ cảnh (hướng lệnh, đòn bẩy, biến động, động lượng...), bài học bằng
tiếng Việt, và cập nhật thống kê theo **chữ ký tín hiệu** (hướng + nhóm biến động + nhóm động lượng).

| Điều kiện | Quy tắc sinh ra |
|---|---|
| Tín hiệu thua ≥ 4 lần, tỷ lệ thua ≥ 60%, lỗ ròng | `GIAM_QUY_MO` (chỉ dùng 50% quy mô) |
| Tín hiệu thua ≥ 6 lần, tỷ lệ thua ≥ 65%, lỗ ròng | nâng lên `CAM` |
| Một lệnh mất ≥ 20% vốn với đòn bẩy > 1.5x | `GIOI_HAN_DON_BAY` theo mức biến động |
| Agent chết | `CAM` lệnh chí mạng, mức `chet_nguoi` |
| Quy tắc `GIAM_QUY_MO` có ≥ 8 mẫu, thắng > 50%, lãi ròng | tự gỡ bỏ (`hieu_luc: false`) |

Ví dụ một quy tắc trong `memory_bank.json`:

```json
{
  "id": "QT-0008",
  "mo_ta": "Cấm BÁN (SHORT) khi biến động từ 2.0% đến 4.0% và động lượng từ +3.0% trở lên — vết xe đổ: thua 8/12 lần, lỗ ròng $2.46.",
  "dieu_kien": [
    {"truong": "direction", "toan_tu": "==", "gia_tri": "SHORT"},
    {"truong": "volatility", "toan_tu": ">=", "gia_tri": 0.02},
    {"truong": "volatility", "toan_tu": "<", "gia_tri": 0.04},
    {"truong": "momentum", "toan_tu": ">=", "gia_tri": 0.03}
  ],
  "hanh_dong": "CAM",
  "muc_do": "nghiem_trong",
  "hieu_luc": true
}
```

**Ghi an toàn**: mọi lần ghi đều đọc lại bản mới nhất, sửa, ghi ra tệp tạm cùng thư mục, `fsync`, rồi
`os.replace` (atomic). Thao tác được bảo vệ bởi khoá luồng và khoá tệp `fcntl` (trên Linux/macOS), nên
nhiều tiến trình `main.py` có thể dùng chung một tệp bộ nhớ. Tệp hỏng sẽ được sao lưu thành
`memory_bank.corrupt.json` và khởi tạo lại.

## Chạy test

```bash
cd autonomous_agent_project
python -m unittest discover -s tests -t . -v
# hoặc, nếu đã cài pytest:
python -m pytest -q
```

Bộ test bao phủ: tự hủy (đúng ngưỡng $0.5, sau lệnh lỗ, chết đói, ngừng hoạt động sau khi chết), nhân
bản (ngưỡng $40, trích $20, thế hệ/ID cha, đột biến trong biên, thừa kế bộ nhớ, trần quần thể), bộ lọc
quy tắc (CAM, giảm quy mô, giới hạn đòn bẩy, ưu tiên, quy tắc hết hiệu lực, điều kiện sai định dạng),
bắt buộc đọc bộ nhớ, và bộ nhớ Reflexion (atomic write, khôi phục tệp hỏng, đúc kết/nâng cấp/gỡ quy tắc).
