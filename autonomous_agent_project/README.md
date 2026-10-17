# autonomous_agent_project — Hệ sinh thái Autonomous Adaptive AI Agent

Mô phỏng một quần thể "agent kinh tế" tự trị: mỗi agent khởi đầu với **đúng $20.0**, tự giao dịch
trên một thị trường giả lập để kiếm tiền, **tự học từ thất bại** (Reflexion), **tự hủy** khi cạn vốn
và **tự nhân bản** khi vừa đủ giàu vừa đủ "khoẻ" (điểm Fitness). Gen của thế hệ sau được tạo bằng
**Thuật toán Di truyền**: lai ghép từ nhóm tinh hoa, đột biến thích ứng, và kiểm soát đa dạng quần thể. Mọi agent dùng chung một bộ nhớ `data/memory_bank.json`, nên bài học
xương máu của một agent (kể cả cái chết của nó) trở thành "luật sinh tồn" cho cả hệ sinh thái.

* Phần mô phỏng chỉ dùng **thư viện chuẩn Python 3.10+**, không cần API key. Riêng module lấy giá thật từ Binance cần `requests`.
* Giao dịch **hoàn toàn mô phỏng**; lớp thực thi được tách qua interface để có thể thay bằng sàn thật.

## Cấu trúc thư mục

```
autonomous_agent_project/
├── main.py              # Khởi tạo & điều phối vòng lặp mô phỏng thời gian thực (CLI)
├── core/
│   ├── agent.py         # Vòng đời agent, Fitness, Tự hủy, Tự nhân bản, GeneticController
│   ├── strategy.py      # Thị trường, chiến lược, interface thực thi, bộ lọc quy tắc, toán tử gen
│   ├── memory.py        # Reflexion: Post-Mortem, đúc kết quy tắc, ghi JSON atomic
│   ├── price_feed.py    # Lấy giá/nến thật từ API công khai Binance (cần requests)
│   ├── advanced_strategy.py # MultiDimensionalAnalyzer: SMC + Elliott (heuristic) + Hidden Base
│   ├── checkpoint.py    # Lưu/khôi phục checkpoint quần thể (atomic, có schema_version)
│   └── dashboard.py     # Dashboard terminal bằng rich (tuỳ chọn): quần thể, sự kiện, di truyền
├── advanced_demo.py     # Demo phân tích đa khung thời gian (nến giả lập hoặc --live)
├── evolution_demo.py    # Demo tiến hoá: chạy lại là tiếp tục từ thế hệ đã lưu
├── dashboard_demo.py    # Demo dashboard với dữ liệu giả
├── data/
│   └── memory_bank.json # Bộ nhớ chung (tự tạo khi chạy, không commit)
├── checkpoints/         # Checkpoint quần thể (tự tạo khi bật checkpoint, không commit)
├── tests/               # Unit test (unittest, tương thích pytest)
└── requirements.txt     # requests (price_feed), rich (dashboard, tuỳ chọn); mô phỏng dùng thư viện chuẩn
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
| `--tick` | 0.05 (2.0 khi `--live-price`) | Số giây nghỉ giữa hai tick (`0` = nhanh nhất) |
| `--live-price` | tắt | Dùng giá thật từ Binance thay cho thị trường giả lập (lệnh vẫn mô phỏng) |
| `--symbol` | BTCUSDT | Cặp giao dịch khi bật `--live-price` |
| `--seed` | ngẫu nhiên | Hạt giống để tái lập kết quả |
| `--agents` | 5 | Số agent thế hệ 0 |
| `--max-agents` / `--population-cap` | 30 | Trần số agent sống cùng lúc (hết chỗ thì hoãn sinh sản) |
| `--operating-cost` | 0.02 | Chi phí vận hành mỗi tick ($) — agent không làm gì cũng sẽ "chết đói" |
| `--mutation-rate` | 0.1 | Biên độ đột biến cơ sở (tự co giãn 0.03–0.35 theo áp lực thất bại) |
| `--fitness-threshold` | 1.5 | Điểm Fitness tối thiểu để được nhân bản |
| `--elite-size` | 3 | Số agent Fitness cao nhất tham gia lai ghép |
| `--crossover-prob` | 0.7 | Xác suất lai ghép (ngược lại: tự đột biến từ gen cha) |
| `--min-diversity` | 0.05 | Khoảng cách gen tối thiểu giữa hai agent (chống đồng huyết thống) |
| `--diversity-every` | 25 | Rà soát trùng lặp gen toàn quần thể sau mỗi N tick (`0` = tắt) |
| `--summary-every` | 25 | In dòng tóm tắt sau mỗi N tick |
| `--memory` | `data/memory_bank.json` | Đường dẫn tệp bộ nhớ chung |
| `--reset-memory` | tắt | Xoá sạch bộ nhớ trước khi chạy |
| `--checkpoint-every` | 0 (tắt) | Lưu checkpoint quần thể sau mỗi N tick, cộng một lần khi kết thúc/Ctrl+C |
| `--checkpoint-file` | `checkpoints/ecosystem_checkpoint.json` | Tệp checkpoint (tách biệt với bộ nhớ chung) |
| `--resume` | tắt | Khôi phục quần thể từ checkpoint rồi chạy tiếp `--steps` tick |
| `--dashboard` | tắt | Dashboard trực quan trên terminal (cần `rich`; không có thì dùng log thường) |
| `--live-dashboard` | tắt | Viết tắt: `--live-price --dashboard --steps 30 --tick 2` (cờ ghi rõ được ưu tiên) |
| `--no-fsync` | tắt | Bỏ `fsync` khi ghi để chạy nhanh hơn (vẫn atomic nhờ `os.replace`) |
| `-v` / `-q` | | Chi tiết hơn / chỉ cảnh báo |

### Đoạn output mẫu

`python main.py --tick 0 --seed 3 --reset-memory --summary-every 100` (rút gọn):

```
🧬 Tự nhân bản → A-006 (thế hệ 1), Fitness cha 2.32. Cha còn $20.23, con nhận $20.00. Gen: lai ghép A-005×A-002×A-003, biên độ đột biến 0.27 (áp lực thất bại 0.33), khoảng cách gen gần nhất 0.110. ...
📊 Tick  100 | giá    70.95 | chế độ Suy thoái   | sống  6, chết  0 | tổng vốn $ 139.74 | thế hệ cao nhất 1 | quy tắc 9 (8 hiệu lực) | Fitness tốt nhất 2.18 | đa dạng gen 0.296 | biên độ đột biến 0.24
⛔ Đủ $41.40 nhưng Fitness 1.08 < ngưỡng 1.50 (lợi nhuận ròng +107%, Max DD 42%, ổn định 0.42 (thắng 60%, σ lệnh 4.4%)) — chưa được nhân bản
🧬 Tự nhân bản → A-007 (thế hệ 1), Fitness cha 1.53. ... Gen: lai ghép A-005×A-003, biên độ đột biến 0.08 (áp lực thất bại 0.14), khoảng cách gen gần nhất 0.076. ...
💀 TERMINATED — Vốn cạn còn $0.48 (≤ $0.50) tại tick 375. 114/164 lệnh thua, ..., Max DD 98%. Lệnh chí mạng: BÁN (SHORT) đòn bẩy 3.3x, ...
📊 Tick  400 | giá    57.09 | chế độ Suy thoái   | sống  7, chết  1 | tổng vốn $ 177.55 | thế hệ cao nhất 1 | quy tắc 27 (25 hiệu lực) | Fitness tốt nhất 2.68 | đa dạng gen 0.187 | biên độ đột biến 0.10

 BÁO CÁO CUỐI CÙNG sau 500 tick
 ID     Thế hệ Cha    Trạng thái        Vốn  Fitness  Max DD  Lệnh Thắng  Thua Bị chặn Điều chỉnh  Con
 A-004       0 -      TERMINATED  $    0.48    -0.24     98%   164    50   114     171         35    0
 A-005       0 -      ACTIVE      $   34.71     5.03     24%   366   219   147      57         48    2
 A-006       1 A-005  ACTIVE      $   41.31     1.34     37%   284   165   119      50         39    0
 A-008       1 A-002  ACTIVE      $   35.37     8.18      5%    92    58    34      11          0    0
 ...
 Sống: 12 | Đã chết: 1 | Tổng vốn còn lại: $339.67 | Đa dạng gen: 0.243
```

Để ý A-006: có $41.31 nhưng Fitness chỉ 1.34 (Max DD 37%) nên vẫn chưa được nhân bản.

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
4. **Tự nhân bản**: nếu tổng tài sản **≥ $40.0** *và* **Fitness ≥ ngưỡng** (mặc định 1.5) → trích
   **$20** sinh agent con. Con thừa hưởng toàn bộ `memory_bank` (cùng tệp chung) và bộ gen do
   `GeneticController` tạo ra (xem mục *Thuật toán Di truyền*). Thế hệ, ID cha, nguồn gen, biên độ đột
   biến và Fitness của cha được ghi lại trong `pha_he`.
5. **Đọc bộ nhớ (bắt buộc)** từ đĩa → chiến lược đề xuất lệnh → **bộ lọc quy tắc** chặn/điều chỉnh →
   mở vị thế qua `ExecutionEngine`. `TradingStrategy.decide()` ném `MemoryNotConsultedError` nếu không
   nhận được ảnh chụp bộ nhớ đọc đúng ở tick hiện tại, nên không thể "quên" bước tra cứu vết xe đổ.

### Thuật toán Di truyền (`core/agent.py` + `core/strategy.py`)

**Hàm thích nghi (Fitness)** — `Agent.fitness()` trả về `FitnessReport`:

```
Fitness = (lợi nhuận ròng / max(Max Drawdown, 5%)) × hệ số ổn định
lợi nhuận ròng = (vốn hiện tại + vốn đã trao cho con − vốn khởi điểm) / vốn khởi điểm
hệ số ổn định  = tỷ lệ thắng / (1 + 10 × độ lệch chuẩn lợi suất 50 lệnh gần nhất)
```

Max Drawdown đo trên đường tài sản có cộng lại vốn đã trao cho con, nên việc sinh con không bị tính là
sụt giảm. Agent giàu nhưng sụt sâu hoặc lệnh thất thường sẽ bị cổng Fitness chặn (log `⛔`).

**Lai ghép (Crossover)** — với xác suất `--crossover-prob`, gen con được lai ghép đồng nhất từ nhóm
tinh hoa: tối đa `--elite-size` agent đang sống có Fitness cao nhất và đạt ngưỡng, luôn gồm agent đang
sinh sản. Mỗi gen được lấy từ một thành viên tinh hoa, chọn ngẫu nhiên theo trọng số Fitness
(`StrategyParams.crossover`). Nếu nhóm tinh hoa chỉ có một agent, gen con được tự đột biến từ gen cha.

**Đột biến thích ứng (Adaptive Mutation)** — áp lực thất bại = số lệnh lỗ ghi trong `memory_bank`
(bộ đếm `that_bai_theo_tick`) trong 30 tick gần nhất, chia cho số agent đang sống và độ dài cửa sổ.
`adaptive_mutation_rate()` nội suy tuyến tính: áp lực 0 → 0.03 (tinh chỉnh chiến lược đang tốt); áp lực
0.2 → biên độ cơ sở 0.1; áp lực ≥ 0.4 → 0.35 (mở rộng tìm kiếm chiến lược mới).

**Kiểm soát đa dạng (Diversity Control)** — bộ gen được chuẩn hoá về [0, 1], khoảng cách giữa hai
agent là khoảng cách Euclid chuẩn hoá (`StrategyParams.distance`).

* Khi sinh con, `diversify()` đảm bảo gen con cách mọi agent đang sống ít nhất `--min-diversity`. Nếu
  chưa đủ xa, gen con được đột biến lại với biên độ tăng dần (×1.5 mỗi lần, tối đa 6 lần). Nếu vẫn trùng,
  một nửa số gen được thay bằng **gen nhập cư** ngẫu nhiên.
* Cứ mỗi `--diversity-every` tick, `GeneticController.enforce_diversity()` rà soát cả quần thể. Trong
  mỗi cặp "đồng huyết thống", agent có Fitness thấp hơn phải **phân hoá ngách** (đổi gen, log `🧪`).
* Dòng tóm tắt hiển thị *đa dạng gen*, tức khoảng cách trung bình giữa mọi cặp agent.

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

## Lấy giá thật từ Binance (`core/price_feed.py`)

Module cần `pip install requests` và không cần API key. Có thể dùng độc lập:

```bash
pip install -r requirements.txt
python -m core.price_feed            # Giá BTCUSDT hiện tại: 82,030.63
python -m core.price_feed ETHUSDT
```

```python
from core.price_feed import get_latest_price
price = get_latest_price("BTCUSDT")  # float, hoặc None nếu lỗi
```

Ở các vùng bị Binance hạn chế, `api.binance.com` trả **HTTP 451**. Khi đó hàm in lý do lỗi rồi tự chuyển
sang endpoint dữ liệu công khai `data-api.binance.vision`, vốn trả cùng định dạng.

### Chạy hệ sinh thái trên giá thật (`--live-price`)

```bash
python main.py --live-price                                  # BTCUSDT, mỗi tick lấy giá mới, nghỉ 2 giây
python main.py --live-price --symbol ETHUSDT --steps 300 --tick 5
```

Mỗi tick, `LiveMarketFeed` lấy một giá mới từ Binance rồi tính động lượng và biến động giống hệt thị
trường giả lập. Nhờ vậy agent, bộ lọc quy tắc, Fitness và GA chạy nguyên vẹn. **Lệnh vẫn chỉ là mô
phỏng**, không có tiền thật nào được giao dịch.

Cơ chế dự phòng khi API lỗi:

* Không lấy được giá lúc khởi động (mất mạng, sai `--symbol`): chạy bằng thị trường **giả lập**, bắt
  đầu ở giá sát thực tế của cặp đó (`REALISTIC_FALLBACK_PRICES` trong `core/price_feed.py`): BTCUSDT
  82,000, ETHUSDT 2,400, BNBUSDT 600, SOLUSDT 180, XRPUSDT 0.6. Cặp khác bắt đầu từ 100. Ví dụ:

  ```
  [Cảnh báo] Không thể kết nối Binance API. Kích hoạt chế độ giả lập với giá khởi điểm thực tế: 82,000.00 cho BTCUSDT. Chi tiết: …
  ```

  Khi lấy được giá, chương trình in `[Thành công] Đã lấy giá thật từ Binance: 81,573.85 (BTCUSDT)`. Chế độ
  giả lập mặc định (không có `--live-price`) vẫn bắt đầu từ 100, nên kết quả theo seed không đổi.
  Mô phỏng chỉ dùng lợi suất %: lãi/lỗ, phí, trượt giá, động lượng và biến động đều tính theo tỷ lệ.
  Vì vậy cùng seed, giá khởi điểm 82,000, 0.6 hay 100 cho ra cùng một quần thể. Giá dưới 1 hiển thị 4
  chữ số thập phân (`0.6124`), dưới 0.01 thì 8 chữ số.
* Lỗi giữa chừng: tick đó dùng **giá giả lập**, là một bước ngẫu nhiên từ giá cuối theo biến động gần
  đây. Dòng tóm tắt hiển thị chế độ `Giả lập (API lỗi)`. Khi API hoạt động lại, chương trình tự quay về
  giá thật.

Lưu ý: giá thật thay đổi rất ít giữa các tick vài giây, thường nhỏ hơn phí giao dịch 0.05% mỗi chiều.
Agent vì thế chủ yếu đứng ngoài thị trường và chịu chi phí vận hành. Nên dùng `--tick` lớn hơn, ví dụ
30–60 giây, để mỗi tick có biến động đáng kể.

## Dashboard trên terminal (`core/dashboard.py`)

```bash
pip install rich                                   # tuỳ chọn, chỉ cần cho dashboard
python main.py --dashboard --tick 0.05 --seed 7    # mô phỏng thật, hiển thị bằng dashboard
python main.py --dashboard --checkpoint-every 100  # kèm checkpoint, Ctrl+C vẫn lưu
python dashboard_demo.py --steps 40 --delay 0.3    # demo độc lập với dữ liệu giả
python main.py --live-dashboard                    # giá thật Binance + dashboard, 30 bước, tick 2 giây
python main.py --live-dashboard --symbol ETHUSDT --steps 100   # cờ ghi rõ được ưu tiên
```

`--live-dashboard` là viết tắt của `--live-price --dashboard --steps 30 --tick 2 --symbol BTCUSDT`.
Gọi từ Python thì dùng `run_simulation_with_dashboard`. Hàm này chạy mô phỏng thật: quần thể, agent
tốt nhất và thế hệ đều lấy từ hệ sinh thái. Các tham số CLI khác truyền qua `kwargs`:

```python
from main import run_simulation_with_dashboard
run_simulation_with_dashboard(symbol="BTCUSDT", steps=30, tick_interval=2, seed=7, checkpoint_every=10)
```

Không lấy được giá lúc khởi động thì chuyển sang thị trường giả lập, bắt đầu ở giá sát thực tế của cặp. Lỗi giữa chừng thì tick đó dùng
giá giả lập. Thiếu `rich` hoặc không phải terminal thì dùng log thường.

Dashboard gồm bốn phần:

* **Thị trường & agent:** bước hiện tại / tổng, giá, biến động, chế độ thị trường, số agent sống/chết,
  tổng vốn, độ đa dạng gen, biên độ đột biến, số quy tắc hiệu lực, agent tốt nhất (`N/A` khi chưa có).
* **Quần thể đang sống:** ID, thế hệ, cha, vốn, Fitness, trạng thái, xếp theo Fitness (tối đa 12 dòng).
* **Sự kiện gần nhất:** agent ra đời, agent tự hủy (kèm nguyên nhân), quy tắc **CẤM** mới.
* **Tiêu đề và chân trang:** cặp giao dịch, nguồn giá (thật/giả lập), tệp checkpoint, lần lưu cuối.

Khi dashboard chạy, log từng agent bị ẩn (sinh/chết đã có trong bảng sự kiện). Cảnh báo khác vẫn in
phía trên khung. Báo cáo cuối in như bình thường sau khi dashboard đóng. `Ctrl+C` dừng êm sau tick
hiện tại và lưu checkpoint nếu đã bật.

Nếu chưa cài `rich`, hoặc terminal không hỗ trợ (không phải TTY, `TERM=dumb`, chuyển hướng ra tệp),
chương trình in cảnh báo rồi chạy với log thường. Dashboard chỉ hiển thị, không ảnh hưởng kết quả mô
phỏng: cùng seed thì có và không có `--dashboard` cho ra cùng một quần thể. Biến môi trường
`NO_COLOR=1` tắt màu.

## Checkpoint quần thể (`core/checkpoint.py`)

Mặc định không có checkpoint, hành vi như cũ. Bật lên bằng cờ:

```bash
python main.py --steps 300 --tick 0 --seed 7 --checkpoint-every 50   # lưu tại tick 50, 100, … và khi Ctrl+C
python main.py --steps 200 --tick 0 --resume                         # chạy tiếp tick 301 → 500
```

Checkpoint chứa toàn bộ quần thể: agent đang sống và đã chết, vốn, trạng thái, thế hệ, ID cha, tham số
gen, thống kê, vị thế đang mở, lợi suất các lệnh gần nhất. Ngoài ra có tick hiện tại, bộ đếm ID, chuỗi
giá của thị trường, thống kê Fitness của quần thể, bản sao agent tốt nhất từng ghi nhận, và trạng
thái của mọi bộ sinh số ngẫu nhiên. Nhờ vậy, chạy 213 tick rồi `--resume` thêm 287 tick cho báo cáo
cuối **giống hệt** chạy liền 500 tick cùng seed. Khi resume, `--seed` bị bỏ qua.

* Ghi atomic: tệp tạm → `fsync` → `os.replace`. Ctrl+C hay mất điện giữa lúc ghi không làm hỏng
  checkpoint cũ.
* Có trường `schema_version`. Checkpoint của phiên bản mới hơn bị từ chối và giữ nguyên tệp.
* Tệp hỏng hoặc sai cấu trúc được đổi tên thành `*.corrupt.json`, rồi chương trình khởi tạo quần thể
  mới. `--resume` khi chưa có tệp cũng khởi tạo mới.
* `memory_bank.json` vẫn là bộ nhớ riêng. `--resume` cần dùng cùng `--memory` với lần trước, và
  không được dùng cùng `--reset-memory`.
* Với `--live-price`, giá thật không "tua lại" được: chỉ tick và quần thể được khôi phục.

Demo tối giản (bản đã sửa lỗi của `run_evolution_with_checkpoint`):

```bash
python evolution_demo.py           # thế hệ 1–5
python evolution_demo.py           # khôi phục, chạy tiếp thế hệ 6–10
python evolution_demo.py --reset   # bắt đầu lại
```

```
[Checkpoint] Khôi phục thành công thế hệ 5 từ …/autonomous_agent_project/checkpoints/latest_checkpoint.json
-> Tiếp tục tiến hóa từ thế hệ 6

--- Đang chạy Thế hệ (Generation) 6 ---
🌟 Gen tốt nhất hiện tại (ID: 0, ghi nhận ở thế hệ 6): Fitness = 7.0818, gen = [0.6099, 0.9816]
[Checkpoint] Đã lưu thành công trạng thái thế hệ 6 vào …/checkpoints/latest_checkpoint.json
```

## Phân tích đa khung thời gian (`core/advanced_strategy.py`)

`MultiDimensionalAnalyzer` kết hợp ba lớp phân tích trên nến OHLC (định dạng kline Binance):

| Lớp | Khung | Cách làm |
| --- | --- | --- |
| SMC | HTF (mặc định 4h) | Swing fractal → BOS (close vượt đỉnh/thủng đáy swing gần nhất), sau đó HH/HL hoặc LH/LL |
| Price Action / Hidden Base | LTF (mặc định 5m) | Phá vỡ nền giá chặt, sau đó nến nhấn chìm, sau đó thân nến ≥ 30% biên độ; doji → HOLD |
| Elliott | HTF | **Heuristic động lượng**, không đếm sóng thật: động lượng ngắn mạnh hơn dài → `IMPULSE_WAVE_3`, cùng chiều → `IMPULSE_WAVE_5`, ngược chiều → `CORRECTIVE` |

Điểm hợp lưu: +2 khi HTF BULLISH và LTF BUY, −2 khi BEARISH và SELL. Điểm này nhân 1.5 khi sóng 3
**cùng chiều**. Kết quả ≥ 2 → LONG, ≤ −2 → SHORT, còn lại → WAIT.

```bash
python advanced_demo.py                       # nến giả lập xu hướng tăng (tái lập nhờ --seed)
python advanced_demo.py --scenario giam       # kịch bản giảm; di_ngang → thường WAIT
python advanced_demo.py --dummy-goc           # tái hiện dữ liệu dummy gốc (luôn WAIT)
python advanced_demo.py --live --symbol ETHUSDT --htf 1h --ltf 5m   # nến thật, lỗi → nến giả lập
```

Output mẫu với `--live`:

```
📥 Nguồn nến: Binance BTCUSDT thật (4h/5m) — HTF 100 nến, LTF 50 nến, close cuối 81,502.01
📊 Kết quả phân tích đa khung thời gian:
   - Xu hướng HTF (SMC): BEARISH
   - Tín hiệu LTF (Price Action/Hidden Base): BUY
   - Trạng thái sóng Elliott (heuristic động lượng): IMPULSE_WAVE_3 (BEARISH)
   - Điểm hợp lưu (Confluence Score): 0.0
🤖 Quyết định của Agent: KIÊN NHẪN CHỜ ĐỢI (WAIT FOR SETUP).
```

Nến thật lấy qua `core.price_feed.fetch_klines` (`/api/v3/klines`). Khi bị chặn HTTP 451, hàm thử tiếp
`data-api.binance.vision`. Bộ phân tích này **chưa nối vào vòng mô phỏng `main.py`**, vì thị trường mô
phỏng chỉ sinh giá đóng cửa và không có nến OHLC.

## Chạy test

```bash
cd autonomous_agent_project
python -m unittest discover -s tests -t . -v
# hoặc, nếu đã cài pytest:
python -m pytest -q
```

Bộ test (136 ca, không cần mạng, không cần TTY) bao phủ: giá khởi điểm dự phòng theo cặp và tính độc lập với thang giá (`tests/test_fallback_price.py`); `run_simulation_with_dashboard` và `--live-dashboard` với giá Binance được mock (`tests/test_live_dashboard.py`); dashboard (render ra chuỗi, best_agent None, bảng quần thể, sự kiện sinh/chết/CẤM, dự phòng khi thiếu rich hoặc không phải TTY, Ctrl+C) (`tests/test_dashboard.py`); checkpoint (ghi atomic, schema, tệp hỏng, bản sao best_agent, resume cho kết quả y hệt chạy liền, Ctrl+C) (`tests/test_checkpoint.py`); phân tích đa khung SMC/Price Action/Elliott heuristic, `fetch_klines` và demo (`tests/test_advanced_strategy.py`); chế độ giá thật và cơ chế dự phòng về giá giả lập (`tests/test_live_market.py`, `tests/test_price_feed.py`); Fitness (công thức, Max DD, cổng nhân bản), lai ghép, đột biến thích ứng,
chống trùng lặp gen và phân hoá ngách (`tests/test_genetics.py`); tự hủy (đúng ngưỡng $0.5, sau lệnh lỗ, chết đói, ngừng hoạt động sau khi chết), nhân
bản (ngưỡng $40, trích $20, thế hệ/ID cha, đột biến trong biên, thừa kế bộ nhớ, trần quần thể), bộ lọc
quy tắc (CAM, giảm quy mô, giới hạn đòn bẩy, ưu tiên, quy tắc hết hiệu lực, điều kiện sai định dạng),
bắt buộc đọc bộ nhớ, và bộ nhớ Reflexion (atomic write, khôi phục tệp hỏng, đúc kết/nâng cấp/gỡ quy tắc).
