# Autonomous Adaptive AI Agent

Kho mã này chứa dự án [`autonomous_agent_project/`](autonomous_agent_project/README.md): mô phỏng một hệ
sinh thái agent kinh tế tự trị — mỗi agent khởi đầu với $20, tự giao dịch (mô phỏng), tự học từ thất
bại qua bộ nhớ chung `memory_bank.json`, tự hủy khi vốn ≤ $0.5 và tự nhân bản khi tài sản ≥ $40 và
điểm Fitness đạt ngưỡng. Gen của thế hệ sau được tạo bằng Thuật toán Di truyền (lai ghép tinh hoa, đột
biến thích ứng, kiểm soát đa dạng quần thể).

Chỉ cần Python 3.10+ (thư viện chuẩn), không cần API key.

```bash
cd autonomous_agent_project
python main.py --tick 0 --seed 5 --reset-memory   # chạy mô phỏng
python -m unittest discover -s tests -t . -v      # chạy test
```

Xem hướng dẫn chi tiết, tham số dòng lệnh và cơ chế hoạt động trong
[`autonomous_agent_project/README.md`](autonomous_agent_project/README.md).
