# Báo cáo tổng hợp — Phân tích EEG Vị giác

📄 **Mở file chính:** [`BAO_CAO_TONG_HOP_EEG.md`](BAO_CAO_TONG_HOP_EEG.md)

Báo cáo đầy đủ từ dữ liệu thô → tiền xử lý → epoching/căn onset → ERP → thống kê → Machine Learning → Deep Learning → kết luận, kèm **23 hình** trong `figures/`.

> 💡 Xem đẹp nhất bằng trình xem Markdown có render hình (VS Code preview, Typora, GitHub) — hình sẽ hiện inline.

## Nội dung nhanh
- **Kết quả phân loại cao nhất:** DL (ShallowConvNet) balanced_acc = **0.674**, nhỉnh hơn ML tốt nhất ổn định (GradBoost = 0.649). Chi tiết ở Mục 11.
- **T=0 / căn onset:** epoch lưu trên đĩa dùng T=0 tại trigger; realign về onset thật thực hiện lúc phân tích qua `apply_woody_realign`. Chi tiết + lưu ý kỹ thuật ở Mục 5.

## Files
| File | Nội dung |
|---|---|
| `BAO_CAO_TONG_HOP_EEG.md` | Báo cáo chính (13 mục) |
| `make_summary_fig.py` | Script tạo Hình 0 & 22 (so sánh ML vs DL) |
| `figures/` | 23 hình, đặt tên theo giai đoạn (02→08) |
