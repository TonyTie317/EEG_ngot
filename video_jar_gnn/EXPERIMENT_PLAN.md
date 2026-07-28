# Kế hoạch thí nghiệm video JAR

## Mục tiêu và đơn vị thống kê

Hai bài toán được đánh giá riêng:

- `binary`: `Vừa phải (JAR=3)` và `Khác (JAR=1,2,4,5)`.
- `jar3`: `Không đủ (1,2)`, `Vừa phải (3)`, `Quá nhiều (4,5)`.

Đơn vị nhãn độc lập là một cặp `subject_id × ma_mau`, không phải một
repeat. Năm repeat của cùng người và mẫu có cùng JAR, vì vậy pipeline advanced
gộp năm repeat trước khi tính loss. Sau khi bỏ nước 605, dữ liệu có 28 người ×
5 mẫu = 140 condition độc lập.

Nhãn phải được nối bằng `subject_id + ma_mau`; không thể chỉ dùng `ma_mau`
giữa mọi người vì cùng một mẫu có thể nhận JAR khác nhau. `repeat` chỉ dùng để
gắn đúng đoạn video và gom năm lần thử của condition.

## Thứ tự thí nghiệm

| Giai đoạn | Input | Model | Câu hỏi cần trả lời |
|---|---|---|---|
| A0 | expression_v2, không JAR | Repeat/window audit | Năm lần lặp có tín hiệu condition ổn định không? |
| A | expression_v2 | Logistic + chọn window/feature nested | Proxy mới có dự đoán được JAR ở người chưa thấy không? |
| B | Legacy/video đã hiệu chỉnh | Raw-, trial-, neutral-, water-delta | Cách bỏ nền nào giảm khác biệt giữa người? |
| C | expression_v2 | TCN repeat-set nhỏ | Chuỗi thời gian có hơn feature thủ công không? |
| D | expression_v2 | ST-GCN/GRU nhỏ | Graph hoặc recurrent encoder có thêm giá trị không? |

Luôn chạy A0 rồi A trước. Chỉ kết luận video hữu ích khi face-only vượt chance ổn
định trên subject chưa thấy và khoảng tin cậy không quá rộng. `ma_mau` không
phải model input; nó chỉ là khóa nối/gom condition và chọn nước 605.

## Xử lý thời gian video

CSV frame label có một dòng cho mỗi frame video 60 fps. Pipeline không dùng
tần số EEG 100 Hz để cắt video. Đoạn active được chọn theo `t_lsl` trong 10
giây thực.

Cache legacy resample mỗi trial về 96 time point để train nhanh. Cache
`expression_v2` mặc định giữ 600 mốc/10 giây, chuẩn hóa head pose, tách
trái/phải thành 20 proxy và lưu hai mask observed/imputed. Proxy được tính
trên frame nguồn duy nhất theo `sampled_lsl`, nội suy sang `target_lsl`, rồi
mới ước lượng motion bằng local quadratic 300 ms.

Lệnh chính:

```bash
python -m video_jar_gnn extract --representation expression_v2
python -m video_jar_gnn audit-expression
python -m video_jar_gnn train-expression --task binary
python -m video_jar_gnn train-expression --task jar3
```

Response window chỉ được chọn trong inner subject-CV của `train-expression`;
không chọn `4:6` hay cửa sổ khác bằng metric outer-test. Sensitivity analysis
legacy với ba mức vẫn có thể lưu ra ba thư mục riêng:

```bash
# 9.6 time point/s, cache hiện tại
python -m video_jar_gnn extract --num-frames 96

# 30 time point/s
python -m video_jar_gnn extract \
  --num-frames 300 \
  --output-dir output/video_jar_gnn/graphs_300 \
  --output-manifest output/video_jar_gnn/graph_manifest_300.csv

# Giữ đủ 60 time point/s trong 10 giây
python -m video_jar_gnn extract \
  --num-frames 600 \
  --output-dir output/video_jar_gnn/graphs_600 \
  --output-manifest output/video_jar_gnn/graph_manifest_600.csv
```

Không ghi 300/600 frame vào thư mục graph 96 frame, vì cache validator sẽ báo
khác cấu hình. Với 600 time point, nên bắt đầu bằng TCN và
`--batch-size 2`; ST-GCN giữ activation theo cả time × node nên tốn VRAM hơn.

## Baseline và ablation bắt buộc

Lệnh sau chạy baseline video-only có
`VarianceThreshold → StandardScaler → SelectKBest → LogisticRegression`.
Mọi phép fit và chọn hyperparameter nằm trong inner subject folds:

```bash
python -m video_jar_gnn train-video-features --task jar3
python -m video_jar_gnn train-video-features --task binary
```

Ba representation chính là `raw`, `trial_delta`, `water_delta`. Lệnh
`train-classical` cũ còn tồn tại để tái lập benchmark chẩn đoán code/fusion,
nhưng không phải đường huấn luyện advanced.

Sau khi `train-expression` hoàn tất đủ outer folds, chạy deep video-only
compact trên representation mới:

```bash
python -m video_jar_gnn train-advanced \
  --representation expression_v2 \
  --task jar3 \
  --model tcn \
  --device cuda

python -m video_jar_gnn train-advanced \
  --representation expression_v2 \
  --task binary \
  --model tcn \
  --device cuda
```

`expression_v2` tự resolve `preprocess=raw` vì cache đã có delta/robust-z theo
thời gian thật, đồng thời tắt rotation/relational transform dành riêng cho
legacy. `jar3` mặc định dùng ordinal loss. Với từng task, so sánh `stgcn`,
`tcn`, `gru`; không chọn kiến trúc bằng outer-test metric. Nếu cần chọn kiến
trúc chính thức, cố định lựa chọn từ thí nghiệm thăm dò hoặc thêm một tầng
model-selection bên trong.

## Hai kiểu baseline khuôn mặt

### Neutral ngay trước trial

Mỗi trial có một đoạn `ma_mau=0` liên tục ngay trước nó. Tạo cache riêng với
đúng 60 frame cho một giây video:

```bash
python -m video_jar_gnn extract \
  --manifest output/video_jar_gnn/manifest.csv \
  --output-dir output/video_jar_gnn/graphs_neutral \
  --output-manifest output/video_jar_gnn/graph_manifest_neutral.csv \
  --pre-context-seconds 1 \
  --baseline-frames 60

python -m video_jar_gnn train-advanced \
  --manifest output/video_jar_gnn/graph_manifest_neutral.csv \
  --task jar3 \
  --preprocess neutral_delta_motion
```

### Nước 605 của cùng người

Nước được dùng như một phép calibration không nhãn:

```text
sweet_graph - mean(water_graphs của cùng subject)
```

JAR của nước không đi vào loss. Tuy nhiên đây là chế độ triển khai yêu cầu có
video uống nước của cả người mới:

```bash
python -m video_jar_gnn train-advanced \
  --task jar3 \
  --preprocess water_delta \
  --no-relational-features
```

Không dùng `absolute_water_delta` làm cấu hình chính: audit hiện tại cho thấy
nhánh absolute giữ lại hình dạng khuôn mặt mang tính nhận dạng người và làm
giảm BAcc so với delta-only.

## Capacity và chọn epoch

Advanced trainer không có code embedding, code prior hoặc fusion. Mặc định
12 hidden channels cho khoảng 7 nghìn tham số, phù hợp hơn 110–115 condition
train ở mỗi outer fold so với cấu hình 35 nghìn tham số cũ.

Epoch refit được chọn từ validation curve trung bình giữa các inner folds,
không lấy trung vị của các epoch argmax riêng lẻ. Không tăng epoch chỉ vì
train accuracy còn tăng; nếu validation giảm thì đó là quá khớp.

## Quy tắc đánh giá và báo cáo

- Outer CV tách hoàn toàn theo người; mặc định 5 fold.
- Hyperparameter/epoch chỉ được chọn bằng inner subject folds.
- Metric chính: balanced accuracy và macro-F1 ở condition-level.
- Với JAR3 báo thêm MAE và quadratic weighted kappa.
- Báo confusion matrix và recall từng lớp.
- Dùng bootstrap theo whole subject cho khoảng tin cậy và chênh lệch so với
  majority/chance cân bằng.
- Không chọn cấu hình tốt nhất bằng cách nhìn cả bảy outer-CV kết quả rồi báo
  chính kết quả đó như một ước lượng không thiên lệch. Bảng ablation dùng để
  hiểu tín hiệu; xác nhận model cuối bằng seed/split hoặc cohort độc lập.

Kết quả cần đọc trong:

- `runs_video_features/<task>/<run-signature>/feature_metrics.csv`
- `runs_video_features/<task>/<run-signature>/summary.json`
- `runs_advanced/<task>/<configuration>/<run-signature>/summary.json`
- `predictions_condition.csv` để audit từng người/mẫu.
