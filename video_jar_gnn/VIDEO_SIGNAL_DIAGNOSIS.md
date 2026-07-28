# Chẩn đoán tín hiệu video và kế hoạch huấn luyện

## Kết luận

Run nhị phân
`tcn_absolute_water_delta_segments_rot-relations_mean_std_none_ce`
đạt BAcc `0.452`, macro-F1 `0.450`, AUC `0.425`; bootstrap theo subject cho
BAcc 95% CI `[0.362, 0.548]`. Kết quả này chưa chứng minh mô hình tốt hơn mức
ngẫu nhiên cân bằng.

Nguyên nhân chính không phải thiếu epoch. Có quá khớp ở epoch muộn, nhưng
representation hiện tại chứa nhiều dấu vết nhận dạng người và rất ít phương
sai liên quan đến nhãn:

- epoch tốt nhất của 15 inner fold nằm trong khoảng `1–15`, trung vị `6`;
- ở cuối train, accuracy train trung vị tăng tới `0.76` trong khi validation
  BAcc trung vị giảm còn `0.451`;
- `81.9%` phương sai xác suất dự đoán được giải thích bởi subject;
- relational channels có khoảng `74.8–81.5%` phương sai theo subject, trong
  khi phương sai theo nhãn chỉ khoảng `0.45–0.74%`;
- detection ratio giữa hai lớp gần như giống nhau (`0.959` và `0.961`), nên
  lỗi phát hiện mặt không phải nguyên nhân chính.

Kiểm tra logistic có kiểm soát trên cùng outer subject folds:

| Representation | BAcc | Macro-F1 |
|---|---:|---:|
| `water_delta`, không relations | 0.533 | 0.530 |
| `water_delta` + relations | 0.508 | 0.501 |
| `absolute_water_delta`, không relations | 0.411 | 0.413 |
| `absolute_water_delta` + relations | 0.429 | 0.430 |

Nhánh absolute giữ lại hình dạng khuôn mặt tuyệt đối và làm giảm mạnh khả năng
tổng quát sang người mới. Relational feature hiện được broadcast qua 15 node,
vừa lặp thông tin vừa thiên về danh tính. Hai phần này nên được bỏ trước khi
đổi optimizer hoặc tăng số epoch.

## Kết quả chọn feature nested 5-fold

Baseline `train-video-features` đã được chạy đủ 5 outer folds trên 140
condition. `VarianceThreshold`, scaler, `SelectKBest` và logistic đều chỉ fit
bằng inner/outer-train:

| Task | Representation | BAcc | Macro-F1 | Bootstrap BAcc 95% CI |
|---|---|---:|---:|---:|
| Binary | raw | 0.459 | 0.460 | [0.371, 0.539] |
| Binary | trial-delta | 0.541 | 0.541 | [0.476, 0.604] |
| Binary | water-delta | 0.461 | 0.461 | [0.387, 0.535] |
| JAR3 | raw | 0.364 | 0.363 | [0.274, 0.451] |
| JAR3 | trial-delta | 0.371 | 0.370 | [0.297, 0.447] |
| JAR3 | water-delta | 0.407 | 0.403 | [0.331, 0.484] |

Inner CV được phép chọn `16/32/64` feature hoặc giữ toàn bộ feature không
hằng rồi dùng L2 regularization. Binary trial-delta và JAR3 water-delta nhỉnh
hơn chance, nhưng cả hai khoảng tin cậy vẫn chứa chance (`0.5` và `0.333`).
Lựa chọn số feature thay đổi mạnh giữa các fold, từ `16/32/64` tới khoảng
`1.034` feature. Điều này cho thấy chưa có một nhóm landmark-summary nhỏ, ổn
định theo nhãn; tín hiệu tốt nhất hiện tại có vẻ yếu và phân tán.

Dữ liệu **đạt về mặt kỹ thuật**: đủ 840/840 graph, mọi condition ngọt có năm
repeat, detection ratio khoảng 0.96 và tương đương giữa hai lớp. Nhưng
representation 15 node hiện tại **yếu về giá trị dự đoán cross-subject**.
Chất lượng phát hiện mặt tốt không đồng nghĩa biểu cảm chứa đủ thông tin JAR,
và bước tóm tắt/landmark có thể đã bỏ mất micro-expression cần thiết.

## Hướng thay thế đã implement: `expression_v2`

Pipeline mới không tái sử dụng graph 15 node mà extract lại video thành
`[600,20,8]` trong một cache riêng. Nó giữ trái/phải riêng, giảm head pose
theo từng trial, chỉ nội suy gap mất mặt tối đa 0,5 giây và ước lượng motion
sau khi nội suy từ `sampled_lsl` thật sang grid `target_lsl`. Các node là
proxy hình học biểu cảm, không phải FACS AU score đã được kiểm định.

Thứ tự đánh giá mới:

```bash
python -m video_jar_gnn extract --representation expression_v2
python -m video_jar_gnn audit-expression
python -m video_jar_gnn train-expression --task binary
python -m video_jar_gnn train-expression --task jar3
```

`audit-expression` không đọc JAR; nó so độ giống nhau của năm repeat trong
sáu response window theo giây thật. `train-expression` mới chọn window,
feature count và regularization hoàn toàn trong inner subject-CV. Vì vậy
không được nhìn audit/outer metric rồi thủ công chốt cửa sổ tốt nhất để báo
trên cùng dữ liệu như một kết quả không thiên lệch.

Chỉ chuyển sang:

```bash
python -m video_jar_gnn train-advanced \
  --representation expression_v2 \
  --task binary \
  --device cuda
```

khi full-data audit và baseline logistic cho thấy tín hiệu ổn định. Kết quả
smoke trên vài condition chỉ xác nhận code/shape, không trả lời chất lượng
phân loại.

## Hợp đồng video-only

Advanced trainer không còn nhận `ma_mau`, code embedding hoặc code prior làm
input. `ma_mau` chỉ tồn tại ở tầng chuẩn bị dữ liệu để:

1. nối đúng JAR với `subject_id`;
2. gom năm repeat của cùng condition;
3. nhận biết mã nước `605` khi tạo water reference không nhãn.

Checkpoint và `summary.json` ghi rõ:

```json
{
  "input_contract": "video_graph_only",
  "uses_ma_mau_as_model_feature": false
}
```

## Cấu hình mới

Mặc định advanced trainer là TCN gọn, không relational features, mean
aggregation, 12 hidden channels và khoảng 7 nghìn tham số. `auto` dùng
`trial_delta` cho binary và `water_delta` cho JAR3:

```bash
python -m video_jar_gnn train-advanced \
  --task binary \
  --device cuda
```

Lệnh tương đương viết đầy đủ:

```bash
python -m video_jar_gnn train-advanced \
  --task binary \
  --model tcn \
  --preprocess trial_delta \
  --no-relational-features \
  --aggregation mean \
  --hidden-channels 12 \
  --device cuda
```

JAR3 dùng `water_delta` và tự chọn ordinal objective:

```bash
python -m video_jar_gnn train-advanced \
  --task jar3 \
  --device cuda
```

Số epoch refit không còn lấy trung vị của ba epoch argmax nhiễu. Trainer lấy
trung bình BAcc, macro-F1 và loss tại từng epoch trên tất cả inner folds, rồi
chọn một điểm trên phần learning curve mà mọi fold đều đã quan sát. File
`selection_curve_pooled.csv` lưu bằng chứng lựa chọn đó.

## Thứ tự ablation cố định

Không chọn cấu hình theo điểm outer-test rồi báo lại cùng điểm đó. Nên giữ
outer folds cố định, dùng inner folds cho lựa chọn và thay một yếu tố mỗi lần:

1. TCN `trial_delta` cho binary hoặc `water_delta` cho JAR3, không relations,
   mean aggregation.
2. Đổi `--aggregation mean_std`.
3. Đổi `--temporal-pooling global`.
4. So sánh `--hidden-channels 8` và `12`.
5. Chỉ sau đó so sánh `--model stgcn`.

Khi đã chốt tối đa hai cấu hình bằng inner-CV, chạy ba seed `42`, `43`, `44`
để báo trung bình và độ lệch, không chọn seed cao nhất.

Water-delta dùng video nước của chính người cần dự đoán nhưng không dùng JAR
nước. Vì thế đây là giao thức **có một lượt calibration nước**. Nếu cần đánh
giá zero-shot hoàn toàn trên người mới, phải báo riêng kết quả `raw`,
`trial_delta` hoặc cache `neutral_delta`.
