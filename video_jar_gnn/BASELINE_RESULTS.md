# Kết quả baseline condition-level

Ngày chạy: 2026-07-27. Input là cache video hiện tại `[96,15,10]`. Mẫu nước
605 không được dùng làm target; `water_delta` chỉ dùng graph nước cùng người
làm reference.

Đây là nested 5-fold CV tách hoàn toàn theo subject. `C` của logistic và
trọng số late-fusion được chọn bằng bốn inner subject folds. Mỗi
`subject_id × ma_mau` chỉ được tính một lần sau khi gộp năm repeat.

## JAR3

| Cấu hình | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| `code_only` | 0.629 | 0.625 |
| `face_raw` | 0.384 | 0.383 |
| `face_trial_delta` | 0.400 | 0.399 |
| `face_water_delta` | 0.354 | 0.354 |
| `fusion_raw` | 0.637 | 0.631 |
| `fusion_trial_delta` | 0.629 | 0.624 |
| `fusion_water_delta` | 0.620 | 0.618 |

`fusion_raw` cao hơn `code_only` 0.008 điểm tuyệt đối, nhưng bootstrap paired
theo subject cho khoảng 95% của chênh lệch là `[-0.043, 0.064]`. Vì khoảng
này đi qua 0, chưa có bằng chứng video thêm thông tin ổn định ngoài mã mẫu.

Với `code_only`, MAE là 0.371 và quadratic weighted kappa là 0.711.

## Nhị phân vừa phải / khác

| Cấu hình | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| `code_only` | 0.656 | 0.642 |
| `face_raw` | 0.447 | 0.446 |
| `face_trial_delta` | 0.485 | 0.485 |
| `face_water_delta` | 0.552 | 0.552 |
| `fusion_raw` | 0.606 | 0.596 |
| `fusion_trial_delta` | 0.556 | 0.549 |
| `fusion_water_delta` | 0.594 | 0.588 |

`face_water_delta` là face-only tốt nhất, nhưng vẫn thấp hơn code-only 0.104
điểm. Khoảng bootstrap 95% của `face_water_delta - code_only` là
`[-0.217, 0.006]`; của `fusion_water_delta - code_only` là
`[-0.135, 0.009]`. Không fusion nào vượt code-only trên toàn bộ 5 fold.

## Diễn giải

- Baseline chỉ dùng mã mẫu đang mạnh; đây không phải target leakage vì model
  của mỗi outer fold chỉ fit nhãn outer-train.
- Graph khuôn mặt 96 time point hiện có tín hiệu yếu và không ổn định giữa
  người. Tăng epoch/model size trước khi cải thiện input có nguy cơ overfit.
- Bước tiếp theo hợp lý là thử cache 300/600 time point, neutral pre-context,
  chuẩn hóa góc mắt và feature quan hệ trong `train-advanced`.
- Water calibration phải được báo như một setting riêng vì khi triển khai nó
  cần video nước của người mới.

Artefact đầy đủ:

- `output/video_jar_gnn/runs_classical/jar3`
- `output/video_jar_gnn/runs_classical/binary`

Mỗi thư mục có `ablation_metrics.csv`, `predictions_condition.csv`,
`fold_metrics.csv`, confusion matrices, bootstrap trong `summary.json`, và
model `.joblib` của từng outer fold.
