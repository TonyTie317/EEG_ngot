# Video ST-GCN cho phân loại JAR

Thư mục này là pipeline độc lập để phân loại phản ứng khuôn mặt theo độ ngọt
vừa phải. Mã trong `Video/` chỉ được dùng làm tham khảo; pipeline mới không sửa
pipeline EEG và không cắt/ghi đè 108 GB video gốc.

## Bài toán và dữ liệu đã kiểm tra

Ba nguồn được nối bằng khóa duy nhất
`(subject_id, ma_mau, repeat)`:

- `data/data_video/Nxx_vid-001.mp4`: 28 video.
- `data/data_video (2)/Nxx_vid.csv`: nhãn frame `ma_mau`, `lan_lap`.
- `data/datadone/sub-Pxxx_..._eeg.csv`: nguồn chuẩn của JAR.

`N01` tương ứng `P001`. Hai người 12 và 22 không có trong cả ba nguồn. Dữ
liệu có đủ `28 × 6 × 5 = 840` trial, mỗi trial được đánh dấu 600 frame. Sáu
mã mẫu là năm mẫu ngọt `189, 258, 453, 762, 893` và nước `605`.

Hai target:

- `binary`: `0=Khác (JAR 1,2,4,5)`, `1=Vừa phải (JAR 3)`.
- `jar3`: `0=Không đủ (1,2)`, `1=Vừa phải (3)`,
  `2=Quá nhiều (4,5)`.

Phân bố đủ 840 trial là `415 / 220 / 205` cho JAR3 và `620 / 220` cho
nhị phân. Vì JAR giống nhau ở năm lần lặp của cùng người và mẫu, chỉ có
168 rating độc lập nếu tính nước, hoặc 140 rating cho năm mẫu ngọt.

### Video 60 fps, không phải EEG 100 Hz

Việc cắt trial chỉ đọc CSV frame label của video: một dòng tương ứng một frame
video 60 fps. Pipeline không dùng tần số EEG 100 Hz để map frame. Khóa nối
đúng là `subject_id + ma_mau + repeat`; JAR được kiểm tra nhất quán theo
`subject_id + ma_mau` rồi gắn vào năm repeat. Chỉ dùng `ma_mau` mà bỏ
`subject_id` sẽ nối nhãn sai giữa những người khác nhau.

`--num-frames 96` của representation `legacy` là bước giảm mẫu có chủ đích
từ 600 frame/10 giây để tạo cache nhỏ, không phải giả định video có 96 frame.
Representation mới `expression_v2` mặc định tạo 600 time point trong 10 giây
và tự dùng thư mục cache/manifest riêng. `target_lsl` mới là trục thời gian
cho đạo hàm và response window; khi capture thực tế chậm hơn 60 fps, một số
frame nguồn gần nhất có thể được chọn lặp nhưng tensor vẫn có 600 mốc đều
theo thời gian thật.

## Thiết kế

Pipeline gồm các phần:

1. `prepare`: đọc ba nguồn ở chế độ read-only, kiểm tra thiếu/trùng/mâu thuẫn
   và tạo manifest 840 dòng.
2. `extract`: lấy trực tiếp các frame cần thiết từ video dài, chạy
   MediaPipe FaceMesh; có graph legacy `15 × 10` và graph expression_v2
   `20 × 8` trong hai cache tách biệt.
3. `train`: baseline ST-GCN cũ ở mức repeat, giữ lại để tái lập các run đầu.
4. `train-video-features`: baseline video-only có chọn feature hoàn toàn bên
   trong inner subject folds.
5. `train-classical`: benchmark chẩn đoán lịch sử; các nhánh code/fusion không
   phải input của mô hình advanced.
6. `train-advanced`: huấn luyện ST-GCN, TCN hoặc GRU pure-video trên cả tập
   repeat và chỉ tính một loss cho mỗi `subject × ma_mau`.
7. `audit-expression`: đo độ ổn định của năm repeat theo cửa sổ giây thật,
   hoàn toàn không đọc JAR.
8. `train-expression`: baseline logistic dung lượng nhỏ; response window,
   số feature và regularization chỉ được chọn trong inner subject folds.

Các sửa đổi quan trọng so với mã tham khảo:

- Không gộp `min/max(ma_mau)`, vì cách đó nối nhầm năm repeat và cả background.
- Mặc định resample 96 frame theo `t_lsl` trong đúng 10 giây thật. Có 129/840
  block 600-frame dài hơn 10,5 giây do capture bị drop frame.
- Node mắt trên/dưới dùng landmark riêng, không còn feature trùng nhau.
- Tọa độ được chuẩn hóa theo tâm và khoảng cách hai mắt; aspect dùng log-ratio
  có chặn để loại outlier khoảng 160.000 của pipeline cũ.
- Feature normalization chỉ fit trên subject ở fold train.
- Model thêm self-loop đúng một lần, BatchNorm đúng layout, và dùng edge weight
  dương đối xứng trên các cạnh giải phẫu có sẵn.
- Số epoch được chọn từ learning curve validation trung bình của tất cả inner
  subject folds; model sau đó được khởi tạo lại và fit trên toàn bộ outer-train
  trước khi đánh giá subject chưa từng thấy.
- Metric chính là balanced accuracy và macro-F1; accuracy nhị phân đơn thuần
  dễ gây hiểu nhầm vì baseline đoán toàn `Khác` đã đạt 68,6% trên cấu hình
  mặc định không-nước (73,8% nếu tính cả nước).
- Advanced model và checkpoint không có code embedding/code prior. `ma_mau`
  chỉ là khóa nối/gom repeat ở tầng dữ liệu và nhận biết nước 605; nó không
  được trả về dataset batch hay đưa vào model.

### Representation `expression_v2` (khuyến nghị)

Cache cũ không lưu 468 landmark nên không thể chuyển đổi sang representation
mới; phải extract lại từ video. Mỗi trial mới có shape `[600,20,8]`:

- 20 proxy hình học hai bên cho mày, mắt, má, mũi, khóe môi và hàm;
- 8 kênh `value`, `delta`, `velocity`, `acceleration`, `abs_velocity`,
  `baseline_robust_z`, `observed`, `imputed`;
- chuẩn hóa translation/scale và giảm roll/yaw/pitch bằng hệ trục mắt–mũi
  cùng rigid alignment theo template riêng của từng trial ở 1,25 giây đầu;
- chỉ nội suy gap mất mặt nội bộ không quá 0,5 giây;
- value/vận tốc/gia tốc ước lượng bằng local quadratic 300 ms để tránh
  khuếch đại landmark jitter ở 60 Hz.

Đây là **expression geometry proxy**, không phải điểm FACS AU đã được kiểm
định. Hai kênh mask không bị standardize hay thêm noise. `ma_mau` chỉ dùng
để gom đúng năm repeat của một condition; không nằm trong `graph_seq` và
không phải predictor.

Thiết kế ablation, thứ tự thí nghiệm và tiêu chí kết luận nằm trong
[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md). Kết quả nested 5-fold hiện tại
được lưu tại [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md).

## Cài môi trường

Python mặc định hiện tại chưa có các thư viện video/ML. Nên tạo môi trường
Python 3.10 hoặc 3.11:

```bash
# Cách 1: conda
conda create -n video-jar-gnn python=3.11 -y
conda activate video-jar-gnn
python -m pip install -r video_jar_gnn/requirements.txt

# Cách 2: venv, nếu máy đã có executable python3.11
python3.11 -m venv .venv-video
source .venv-video/bin/activate
python -m pip install -U pip
python -m pip install -r video_jar_gnn/requirements.txt
```

Kiểm tra API trước khi chạy video:

```bash
python -c "import cv2, mediapipe as mp; print(cv2.__version__, mp.__version__); assert hasattr(mp, 'solutions')"
```

Pipeline giải mã bằng OpenCV nên không cần gọi executable `ffmpeg` hoặc
`ffprobe`. `mediapipe` được pin ở `0.10.21` vì extractor dùng API
`mp.solutions.face_mesh`; không tự ý nâng version nếu chưa chuyển extractor
sang MediaPipe Tasks.

Extractor FaceMesh hiện dùng MediaPipe Solutions/XNNPACK trên CPU; các dòng
EGL/OpenGL không có nghĩa toàn bộ extraction chạy CUDA. `--device cuda` chỉ
áp dụng cho `train`/`train-advanced`. `train-expression` là scikit-learn
logistic regression nên cũng chạy CPU.

## Chạy

### 1. Tạo và audit manifest

```bash
python -m video_jar_gnn prepare \
  --video-dir data/data_video \
  --frame-label-dir "data/data_video (2)" \
  --jar-dir data/datadone \
  --output output/video_jar_gnn/manifest.csv
```

Kết quả:

- `output/video_jar_gnn/manifest.csv`
- `output/video_jar_gnn/manifest.audit.json`

Lệnh dừng ngay nếu một trong 840 khóa join bị thiếu, trùng hoặc có JAR mâu
thuẫn.

### 2. Smoke test extraction

Trước khi xử lý toàn bộ 108 GB, kiểm tra hai trial:

```bash
python -m video_jar_gnn extract \
  --manifest output/video_jar_gnn/manifest.csv \
  --subjects P001 \
  --limit 2 \
  --output-dir output/video_jar_gnn/graphs \
  --output-manifest output/video_jar_gnn/graph_manifest_smoke.csv
```

Sau khi kiểm tra `extract_status`, `detection_ratio` và các `.npz`, chạy đầy đủ:

```bash
python -m video_jar_gnn extract \
  --manifest output/video_jar_gnn/manifest.csv \
  --output-dir output/video_jar_gnn/graphs \
  --output-manifest output/video_jar_gnn/graph_manifest.csv
```

Extraction có thể chạy lại an toàn: graph đã tồn tại sẽ được dùng lại; thêm
`--overwrite` khi thực sự muốn trích lại. Để lấy mẫu trên toàn khoảng 600
frame cũ cho sensitivity analysis, dùng `--duration-mode labelled`; chế độ
này vẫn resample về `--num-frames` (mặc định 96). Nếu cần giữ đủ 600 điểm,
dùng cả `--duration-mode labelled --num-frames 600`.

`graph_manifest.csv` ghi cả `n_unique_frames` và `timing_source`. Cảnh báo
`unique source frames` không làm dừng extraction: nó cho biết timestamp bị
giãn/drop-frame khiến frame nguồn gần nhất được chọn lặp khi resample đều.

### 2b. Extract và kiểm tra `expression_v2`

Smoke-test mười trial đầu (hai condition, đủ năm repeat mỗi condition):

```bash
python -m video_jar_gnn extract \
  --representation expression_v2 \
  --limit 10
```

Mặc định lệnh trên dùng:

- `600` time point trong khoảng 10 giây theo `target_lsl`;
- `output/video_jar_gnn/graphs_expression_v2`;
- `output/video_jar_gnn/graph_manifest_expression_v2.csv`.

Sau khi smoke-test đạt, bỏ `--limit` để extract toàn bộ. Lệnh có thể tiếp tục
an toàn vì cache đúng cấu hình được tái sử dụng:

```bash
python -m video_jar_gnn extract \
  --representation expression_v2
```

Đo repeat reliability trước khi huấn luyện. Công cụ này mặc định bỏ nước 605
và không đọc nhãn JAR:

```bash
python -m video_jar_gnn audit-expression \
  --manifest output/video_jar_gnn/graph_manifest_expression_v2.csv
```

Sáu cửa sổ mặc định là `0:2`, `2:4`, `4:6`, `6:8`, `8:10`, `0:10` giây.
Đọc `window_metrics.csv`: ICC subject-centered càng cao, tỉ số khoảng cách
within/between càng nhỏ hơn 1 và pair AUC càng lớn hơn 0,5 thì năm repeat càng
chứa tín hiệu condition ổn định. Không diễn giải audit smoke chỉ có một người
như kết quả toàn tập.

Baseline phân loại được khuyến nghị chạy trước GNN:

```bash
python -m video_jar_gnn train-expression \
  --task binary

python -m video_jar_gnn train-expression \
  --task jar3
```

Đây là logistic regression chạy CPU, không nhận `--device cuda`. Trong từng
outer fold, inner subject-CV chọn đồng thời response window, số feature và
regularization `C`; scaler, imputer, variance filter và SelectKBest cũng chỉ
fit trên subject train. Output mặc định nằm trong
`output/video_jar_gnn/runs_expression/<task>/...`. Có thể smoke-test bằng
`--fold-index 0`; không dùng kết quả partial fold làm kết quả cuối.

### 3. Phân loại JAR3

Mặc định loại nước `605` vì JAR về mức độ ngọt của nước gần như hằng
(27/28 rating là 1, một rating là 2) và không phải đánh giá mức sucrose:

```bash
python -m video_jar_gnn train \
  --manifest output/video_jar_gnn/graph_manifest.csv \
  --task jar3 \
  --output-dir output/video_jar_gnn/runs/jar3
```

Để chạy sensitivity analysis có nước, thêm `--include-water`.

### 4. Phân loại vừa phải / khác

```bash
python -m video_jar_gnn train \
  --manifest output/video_jar_gnn/graph_manifest.csv \
  --task binary \
  --output-dir output/video_jar_gnn/runs/binary
```

Mặc định là stratified group 5-fold. Đánh giá chặt hơn bằng 28-fold LOSO:

```bash
python -m video_jar_gnn train \
  --manifest output/video_jar_gnn/graph_manifest.csv \
  --task jar3 \
  --loso \
  --output-dir output/video_jar_gnn/runs/jar3_loso
```

Có thể chạy thử một fold với `--fold-index 0 --epochs 2 --patience 1`.

### 5. Baseline video feature-selection nên chạy trước

Baseline này loại feature hằng, chọn `16/32/64` hoặc toàn bộ feature và chọn
regularization hoàn toàn trong inner subject folds; không tạo feature từ mã
mẫu:

```bash
python -m video_jar_gnn train-video-features --task binary
python -m video_jar_gnn train-video-features --task jar3
```

`water_delta` dùng nước 605 như reference không nhãn. Cột `ma_mau` nếu có
trong output chỉ giúp truy vết condition, không nằm trong ma trận feature.
`selected_feature_indices.csv` ghi cả vùng mặt, kênh gốc, thống kê thời gian
và F-score tính trên train fold để kiểm tra mô hình đang dựa vào tín hiệu nào.

### 6. Benchmark classical lịch sử

Hai lệnh này chạy cùng subject splits cho bảy ablation: `code_only`, ba
face-only (`raw`, `trial_delta`, `water_delta`) và ba late-fusion:

```bash
python -m video_jar_gnn train-classical --task jar3
python -m video_jar_gnn train-classical --task binary
```

Mỗi condition chỉ xuất hiện một lần sau khi gộp tối đa năm repeat. Logistic
regression dùng class weight; `C` và trọng số fusion đều được chọn bằng inner
subject folds. Mẫu nước 605 chỉ được dùng làm reference không nhãn trong
`water_delta`, không được dùng làm target.

Output mặc định:

- `output/video_jar_gnn/runs_classical/jar3/<run-signature>`
- `output/video_jar_gnn/runs_classical/binary/<run-signature>`

Đọc bảng chính tại `ablation_metrics.csv`; `summary.json` có bootstrap theo
subject và chênh lệch so với code-only. Thư mục run chứa seed, full/fold và
hash cấu hình; trainer từ chối ghi vào một thư mục không rỗng để không trộn
artefact của hai thí nghiệm.

Các nhánh code-only/fusion ở lệnh này chỉ dùng để chẩn đoán dữ liệu và tái lập
kết quả cũ; không được dùng trong `train-advanced`.

### 7. Condition-level ST-GCN / TCN / GRU

Advanced trainer hiện là pure-video. Cấu hình mặc định đã được thu gọn thành
TCN không relational feature, mean aggregation và 12 hidden channels (khoảng
7 nghìn tham số). `auto` dùng `trial_delta` cho binary và `water_delta` cho
JAR3:

```bash
python -m video_jar_gnn train-advanced \
  --task binary \
  --device cuda
```

JAR3 mặc định dùng ordinal objective:

```bash
python -m video_jar_gnn train-advanced \
  --task jar3 \
  --device cuda
```

Trainer này tạo một sample `[repeat, node, time, feature]` cho mỗi
`subject × ma_mau`, encode từng repeat bằng shared encoder rồi masked-pool
trước khi tính một loss. Mặc định có:

- cân bằng lớp trên 140 condition, không phải trên 700 repeat;
- chuẩn hóa góc nghiêng theo hai mắt;
- relational feature tắt mặc định vì audit cho thấy bị chi phối bởi subject;
- pooling riêng ba pha `0–20%`, `20–50%`, `50–100%`;
- mean pooling giữa repeat; `mean_std` chỉ là một ablation;
- ordinal loss cho JAR3;
- cluster bootstrap theo subject.

Để so sánh kiến trúc, thay `--model` bằng `stgcn`, `tcn`, `gru`. Bằng chứng
và thứ tự ablation nằm tại
[`VIDEO_SIGNAL_DIAGNOSIS.md`](VIDEO_SIGNAL_DIAGNOSIS.md). Chạy
`--fold-index 0 --epochs 2 --min-epochs 1 --patience 1` trước như smoke test.

Với cache mới, luôn ghi rõ representation. `auto` dùng trực tiếp các kênh
delta/robust-z đã được extractor tính theo 1,25 giây thật và tắt phép xoay
legacy:

```bash
python -m video_jar_gnn train-advanced \
  --representation expression_v2 \
  --task binary \
  --model tcn \
  --device cuda

python -m video_jar_gnn train-advanced \
  --representation expression_v2 \
  --task jar3 \
  --model tcn \
  --device cuda
```

Chỉ chạy deep model sau khi baseline chọn response window cho thấy tín hiệu
ổn định hơn chance. Với 600 time point, bắt đầu bằng TCN nhỏ; ST-GCN giữ
activation theo cả time × node nên tốn VRAM hơn.

### 8. Baseline trung tính đúng 60 frame/giây

Cache cũ không có frame ngay trước trial. Muốn dùng `neutral_delta`, trích ra
một thư mục riêng; lệnh dưới đây lấy đúng 60 frame trong một giây `ma_mau=0`
liên tục ngay trước mỗi trial:

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
  --preprocess neutral_delta_motion \
  --device cuda
```

Audit dữ liệu hiện tại cho thấy cả 840 trial đều có đoạn `ma_mau=0` ngay
trước trial. Cache lưu thêm `baseline_seq [60,15,10]`, frame/LSL timestamps,
detection ratio và fingerprint riêng.

### 9. Water calibration

Face-only water delta:

```bash
python -m video_jar_gnn train-advanced \
  --task jar3 \
  --model tcn \
  --preprocess water_delta \
  --no-relational-features \
  --device cuda
```

Water reference dùng các graph 605 của chính subject nhưng không dùng JAR
nước. Vì vậy chế độ này yêu cầu một lượt calibration bằng nước khi áp dụng
cho người mới. Không dùng `absolute_water_delta` làm cấu hình chính: audit
hiện tại cho thấy nhánh raw absolute giữ nhiều đặc trưng danh tính và làm giảm
khả năng tổng quát sang subject mới.

## Output huấn luyện

Mỗi run chứa:

- `run_config.json`: toàn bộ tham số, số mẫu bị loại và shape graph.
- `fold_XX/model.pt`: model, train-only normalizer và danh sách subject train/test.
- `fold_metrics.csv`: metric từng fold.
- `predictions_trial.csv`: dự đoán cho từng repeat.
- `predictions_condition.csv`: probability trung bình của các repeat đạt ngưỡng
  quality (tối đa năm); cột `n_repeats` cho biết số repeat thực tế.
- `summary.json`: balanced accuracy, macro-F1, recall từng lớp, baseline.
- `confusion_trial.*`, `confusion_condition.*`: ma trận nhầm lẫn.

Kết luận chính nên dựa trên `subject_condition_level` trong `summary.json`.
Năm repeat gốc chia sẻ cùng một rating; nếu một repeat bị loại bởi ngưỡng
quality, condition còn lại vẫn được gộp và số lượng được ghi rõ. Coi 700
repeat ngọt là 700 quan sát JAR độc lập sẽ phóng đại độ chắc chắn.

## Kiểm thử

Sau khi cài dependencies:

```bash
python -m unittest discover -v -s video_jar_gnn/tests -t .
```

Test suite kiểm tra join theo khóa, resample LSL, graph contract, tensor
forward/backward, detection mask, metric khi fold thiếu lớp và một nested-CV
synthetic hoàn chỉnh.

## Cấu trúc graph

Mỗi `.npz` gồm:

- `graph_seq [T,15,10]`
- `adj [15,15]`
- `sampled_frame_idx`, `sampled_lsl`, `target_lsl`
- `detection_ratio`
- `meta`
- Nếu bật pre-context: `baseline_seq [60,15,10]` và timestamp/detection
  tương ứng.

15 node legacy đại diện lông mày, mắt, mũi, môi và cằm. Mỗi node có tọa độ chuẩn
hóa, cờ detect, diện tích, log-aspect và bốn vận tốc theo giây. Graph chỉ chứa
biểu hiện khuôn mặt; mã mẫu và subject ID chỉ dùng để join/split/evaluate.

Cache expression_v2 có thêm `pose_seq [T,4]` để audit head motion và metadata
tự mô tả node/feature/mask. `graph_seq [T,20,8]` vẫn không chứa mã mẫu,
subject ID hay JAR.
