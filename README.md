# OCR Benchmark Suite

Đánh giá **WER**, **CER**, **tốc độ (latency)** cho các mô hình OCR dùng HuggingFace Transformers.

---

## Cài đặt
- Cài đặt các thư viện theo yêu cầu của mô hình cần test trên Huggingface
- Ví dụ: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

```bash
pip install git+https://github.com/huggingface/transformers
# pip install transformers==4.57.0 # currently, V4.57.0 is not released
```

---

## Chuẩn bị Dataset

Hỗ trợ **3 cấu trúc thư mục**:

### Cấu trúc A — Image + TXT cùng tên (đơn giản nhất)
```
data/
  img_001.jpg
  img_001.txt      ← nội dung: "Xin chào thế giới"
  img_002.png
  img_002.txt
```

### Cấu trúc B — File labels tập trung
```
data/
  images/
    img_001.jpg
    img_002.jpg
  labels.txt       ← format: "img_001.jpg\tground truth text"
```

Hoặc dùng JSON:
```
data/
  images/
    img_001.jpg
  labels.json      ← {"img_001.jpg": "ground truth text"}
```

### Cấu trúc C — Thư mục con (đệ quy)
```
data/
  folder_1/
    img_001.jpg
    img_001.txt
  folder_2/
    img_002.jpg
    img_002.txt
```
---

## Chạy evaluation

- Chạy toàn bộ tập dataset đã upload
```bash
python evaluate.py \
  --dataset ./data \
  --output results/
```

- Chạy thử nghiệm với n samples đầu tiên
```bash
python benchmark.py \
  --dataset ./data \
  --max-samples 50 \
  --output results/
```

Trong thư mục `results` sẽ bao gồm:
- `summaries.json`: Lưu trữ các chỉ số tổng hợp ($WER$, $CER$, $Latency$) của tất cả các mô hình đã thử nghiệm. Giúp so sánh nhanh hiệu năng giữa các phiên bản.
- Các file `<model_1>.csv`, `<model_2>.csv`, ...: Nhật ký chi tiết. Lưu kết quả dự đoán của từng ảnh thực tế. Cho phép kiểm chứng các trường hợp mô hình đọc sai dấu hoặc ký tự đặc biệt.

---

## Metrics giải thích

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **WER** | edit_dist(words) / len(ref_words) | Tỉ lệ lỗi ở cấp từ |
| **CER** | edit_dist(chars) / len(ref_chars) | Tỉ lệ lỗi ở cấp ký tự |
| **Latency avg** | Trung bình thời gian inference | Tốc độ trung bình |
| **p50** | Percentile 50 của latency | Latency median |
| **p95** | Percentile 95 của latency | Latency worst-case |

WER và CER **càng thấp càng tốt** (0% = hoàn hảo).

---