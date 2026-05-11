# 🦷 Dental Disease Classification from X-ray Images

Hệ thống phân loại bệnh răng miệng từ ảnh X-quang sử dụng **Deep Learning (DenseNet201)** kết hợp **Google Gemini API** để đưa ra lời khuyên nha khoa tự động.

---

## 📋 Mô tả bài toán

Phân loại ảnh X-quang răng vào **5 nhóm bệnh**:

| Nhãn | Mô tả |
|------|-------|
| **Cavity** | Sâu răng |
| **Fillings** | Răng đã trám |
| **Impacted Tooth** | Răng mọc lệch / răng khôn kẹt |
| **Implant** | Răng cấy ghép implant |
| **Normal** | Răng bình thường, không bệnh |

**Input:** Ảnh X-quang răng (JPG, PNG)
**Output:** Nhãn phân loại + Độ tin cậy + Lời khuyên nha khoa từ AI

---

## 📈 Kết quả đạt được

### Tổng quan (Test Set — 1,649 ảnh)

| Metric | Giá trị |
|--------|---------|
| **Test Accuracy** | **93%** |
| **Test Loss** | **0.27** |
| Weighted F1-score | 0.93 |
| Macro F1-score | 0.76 |

### Chi tiết theo từng class

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Cavity | 0.43 | 0.41 | 0.42 | 22 |
| Fillings | 0.87 | 0.92 | 0.89 | 315 |
| Impacted Tooth | 0.52 | 0.75 | 0.62 | 32 |
| Implant | 0.95 | 0.90 | 0.93 | 104 |
| Normal | 0.97 | 0.94 | 0.96 | 1,176 |

### Phân tích lỗi (120/1,649 ảnh bị phân loại sai)

| Class thật | Số ảnh sai |
|------------|------------|
| Normal | 65 |
| Fillings | 24 |
| Cavity | 13 |
| Implant | 10 |
| Impacted Tooth | 8 |

**Nhận xét:**
- Model đạt hiệu quả rất tốt với **Normal** (F1=0.96), **Implant** (F1=0.93), **Fillings** (F1=0.89)
- **Cavity** có F1 thấp nhất (0.42) vì đặc điểm hình ảnh sâu răng dễ nhầm với Fillings trên ảnh X-quang (cả hai đều có vùng sáng/tối tương tự)
- **Impacted Tooth** cải thiện recall (0.75) nhờ class weight, nhưng precision vẫn còn thấp do dữ liệu ít

---

## 🔧 Kiến trúc & Kỹ thuật

### Kiến trúc mô hình (Transfer Learning)

```
DenseNet201 (pretrained ImageNet, fine-tuned toàn bộ)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ELU) → Dropout(0.5)
    ↓
Dense(128, ELU) → Dropout(0.3)
    ↓
Dense(32, ELU)
    ↓
Dense(5, Softmax)
```

### Tiền xử lý & Training

- Resize ảnh về **128×128 pixels**, normalize pixel về `[0, 1]`
- Xử lý mất cân bằng dữ liệu bằng `class_weight` từ sklearn
- **Optimizer:** Adam | **Loss:** `sparse_categorical_crossentropy` | **Epochs tối đa:** 50

### Callbacks

| Callback | Cấu hình |
|----------|----------|
| EarlyStopping | monitor=val_loss, patience=20, restore best weights |
| ReduceLROnPlateau | monitor=val_loss, factor=0.1, patience=10 |
| ModelCheckpoint | Lưu model tốt nhất theo val_loss |

### Dataset

| Tập | Số ảnh |
|-----|--------|
| Train | 25,136 |
| Validation | 2,812 |
| Test | 1,649 |
| **Tổng** | **29,597** |

---

## 🏗️ Cấu trúc project

```
project/
├── Model/
│   └── best_model.keras          # Model đã huấn luyện
├── Api/
│   └── api_advice.env            # Chứa GEMINI_API_KEY
├── src/
│   ├── app.py                    # Backend FastAPI
│   ├── frontend.py               # Frontend Streamlit
│   ├── model_training.ipynb      # Notebook huấn luyện model
│   ├── error_analysis.ipynb      # Notebook phân tích lỗi
│   └── gemini_integration.ipynb  # Notebook tích hợp Gemini
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### Yêu cầu

- Python 3.9+
- GPU (khuyến nghị) hoặc CPU

### Bước 1: Cài đặt thư viện

```bash
pip install tensorflow fastapi uvicorn streamlit pillow numpy \
            google-generativeai python-dotenv requests
```

### Bước 2: Cấu hình API Key

Tạo file `Api/api_advice.env` với nội dung:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Lấy API key miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey)

### Bước 3: Chạy Backend (FastAPI)

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

API sẽ chạy tại `http://127.0.0.1:8000`
Xem docs tương tác tại `http://127.0.0.1:8000/docs`

### Bước 4: Chạy Frontend (Streamlit)

Mở terminal mới, chạy:

```bash
streamlit run frontend.py
```

Giao diện sẽ tự mở tại `http://localhost:8501`

### Bước 5: Sử dụng

1. Truy cập giao diện Streamlit
2. Upload ảnh X-quang răng (JPG/PNG)
3. Nhấn **"🔍 Phân tích ngay"**
4. Xem kết quả chẩn đoán + lời khuyên từ AI

---

## 🔌 API Endpoint

### `POST /predict`

Nhận ảnh X-quang và trả về kết quả dự đoán + lời khuyên.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@your_xray.jpg"
```

**Response:**
```json
{
  "filename": "xray.jpg",
  "prediction": "Kết quả dự đoán: Sâu răng",
  "confidence": "87.45%",
  "gemini_advice": "Bạn có dấu hiệu sâu răng...",
  "status": "success"
}
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Deep Learning | TensorFlow / Keras |
| Model backbone | DenseNet201 (Transfer Learning) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Data processing | NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Training platform | Kaggle (GPU: 2× Tesla T4) |

---

## 🚀 Hướng phát triển

- Thêm data augmentation (rotation, brightness, flip) để cải thiện generalization
- Tăng input size lên 224×224 để phù hợp hơn với pretrained DenseNet201
- Áp dụng **Grad-CAM** để visualize vùng model tập trung trên ảnh X-quang
- Thu thập thêm dữ liệu cho class Cavity và Impacted Tooth
- Triển khai lên cloud (Docker + cloud hosting)

---

## ⚠️ Lưu ý

Hệ thống này chỉ mang tính chất **tham khảo**. Kết quả AI không thay thế chẩn đoán của bác sĩ nha khoa. Người dùng cần đến phòng khám nha khoa để được kiểm tra và tư vấn chính xác.

---

## 👤 Tác giả

**Trương Văn Thiên** — Dự án học tập về Computer Vision & AI Integration
