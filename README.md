# 🦷 Dental Disease Classification from X-ray Images with AI

Hệ thống phân loại bệnh răng miệng từ ảnh X-quang sử dụng Deep Learning (DenseNet201) kết hợp Gemini API để đưa ra lời khuyên nha khoa tự động.

---

## 📋 Mô tả bài toán

Bài toán phân loại ảnh X-quang răng vào **5 nhóm**:

| Nhãn               | Mô tả                         |
| ------------------ | ----------------------------- |
| **Cavity**         | Sâu răng                      |
| **Fillings**       | Răng đã trám                  |
| **Impacted Tooth** | Răng mọc lệch / răng khôn kẹt |
| **Implant**        | Răng cấy ghép implant         |
| **Normal**         | Răng bình thường, không bệnh  |

**Input:** Ảnh X-quang răng (dental X-ray)  
**Output:** Nhãn phân loại + lời khuyên nha khoa từ AI

---

## 📊 Dataset

| Tập dữ liệu | Số lượng ảnh |
| ----------- | ------------ |
| Train       | 25,136       |
| Validation  | 2,812        |
| Test        | 1,649        |
| **Tổng**    | **29,597**   |

> Dataset được lưu trữ trên Kaggle. Phân phối các class không đồng đều (imbalanced), đặc biệt class **Normal** chiếm đa số.

---

## 🔧 Kỹ thuật xử lý

### 1. Tiền xử lý ảnh

- Resize về kích thước **128×128 pixels**
- Normalize pixel về khoảng `[0, 1]` bằng cách chia cho 255
- Sử dụng `ImageDataGenerator` với `flow_from_directory` để load dữ liệu theo batch

### 2. Xử lý mất cân bằng dữ liệu

- Tính `class_weight` bằng `sklearn.utils.class_weight.compute_class_weight` với chế độ `'balanced'`
- Truyền trọng số vào quá trình huấn luyện để model không bị thiên lệch về class Normal

### 3. Kiến trúc mô hình (Transfer Learning)

```
DenseNet201 (pretrained ImageNet, trainable=True)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, activation='elu') → Dropout(0.5)
    ↓
Dense(128, activation='elu') → Dropout(0.3)
    ↓
Dense(32, activation='elu')
    ↓
Dense(5, activation='softmax')
```

- **Backbone:** DenseNet201 với pretrained weights từ ImageNet
- **Fine-tuning:** Toàn bộ model được unfreeze (`trainable=True`)
- **Optimizer:** Adam (learning rate mặc định)
- **Loss function:** `sparse_categorical_crossentropy`

### 4. Callbacks trong quá trình training

| Callback            | Cấu hình                                            |
| ------------------- | --------------------------------------------------- |
| `EarlyStopping`     | monitor=val_loss, patience=20, restore best weights |
| `ReduceLROnPlateau` | monitor=val_loss, factor=0.1, patience=10           |
| `ModelCheckpoint`   | Lưu model tốt nhất theo val_loss                    |

- Training tối đa **50 epochs**

### 5. Tích hợp Gemini API

- Sau khi model dự đoán nhãn bệnh, gọi **Gemini 2.5 Flash** để sinh lời khuyên nha khoa tùy theo từng bệnh
- Phân tích ảnh bị phân loại sai bằng Gemini multimodal (gửi kèm ảnh X-quang + nhãn thật/dự đoán) để giải thích nguyên nhân nhầm lẫn

---

## 📈 Kết quả đạt được

### Test Set (1,649 ảnh)

| Metric            | Giá trị  |
| ----------------- | -------- |
| **Test Accuracy** | **93%**  |
| **Test Loss**     | **0.27** |
| Weighted F1-score | 0.93     |
| Macro F1-score    | 0.76     |

### Chi tiết theo từng class (Test Set)

| Class          | Precision | Recall | F1-score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| Cavity         | 0.43      | 0.41   | 0.42     | 22      |
| Fillings       | 0.87      | 0.92   | 0.89     | 315     |
| Impacted Tooth | 0.52      | 0.75   | 0.62     | 32      |
| Implant        | 0.95      | 0.90   | 0.93     | 104     |
| Normal         | 0.97      | 0.94   | 0.96     | 1,176   |

### Phân tích lỗi

Tổng số ảnh bị phân loại sai trên test set: **120/1,649 ảnh**

| Class thật     | Số ảnh sai |
| -------------- | ---------- |
| Normal         | 65         |
| Fillings       | 24         |
| Cavity         | 13         |
| Implant        | 10         |
| Impacted Tooth | 8          |

**Nhận xét:**

- Model đạt hiệu quả rất tốt trên **Normal** (F1=0.96), **Implant** (F1=0.93), và **Fillings** (F1=0.89)
- **Cavity** có F1 thấp nhất (0.42) do đặc điểm hình ảnh sâu răng dễ bị nhầm với Fillings (cả hai đều có vùng sáng/tối tương tự nhau trên X-quang)
- **Impacted Tooth** cải thiện recall (0.75) nhờ class weight, nhưng precision vẫn thấp

---

## 🏗️ Cấu trúc project

```
project-dip/
├── project-dip.ipynb       # Notebook chính
├── best_model.keras        # Model đã huấn luyện
├── misclassified/          # Ảnh bị phân loại sai (phân theo class)
│   ├── Cavity/
│   ├── Fillings/
│   ├── Impacted Tooth/
│   ├── Implant/
│   └── Normal/
└── wrong_Class.zip         # Nén toàn bộ ảnh sai để tải về
```

---

## 🛠️ Công nghệ sử dụng

- **Deep Learning:** TensorFlow / Keras
- **Model:** DenseNet201 (Transfer Learning)
- **Data Processing:** NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **LLM Integration:** Google Gemini 2.5 Flash API
- **Platform:** Kaggle (GPU: 2× Tesla T4)

---

## 🚀 Cách chạy

### Cấu hình API Key (Kaggle Secrets)

Vào **Add-ons → Secrets** trong Kaggle Notebook, thêm secret với tên `API`.

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("API")
genai.configure(api_key=api_key)
```

## 🚀 Hướng phát triển

- Thêm data augmentation (rotation, brightness, horizontal flip) để cải thiện generalization
- Tăng input size lên 224×224 để phù hợp hơn với pretrained DenseNet201
- Áp dụng thêm kỹ thuật như Grad-CAM để visualize vùng model tập trung
- Thu thập thêm dữ liệu cho class Cavity và Impacted Tooth để cân bằng dataset hơn

## 👤 Tác giả

**Trương Văn Thịên** — Dự án học tập về Computer Vision & AI Integration

## Hướng dẫn xem trực quan hơn: **ctrl + shift + v**
