import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from dotenv import load_dotenv 
import google.generativeai as genai


# Tải biến môi trường từ file .env
load_dotenv("../Api/api_advice.env")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Khởi tạo model Gemini 
llm_model = genai.GenerativeModel('gemini-2.5-flash')

# --- KHỞI TẠO FASTAPI VÀ DENSENET MODEL ---
app = FastAPI(
    title="Dental AI & Advisory API",
    description="API dự đoán bệnh răng miệng và đưa ra lời khuyên tự động bằng Gemini",
    version="1.1.0"
)

print("Loading Deep Learning model...")
dl_model = tf.keras.models.load_model("../Model/best_model.keras")
print("Model loaded successfully!")

CLASS_NAMES = ['Cavity', 'Fillings', 'Impacted Tooth', 'Implant', 'Normal']

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict_and_advise(file: UploadFile = File(...)):
    try:
        # 1. AI MODEL XỬ LÝ ẢNH
        contents = await file.read()
        processed_image = preprocess_image(contents)
        predictions = dl_model.predict(processed_image)
        
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_class = CLASS_NAMES[predicted_index]
        
        # 2. GEMINI API ĐƯA RA LỜI KHUYÊN
        # Xây dựng prompt tự động dựa trên kết quả dự đoán
        prompt = f"""
        Một hệ thống AI nha khoa vừa phân tích ảnh X-quang của bệnh nhân và phát hiện tình trạng: '{predicted_class}'.
        Độ tin cậy của hệ thống chẩn đoán là {round(confidence * 100, 2)}%.
        Nhiệm vụ của bạn:
        Đóng vai một nha sĩ chuyên nghiệp, hãy đưa ra một lời khuyên ngắn gọn (khoảng 3-4 câu) bằng tiếng Việt 
        cho bệnh nhân về tình trạng này và các bước tiếp theo họ nên làm. 
        Lưu ý: Bắt buộc phải có một câu rào sau cùng rằng đây chỉ là chẩn đoán tham khảo từ AI, bệnh nhân cần đến phòng khám nha khoa để bác sĩ kiểm tra và chụp chiếu lại.
        """
        
        # Gọi Gemini sinh văn bản
        response = llm_model.generate_content(prompt)
        advice = response.text
        
        class_names = {
            "Cavity": "Sâu răng",
            "Fillings": "Trám răng",
            "Impacted Tooth": "Răng khôn mọc lệch",
            "Implant": "Cấy ghép răng",
            "Normal": "Răng bình thường"
        }
        # 3. TRẢ VỀ KẾT QUẢ TỔNG HỢP CHO CLIENT
        return {
            "filename": file.filename,
            "prediction": f"Kết quả dự đoán: {class_names.get(predicted_class, predicted_class)}",
            "confidence": f"{round(confidence * 100, 2)}%",
            "gemini_advice": advice,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}