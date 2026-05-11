import streamlit as st
import requests
from PIL import Image

# 1. Cấu hình trang web (Chỉnh layout thành wide để rộng rãi hơn)
st.set_page_config(page_title="Nha Khoa AI", page_icon="🦷", layout="centered")

st.title("🦷 Trợ lý Nha khoa AI")
st.markdown("""
Hệ thống sử dụng **Deep Learning (DenseNet201)** để phân tích ảnh X-quang răng và 
**Google Gemini** để đưa ra lời khuyên nha khoa tự động.
""")
st.divider()

# 2. Nút tải ảnh lên
uploaded_file = st.file_uploader("Tải ảnh X-quang của bạn lên đây (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Chia 2 dòng chỉ để chứa ảnh và nút bấm
    # col1, col2 = st.columns([1, 1])
    
    # with col1:
    st.image(Image.open(uploaded_file), caption="Ảnh X-quang tải lên", use_container_width=True)
    
    # with col2:
    #     st.write("### Nhấn nút bên dưới để phân tích")
        # Gán nút bấm vào một biến để kiểm tra trạng thái
    analyze_button = st.button("🔍 Phân tích ngay", type="primary", use_container_width=True)
    
    #  PHẦN KẾT QUẢ ĐƯỢC ĐƯA RA NGOÀI (Chiếm 100% chiều ngang)
    if analyze_button:
        st.divider() # Thêm một đường kẻ ngang 
        
        with st.spinner("AI đang phân tích ảnh và hỏi ý kiến bác sĩ Gemini..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    if data["status"] == "success":
                        st.success("✅ Phân tích thành công!")
                        
                        # In đậm kết quả chẩn đoán
                        st.metric(label="Chẩn đoán của AI", value=data["prediction"], delta=f"Độ tin cậy: {data['confidence']}")
                        
                        # Hiển thị lời khuyên ở một khung màu xanh rộng rãi
                        st.info("💡 **Lời khuyên từ Trợ lý nha khoa:**")
                        st.write(data["gemini_advice"])
                    else:
                        st.error(f"Lỗi từ mô hình: {data.get('error')}")
                else:
                    st.error("Không thể kết nối với hệ thống AI. Vui lòng kiểm tra lại server FastAPI.")
            
            except Exception as e:
                st.error(f"Lỗi kết nối mạng: {e}")