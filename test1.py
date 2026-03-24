import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Tải mô hình
model = load_model('model/plant_disease_model.h5')

# Danh sách các lớp bệnh (tuỳ theo dữ liệu huấn luyện)
class_names = ['Khỏe mạnh', 'Bệnh đốm lá', 'Bệnh phấn trắng', 'Bệnh gỉ sắt']

# Tiêu đề
st.title('🌿 Dự đoán bệnh trên lá cây bằng AI')

# Tải ảnh
uploaded_file = st.file_uploader("📷 Vui lòng tải lên ảnh lá cây", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    class_names = ['Khỏe mạnh', 'Bệnh đốm lá', 'Bệnh phấn trắng', 'Bệnh gỉ sắt']
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Ảnh đã tải lên', use_container_width=True)

    # Xử lý ảnh
    img = image_data.resize((224, 224))  # Resize ảnh đúng kích thước mô hình
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Hiển thị kết quả
    st.subheader("🧠 Kết quả dự đoán:")
    st.write(f"**{predicted_class}** ({confidence:.2f}%)")

    # Gợi ý xử lý cơ bản
    if predicted_class != 'Khỏe mạnh':
        st.info("👉 Hãy cắt bỏ lá bị bệnh, hạn chế tưới nước lên lá, sử dụng thuốc phòng trừ phù hợp.")
    else:
        st.success("✅ Cây trồng đang khỏe mạnh!")
