import streamlit as st
from ai.predictor import predict_disease
from ai.chatbot import chat_response
from utils.helper import load_image

st.set_page_config(page_title="Mô hình AI dự đoán bệnh của cây trồng", layout="centered")
st.title("🧠 Mô hình AI dự đoán bệnh của cây trồng")

# Lưu lịch sử trò chuyện
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar upload ảnh
st.sidebar.header("📤 Gửi ảnh cây trồng")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh cây bị bệnh", type=["jpg", "png", "jpeg"])

# Hiển thị và dự đoán bệnh
if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="📷 Ảnh bạn đã gửi", use_container_width=True)
    with st.spinner("⏳ Đang phân tích ảnh..."):
        label, confidence = predict_disease(image)
    st.success(f"🌿 Kết quả: **{label}** ({confidence*100:.2f}%)")
    st.info("💡 Hãy đặt câu hỏi để nhận thêm thông tin về bệnh này.")

# Khu vực trò chuyện
st.subheader("💬 Trò chuyện")
user_input = st.text_input("Nhập câu hỏi hoặc tin nhắn:")

if st.button("Gửi"):
    if user_input.strip() != "":
        st.session_state.chat_history.append(("🧑‍🌾 Bạn", user_input))
        with st.spinner("🤖 Đang trả lời..."):
            try:
                response = chat_response(user_input)
            except Exception as e:
                response = f"⚠️ Lỗi: {str(e)}"
        st.session_state.chat_history.append(("🤖 AI", response))

# Hiển thị lịch sử trò chuyện
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}**: {message}")
