import json
import os
import time
import importlib
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import streamlit as st

from ai.chatbot import chat_response
from ai.predictor import predict_disease
from nlp.disease_info import get_disease_info
from utils.helper import load_image

BASE_DIR = Path(__file__).resolve().parent


def load_env_file(file_path: Path):
    if not file_path.exists():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file(BASE_DIR / ".env")


@st.cache_data
def load_label_maps():
    with open(BASE_DIR / "model" / "class_labels_vi.json", "r", encoding="utf-8") as file:
        vi_map = json.load(file)
    with open(BASE_DIR / "model" / "class_names.json", "r", encoding="utf-8") as file:
        class_names = json.load(file)
    reverse_vi_map = {value: key for key, value in vi_map.items()}
    return class_names, vi_map, reverse_vi_map


def severity_from_confidence(confidence):
    if confidence >= 0.85:
        return "Cao"
    if confidence >= 0.65:
        return "Trung bình"
    return "Thấp"


def build_recommendation(label_en, disease_text):
    if not disease_text:
        return "Giữ cây khô thoáng, theo dõi thêm và tham khảo chuyên gia nông nghiệp nếu triệu chứng lan rộng."
    if "healthy" in label_en.lower():
        return "Cây đang khỏe. Tiếp tục chăm sóc, tưới nước hợp lý và kiểm tra lá định kỳ."
    return disease_text


def fetch_ai_disease_sections(label_vi, label_en, confidence, disease_text):
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

    if not groq_api_key and not gemini_api_key:
        return {
            "cause": disease_text or "Chưa có dữ liệu.",
            "treatment": "Chưa cấu hình GROQ_API_KEY hoặc GEMINI_API_KEY.",
            "prevention": "Chưa cấu hình GROQ_API_KEY hoặc GEMINI_API_KEY.",
            "source": "fallback",
            "error": "Thiếu GROQ_API_KEY và GEMINI_API_KEY",
        }

    symptom_text = disease_text.strip() if disease_text else "Chưa có dữ liệu triệu chứng từ nội bộ."

    prompt = f"""
Bạn là chuyên gia bệnh học cây trồng.

Thông tin đầu vào:
- Tên bệnh tiếng Việt: {label_vi}
- Tên kỹ thuật: {label_en}
- Độ tin cậy mô hình: {confidence:.2f}
- Triệu chứng nội bộ hiện có: {symptom_text}

Hãy trả về đúng JSON hợp lệ với 4 khóa:
symptoms, cause, treatment, prevention

Yêu cầu:
- Viết bằng tiếng Việt
- Ngắn gọn, rõ ràng, dễ hiểu
- Không markdown
- Không giải thích thêm ngoài JSON

Ví dụ:
{{
  "symptoms": "Mô tả triệu chứng dễ hiểu...",
  "cause": "Nguyên nhân gây bệnh...",
  "treatment": "Cách điều trị...",
  "prevention": "Cách phòng ngừa..."
}}
""".strip()

    last_error = ""

    if groq_api_key:
        try:
            GroqClient = importlib.import_module("groq").Groq
        except ModuleNotFoundError:
            last_error = "Groq: chưa cài thư viện 'groq'."
            GroqClient = None

    if groq_api_key and 'GroqClient' in locals() and GroqClient is not None:
        client = GroqClient(api_key=groq_api_key)

        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=groq_model_name,
                    temperature=0.3,
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là chuyên gia bệnh học cây trồng. Chỉ trả về JSON hợp lệ."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                text = response.choices[0].message.content.strip()

                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()

                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end + 1]

                data = json.loads(text)

                symptoms = data.get("symptoms", "").strip()
                cause = data.get("cause", "").strip()
                treatment = data.get("treatment", "").strip()
                prevention = data.get("prevention", "").strip()

                return {
                    "symptoms": symptoms or symptom_text,
                    "cause": cause or "Chưa có dữ liệu nguyên nhân.",
                    "treatment": treatment or "Chưa có dữ liệu điều trị.",
                    "prevention": prevention or "Chưa có dữ liệu phòng ngừa.",
                    "source": "groq",
                    "error": "",
                }

            except Exception as e:
                last_error = f"Groq: {e}"
                time.sleep(2 * (attempt + 1))

    if gemini_api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={gemini_api_key}"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        for attempt in range(2):
            try:
                req = Request(
                    url=url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urlopen(req, timeout=45) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))

                candidates = payload.get("candidates", [])
                if not candidates:
                    raise ValueError("Gemini không trả về candidates")

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError("Gemini không trả về parts")

                text = parts[0].get("text", "").strip()

                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()

                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end + 1]

                data = json.loads(text)

                symptoms = str(data.get("symptoms", "")).strip()
                cause = str(data.get("cause", "")).strip()
                treatment = str(data.get("treatment", "")).strip()
                prevention = str(data.get("prevention", "")).strip()

                return {
                    "symptoms": symptoms or symptom_text,
                    "cause": cause or "Chưa có dữ liệu nguyên nhân.",
                    "treatment": treatment or "Chưa có dữ liệu điều trị.",
                    "prevention": prevention or "Chưa có dữ liệu phòng ngừa.",
                    "source": "gemini",
                    "error": "",
                }

            except HTTPError as e:
                try:
                    detail = e.read().decode("utf-8")
                except Exception:
                    detail = str(e)
                last_error = f"Gemini HTTP {e.code}: {detail}"
                time.sleep(2 * (attempt + 1))
            except (URLError, ValueError, json.JSONDecodeError) as e:
                last_error = f"Gemini: {e}"
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                last_error = f"Gemini: {e}"
                time.sleep(2 * (attempt + 1))

    return {
        "symptoms": symptom_text,
        "cause": "Không lấy được dữ liệu AI.",
        "treatment": "Không lấy được dữ liệu AI.",
        "prevention": "Không lấy được dữ liệu AI.",
        "source": "fallback",
        "error": last_error,
    }

def analyze_uploaded_image(uploaded_file, reverse_vi_map):
    image = load_image(uploaded_file)
    label_vi, confidence = predict_disease(image)
    label_en = reverse_vi_map.get(label_vi, label_vi)

    disease_text = get_disease_info(label_en)
    if not disease_text or not disease_text.strip():
        disease_text = "Hiện tại chưa có thông tin chi tiết cho bệnh này trong dữ liệu nội bộ."

    ai_sections = fetch_ai_disease_sections(label_vi, label_en, confidence, disease_text)

    return {
        "image": image,
        "label_vi": label_vi,
        "label_en": label_en,
        "confidence": confidence,
        "disease_text": disease_text,
        "severity": severity_from_confidence(confidence),
        "recommendation": build_recommendation(label_en, disease_text),
        "ai_sections": ai_sections,
    }


st.set_page_config(
    page_title="FloraGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(90, 165, 94, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(32, 101, 57, 0.12), transparent 20%),
                linear-gradient(180deg, #f7fbf6 0%, #eef5ee 100%);
        }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container{
    max-width: 100%;
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 0rem;
}
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.9rem 1.4rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(18, 38, 24, 0.08);
            margin-bottom: 3rem;
            marin-top: -5rem;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 800;
            color: #14532d;
            font-size: 1.8rem;
            letter-spacing: -0.02em;
        }
        .brand-dot {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            background: linear-gradient(135deg, #7dd38b, #1f7a38);
            display: grid;
            place-items: center;
            color: white;
            font-size: 1.4rem;
            box-shadow: 0 12px 22px rgba(31, 122, 56, 0.25);
        }
        .navlinks {
            display: flex;
            gap: 1.6rem;
            color: #1f2937;
            font-weight: 600;
            flex-wrap: wrap;
            justify-content: center;
        }
        .nav-actions {
            display: flex;
            gap: 0.7rem;
        }
        .nav-btn {
            padding: 0.58rem 1rem;
            border-radius: 12px;
            border: 1px solid #c8d8cc;
            background: white;
            font-weight: 700;
            color: #275d3c;
            box-shadow: 0 5px 12px rgba(16, 24, 40, 0.08);
        }
        .nav-btn.primary {
            background: linear-gradient(135deg, #2f7f49, #2764a7);
            color: white;
            border-color: transparent;
        }
        .hero {
            background:
                linear-gradient(90deg, rgba(12, 58, 27, 0.86), rgba(25, 79, 38, 0.76)),
                radial-gradient(circle at 20% 20%, rgba(124, 211, 139, 0.24), transparent 28%),
                radial-gradient(circle at 80% 30%, rgba(255, 255, 255, 0.08), transparent 20%);
            border-radius: 28px;
            padding: 2.5rem 2rem 3rem;
            color: white;
            position: relative;
            overflow: hidden;
            box-shadow: 0 22px 48px rgba(17, 54, 28, 0.24);
            margin-bottom: -2rem;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 1rem;
            align-items: center;
        }
       
       
        .hero-title {
            font-size: clamp(2rem, 3vw, 3.3rem);
            font-weight: 900;
            margin: 0;
            text-align: center;
            line-height: 1.08;
            letter-spacing: -0.03em;
        }
        .hero-subtitle {
            margin-top: 0.65rem;
            text-align: center;
            font-size: 1.05rem;
            color: rgba(255,255,255,0.86);
        }
        .hero-cta {
            margin-top: 1.5rem;
            display: flex;
            justify-content: center;
        }
        .hero-cta button {
            background: white;
            color: #1f5f37;
            border: none;
            border-radius: 14px;
            padding: 0.9rem 1.4rem;
            font-weight: 800;
            box-shadow: 0 16px 25px rgba(0,0,0,0.15);
        }
        .main-card {
            background: rgba(255,255,255,0.96);
            border-radius: 24px;
            box-shadow: 0 18px 45px rgba(19, 39, 25, 0.12);
            padding: 1.2rem;
            border: 1px solid rgba(172, 198, 176, 0.32);
        }
        .section-title {
            font-size: 1.7rem;
            font-weight: 900;
            color: #111827;
            margin: 0 0 1rem 0;
            letter-spacing: -0.02em;
        }
        .box {
            background: white;
            border-radius: 20px;
            padding: 1rem;
            border: 1px solid rgba(155, 174, 160, 0.35);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.85);
        }
        .upload-box {
            border: 2px dashed #b7c7bc;
            border-radius: 20px;
            padding: 1.2rem 1rem;
            text-align: center;
            background: linear-gradient(180deg, #f9fcf9, #ffffff);
        }
        .upload-icon {
            width: 74px;
            height: 74px;
            border-radius: 999px;
            margin: 0 auto 0.7rem;
            display: grid;
            place-items: center;
            font-size: 2rem;
            color: #2f7f49;
            background: #e8f6ea;
        }
        .upload-note {
            color: #4b5563;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .analysis-image {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid #d9e3db;
        }
        .result-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: #ecf8ef;
            color: #1f7a38;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.9rem;
        }
        .stat-card {
            padding: 0.85rem 0.9rem;
            border-radius: 16px;
            background: #f8faf8;
            border: 1px solid #e3ece4;
        }
        .stat-label { font-size: 0.85rem; color: #6b7280; margin-bottom: 0.25rem; }
        .stat-value { font-size: 1rem; font-weight: 800; color: #111827; }
        .feature-card {
            border-radius: 18px;
            background: white;
            border: 1px solid rgba(177, 195, 181, 0.38);
            box-shadow: 0 8px 20px rgba(17, 54, 28, 0.06);
            padding: 1rem 0.9rem;
            text-align: center;
            font-weight: 800;
            color: #334155;
        }
        .feature-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            margin: 0 auto 0.5rem;
            display: grid;
            place-items: center;
            background: #edf7ef;
            color: #2f7f49;
            font-size: 1.1rem;
        }
        .chat-card {
            background: rgba(255,255,255,0.96);
            border-radius: 24px;
            padding: 1.2rem;
            border: 1px solid rgba(172, 198, 176, 0.32);
            box-shadow: 0 18px 45px rgba(19, 39, 25, 0.12);
        }
        .chat-message {
            padding: 0.8rem 0.95rem;
            border-radius: 16px;
            margin-bottom: 0.75rem;
            line-height: 1.5;
        }
        .chat-user {
            background: #e8f2ff;
            border: 1px solid #cfe1ff;
        }
        .chat-ai {
            background: #f2fbf3;
            border: 1px solid #d6ead9;
        }
        .tabbox {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: #f9fbf9;
            border: 1px solid #e2ebe3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

class_names, vi_map, reverse_vi_map = load_label_maps()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_key" not in st.session_state:
    st.session_state.analysis_key = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

st.markdown(
    """
    <div class="topbar">
        <div class="brand"><div class="brand-dot">🌿</div>FloraGuard AI</div>
    </div>
    <div class="hero">
        <div class="hero-grid">
            <div class="hero-art">
              <img src="banner.jpg" alt="Image plant" style="width:100%; height:100%; object-fit:cover; ">
            </div>
            <div>
                <h1 class="hero-title">PHÁT HIỆN BỆNH LÁ CÂY VỚI CHÍNH XÁC AI</h1>
                <div class="hero-subtitle">Chẩn đoán bệnh thực vật nhanh chóng bằng trí tuệ nhân tạo.</div>
                <div class="hero-cta"><button>BẮT ĐẦU NGAY</button></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">CHẨN ĐOÁN LÁ CÂY</div>', unsafe_allow_html=True)

left_col, center_col, right_col = st.columns([1.05, 1.7, 1.1], gap="large")

with left_col:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="upload-box">
            <div class="upload-icon">☁</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #1f2937;">KÉO & THẢ ẢNH TẠI ĐÂY</div>
            <div class="upload-note">hoặc</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Tải lên lá cây của bạn",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<div style='margin-top: 0.8rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.55rem;'>"
        "<div class='feature-card'><div class='feature-icon'>📷</div>Chụp ảnh</div>"
        "<div class='feature-card'><div class='feature-icon'>🖼️</div>Bộ sưu tập</div>"
        "<div class='feature-card'><div class='feature-icon'>🔗</div>URL Ảnh</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.session_state.analysis_key = None
    st.session_state.analysis_result = None

with center_col:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    if uploaded_file:
        image_key = f"{uploaded_file.name}-{uploaded_file.size}"

        if st.session_state.analysis_key != image_key:
            with st.spinner("Đang phân tích ảnh..."):
                try:
                    analysis_result = analyze_uploaded_image(uploaded_file, reverse_vi_map)
                except Exception as error:
                    st.session_state.analysis_key = None
                    st.session_state.analysis_result = None
                    st.error(f"Không thể phân tích ảnh: {error}")
                    analysis_result = None

            if analysis_result is not None:
                st.session_state.analysis_key = image_key
                st.session_state.analysis_result = analysis_result

        analysis = st.session_state.analysis_result
        if analysis:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                    <div class="result-pill">🟢 {analysis['label_vi']} - {analysis['confidence']*100:.0f}% Tin cậy</div>
                    <div style="color:#6b7280; font-weight:700;">{analysis['confidence']*100:.0f}% HOÀN THÀNH</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(analysis["image"], use_container_width=True, clamp=True)
    else:
        st.markdown(
            """
            <div class="analysis-image" style="padding: 3rem 1rem; text-align:center; background: linear-gradient(180deg, #f7fbf7, #ffffff);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌱</div>
                <div style="font-weight: 800; font-size: 1.05rem; color: #1f2937;">Ảnh phân tích sẽ hiển thị ở đây</div>
                <div style="color:#6b7280; margin-top:0.4rem;">Hãy tải ảnh lá cây để bắt đầu chẩn đoán.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.2rem; font-weight:900; color:#111827; margin-bottom:0.6rem;">KẾT QUẢ PHÂN TÍCH</div>', unsafe_allow_html=True)

    if st.session_state.analysis_result:
        analysis = st.session_state.analysis_result
        ai_sections = analysis["ai_sections"]

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-card"><div class="stat-label">Bệnh chính</div><div class="stat-value">{analysis['label_vi']}</div></div>
                <div class="stat-card"><div class="stat-label">Mức độ nhiễm</div><div class="stat-value">{analysis['severity']}</div></div>
                <div class="stat-card"><div class="stat-label">Độ tin cậy</div><div class="stat-value">{analysis['confidence']*100:.2f}%</div></div>
                <div class="stat-card"><div class="stat-label">Gây ra bởi</div><div class="stat-value">Mô hình AI</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_label = {
            "groq": "Groq API",
            "gemini": "Gemini API",
            "fallback": "fallback nội bộ",
        }.get(ai_sections.get("source", "fallback"), "không xác định")

        st.caption(f"Nguồn nội dung: {source_label}")

        if ai_sections.get("source") == "fallback" and ai_sections.get("error"):
            st.caption(f"Chi tiết lỗi AI: {ai_sections['error']}")

        st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["Triệu chứng", "Nguyên nhân", "Cách điều trị", "Phòng ngừa"])

        symptoms_text = ai_sections.get("symptoms") or analysis.get("disease_text") or "Chưa có dữ liệu triệu chứng."

        with tab1:
            st.markdown(f"<div class='tabbox'>Triệu chứng: {symptoms_text}</div>", unsafe_allow_html=True)
        with tab2:
            st.markdown(f"<div class='tabbox'>Nguyên nhân: {ai_sections.get('cause', 'Chưa có dữ liệu.')}</div>", unsafe_allow_html=True)
        with tab3:
            st.markdown(f"<div class='tabbox'>Cách điều trị: {ai_sections.get('treatment', 'Chưa có dữ liệu.')}</div>", unsafe_allow_html=True)
        with tab4:
            st.markdown(f"<div class='tabbox'>Phòng ngừa: {ai_sections.get('prevention', 'Chưa có dữ liệu.')}</div>", unsafe_allow_html=True)
    else:
        st.info("Tải ảnh lá cây để xem kết quả phân tích chi tiết.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

feature_cols = st.columns(4, gap="medium")
feature_titles = [
    ("HƯỚNG DẪN SỬ DỤNG", "📘"),
    ("XEM LẠI LỊCH SỬ", "⏳"),
    ("THƯ VIỆN BỆNH", "📚"),
    ("CỘNG ĐỒNG", "👥"),
]

for column, (title, icon) in zip(feature_cols, feature_titles):
    with column:
        st.markdown(
            f"""
            <div class="feature-card" style="padding: 1.05rem;">
                <div class="feature-icon">{icon}</div>
                <div>{title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
st.markdown('<div class="chat-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">TRÒ CHUYỆN VỚI AI</div>', unsafe_allow_html=True)

user_input = st.text_input("Nhập câu hỏi hoặc tin nhắn:", placeholder="Ví dụ: Cách xử lý bệnh này như thế nào?")
send_col, _ = st.columns([0.18, 0.82])
with send_col:
    send_clicked = st.button("Gửi", use_container_width=True)

if send_clicked and user_input.strip():
    st.session_state.chat_history.append(("Bạn", user_input.strip()))
    with st.spinner("Đang trả lời..."):
        try:
            response = chat_response(user_input.strip())
        except Exception as error:
            response = f"⚠️ Lỗi: {error}"
    st.session_state.chat_history.append(("AI", response))

for sender, message in st.session_state.chat_history:
    if sender == "Bạn":
        st.markdown(f'<div class="chat-message chat-user"><strong>{sender}:</strong> {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chat-ai"><strong>{sender}:</strong> {message}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)