import os
from groq import Groq


def chat_response(user_message: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    if not api_key:
        return "Chưa cấu hình GROQ_API_KEY."

    if not user_message.strip():
        return "Bạn chưa nhập nội dung."

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là trợ lý AI. Hãy trả lời ngắn gọn, dễ hiểu, bằng tiếng Việt."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        text = response.choices[0].message.content
        return text.strip() if text else "Không có phản hồi từ AI."

    except Exception as e:
        return f"Lỗi Groq API: {e}"