from nlp.disease_info import get_disease_info
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa_pairs = {
    "Làm sao để trị bệnh cháy lá?": "Bạn nên cắt bỏ lá bị bệnh, tránh tưới nước quá nhiều và sử dụng thuốc sinh học phù hợp.",
    "Cây cà chua bị vàng lá là bệnh gì?": "Đó có thể là bệnh do virus xoăn vàng lá gây ra. Bạn nên cách ly cây bệnh và diệt trừ bọ phấn.",
    "Cách nhận biết bệnh sương mai?": "Bệnh sương mai thường gây đốm màu nâu hoặc xám trên lá, đặc biệt vào thời tiết ẩm ướt."
}

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(list(qa_pairs.keys()))


def get_bot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(user_vec, x)
    best_match = sim.argmax()
    if sim[0][best_match] < 0.2:
        return "Xin lỗi, tôi chưa hiểu rõ câu hỏi. Bạn có thể hỏi về các bệnh trên cây trồng hoặc gửi ảnh để tôi kiểm tra."
    question = list(qa_pairs.keys())[best_match]
    return qa_pairs[question]
