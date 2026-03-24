info_dict = {
    "Potato___Late_blight": "Bệnh sương mai gây ra bởi nấm Phytophthora infestans. Diệt nấm bằng thuốc gốc đồng và giảm độ ẩm.",
    "Potato___Early_blight": "Bệnh đốm lá sớm do Alternaria. Cần cắt bỏ lá bệnh, sử dụng thuốc nấm đặc trị.",
    "Tomato_Bacterial_spot": "Bệnh đốm vi khuẩn. Nên sử dụng hạt giống sạch và tránh tưới lên lá.",
    "Tomato_healthy": "Cây cà chua khỏe mạnh. Tiếp tục chăm sóc như hiện tại là tốt.",
    "Tomato_Late_blight": "Bệnh sương mai. Cần giảm độ ẩm, sử dụng thuốc nấm đặc trị.",
    "Tomato_Leaf_Mold": "Bệnh mốc lá. Cần giảm độ ẩm, sử dụng thuốc nấm đặc trị.",
    "Tomato_Septoria_leaf_spot": "Bệnh đốm lá do Septoria. Cần cắt bỏ lá bệnh, sử dụng thuốc nấm đặc trị.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Bệnh mites. Cần sử dụng thuốc trừ sâu đặc trị.",
    "Tomato_Target_Spot": "Bệnh đốm trúng. Cần sử dụng thuốc nấm đặc trị.",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": "Bệnh virus vàng lá. Cần sử dụng thuốc đặc trị và cắt bỏ lá bệnh.",
    "Tomato_Tomato_mosaic_virus": "Bệnh virus mosa. Cần sử dụng thuốc đặc trị và cắt bỏ lá bệnh."
}
def get_disease_info(label):
    return info_dict.get(label, "Hiện tại chưa có thông tin chi tiết cho bệnh này. Vui lòng tham khảo chuyên gia nông nghiệp.")
