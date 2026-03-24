from tensorflow.keras.models import load_model
import numpy as np

# Load mô hình đã lưu
model = load_model("model/model.h5")

# Xem kiến trúc mô hình
model.summary()

# Chọn một lớp (ví dụ lớp cuối cùng)
dense_layer = model.layers[-1]  # Lấy lớp cuối cùng (thường là lớp Softmax)

# Lấy trọng số và bias
weights, biases = dense_layer.get_weights()

# In thông tin
print("🔢 Shape của weights:", weights.shape)   # (số_feature, số_lớp)
print("🔢 Shape của biases:", biases.shape)     # (số_lớp,)

# In một phần nhỏ của trọng số
print("\n🎯 Một vài giá trị đầu tiên của trọng số:")
print(weights[:5])  # In 5 dòng đầu tiên

print("\n🎯 Một vài giá trị của bias:")
print(biases[:5])
