import numpy as np
import json

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    _TF_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    load_model = None
    image = None
    _TF_IMPORT_ERROR = exc

model = None

# Load danh sách nhãn tiếng Anh
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

# Load bản dịch sang tiếng Việt
with open("model/class_labels_vi.json", "r", encoding="utf-8") as f:
    label_translations = json.load(f)

def predict_disease(img):
    global model

    if _TF_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Thiếu thư viện TensorFlow. Hãy cài bằng lệnh: pip install tensorflow"
        ) from _TF_IMPORT_ERROR

    if model is None:
        model = load_model("model/model.h5")

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]

    label_en = class_names[top_idx]
    label_vi = label_translations.get(label_en, label_en)  # fallback nếu không có bản dịch

    return label_vi, confidence
