
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical'
)
class_indices = generator.class_indices  # dict: {'Bacterial Spot': 0, 'Early Blight': 1, ...}
# Đảo ngược thành list theo index
labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    labels[idx] = label
with open('class_names1.json', 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)