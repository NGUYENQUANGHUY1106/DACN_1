import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# --- BƯỚC 1: Khởi tạo thông số ---
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 25
DATASET_DIR = "dataset"
# IMAGE_SIZE = 224: Kích thước ảnh đầu vào là 224x224 (MobileNetV2 yêu cầu vậy).

# BATCH_SIZE = 64: Mỗi lần huấn luyện sẽ xử lý 64 ảnh cùng lúc (giúp tăng tốc, tiết kiệm RAM).

# EPOCHS = 25: Duyệt qua toàn bộ dữ liệu 25 lần để mô hình học đủ.

# DATASET_DIR: Thư mục chứa ảnh chia theo folder lớp bệnh


# --- BƯỚC 2: Tạo dữ liệu huấn luyện và kiểm tra ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)

# --- BƯỚC 3: Tải mô hình MobileNetV2 (pretrained) ---
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE,3))

# Freeze layers
base_model.trainable = False

# --- BƯỚC 4: Tạo mô hình mới ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# --- BƯỚC 5: Huấn luyện mô hình ---
checkpoint = ModelCheckpoint("model/plant_disease_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# --- BƯỚC 6: Vẽ biểu đồ ---
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
