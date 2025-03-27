
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Set dataset directory
DATASET_PATH = "D:\\demo\\small_dataset"  # Adjust if your folder name is different

# ✅ Parameters
IMG_SIZE = (224, 224)  # Image size (adjust based on your model)
BATCH_SIZE = 32
EPOCHS = 10  # Increase for better accuracy

# ✅ Image Data Generator (Augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    validation_split=0.2  # 80% train, 20% validation
)

# ✅ Load Training Data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# ✅ Load Validation Data
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ✅ Get class names
class_names = list(train_generator.class_indices.keys())
print("✅ Classes:", class_names)

# ✅ Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')  # Output layer
])

# ✅ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train Model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# ✅ Save Model
model.save("model.h5")
print("🎉 Model training complete! Saved as model.h5")
