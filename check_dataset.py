from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the same preprocessing as training
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_directory(
    "small_dataset",  # Change this to your dataset path
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("📌 Model Class Order:", train_generator.class_indices)
