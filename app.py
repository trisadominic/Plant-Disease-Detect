from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
try:
    model = load_model("model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# Print model input shape for debugging
try:
    print("ℹ️ Expected Input Shape:", model.input_shape)
except Exception as e:
    print("❌ Error getting model input shape:", e)

# Define class labels (Modify if needed)
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file exists in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            # Open and process image
            image = Image.open(io.BytesIO(file.read()))
            image = image.convert("RGB")  # Ensure RGB format
            expected_size = model.input_shape[1:3]  # Get (height, width)
            image = image.resize(expected_size)

            # Normalize and reshape for model input
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)  

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400

        # Make prediction
        try:
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction)) * 100

            disease_name = class_names[predicted_class]

            return jsonify({"disease": disease_name, "confidence": f"{confidence:.2f}%"})

        except Exception as e:
            return jsonify({"error": f"Model prediction error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
