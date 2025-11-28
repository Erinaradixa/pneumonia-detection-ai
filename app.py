import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
from werkzeug.utils import secure_filename
import webbrowser
import threading

app = Flask(__name__)

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'pneumonia-detection-2024'

# Buat folder yang dibutuhkan
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Load model dengan error handling yang lebih baik
print("🔧 Loading AI model...")
model = None
MODEL_PATHS = [
    'model/model_EFFICIENTNET_final.keras',
    'model/model_EFFICIENTNET_final.h5',
    '../model/model_EFFICIENTNET_final.keras',
    'D:/pneumonia-detection-ai/model/model_EFFICIENTNET_final.keras'
]

for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        try:
            print(f"📂 Trying to load model from: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Model loaded successfully from {model_path}!")
            break
        except Exception as e:
            print(f"❌ Failed to load from {model_path}: {e}")
            continue

if model is None:
    print("⚠️ WARNING: No model loaded! Please check model path.")
    print("📁 Current working directory:", os.getcwd())
    print("📁 Looking for model in:", MODEL_PATHS)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image untuk prediction"""
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image from {image_path}")
        
        print(f"📸 Original image shape: {img.shape}")
        
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        print(f"📐 Resized image shape: {img.shape}")
        
        # Convert color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Expand dimensions
        img_array = np.expand_dims(img, axis=0)
        print(f"📊 Array shape before preprocessing: {img_array.shape}")
        
        # Preprocess untuk EfficientNet
        img_array = preprocess_input(img_array)
        print(f"✅ Preprocessing complete")
        
        return img_array
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise ValueError(f"Image preprocessing failed: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "upload_folder_exists": os.path.exists(UPLOAD_FOLDER)
    })

@app.route('/predict', methods=['POST'])
def predict():
    print("\n" + "="*50)
    print("🔍 Starting prediction process...")
    
    # Check model
    if model is None:
        print("❌ Model not loaded!")
        return render_template('error.html', 
            message="Model AI belum diload. Silakan restart aplikasi."), 500
    
    # Check file
    if 'file' not in request.files:
        print("❌ No file in request")
        return render_template('error.html', 
            message="Tidak ada file yang diupload"), 400
    
    file = request.files['file']
    if file.filename == '':
        print("❌ Empty filename")
        return render_template('error.html', 
            message="Tidak ada file yang dipilih"), 400
    
    if not allowed_file(file.filename):
        print(f"❌ Invalid file type: {file.filename}")
        return render_template('error.html', 
            message="Tipe file tidak didukung. Gunakan JPG, PNG, atau JPEG"), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"💾 Saving file to: {filepath}")
        file.save(filepath)
        
        # Verify file exists
        if not os.path.exists(filepath):
            raise ValueError(f"File was not saved properly: {filepath}")
        
        print(f"✅ File saved successfully")
        
        # Preprocess dan predict
        print("🔄 Preprocessing image...")
        img_array = preprocess_image(filepath)
        
        print("🤖 Running prediction...")
        prediction = model.predict(img_array, verbose=0)
        print(f"📊 Raw prediction: {prediction}")
        
        score = float(prediction[0][0])
        print(f"📈 Score: {score}")
        
        # Interpret results
        if score > 0.5:
            label = "PNEUMONIA"
            confidence = score
            risk_level = "TINGGI" if confidence > 0.8 else "SEDANG"
            recommendation = "Segera konsultasi dengan dokter spesialis paru!"
        else:
            label = "NORMAL"
            confidence = 1 - score
            risk_level = "RENDAH"
            recommendation = "Hasil tampak normal, tetap jaga kesehatan!"
        
        print(f"✅ Prediction complete: {label} ({confidence*100:.2f}%)")
        print("="*50 + "\n")
        
        return render_template('result.html', 
                             filename=filename,
                             prediction=label,
                             confidence=confidence*100,
                             risk_level=risk_level,
                             recommendation=recommendation)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', 
            message=f"Analisis gagal: {str(e)}"), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Pneumonia Detection App...")
    print("="*50)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    print("="*50)
    print("📍 Local URL: http://localhost:5000")
    print("📍 Alternative URL: http://127.0.0.1:5000")
    print("📍 Network URL: http://192.168.1.5:5000")
    print("⏹️  Press CTRL+C to stop")
    print("="*50 + "\n")
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)