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

# Load model
print("🔧 Loading AI model...")
try:
    model = tf.keras.models.load_model('model_EFFICIENTNET_final.keras', compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image untuk prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('error.html', message="Model AI belum diload"), 500
    
    if 'file' not in request.files:
        return render_template('error.html', message="Tidak ada file yang diupload"), 400
    
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="Tidak ada file yang dipilih"), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess dan predict
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array, verbose=0)
            score = float(prediction[0][0])
            
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
            
            return render_template('result.html', 
                                 filename=filename,
                                 prediction=label,
                                 confidence=confidence*100,
                                 risk_level=risk_level,
                                 recommendation=recommendation)
            
        except Exception as e:
            return render_template('error.html', message=f"Analisis gagal: {str(e)}"), 500
    
    return render_template('error.html', message="Tipe file tidak didukung. Gunakan JPG, PNG, atau JPEG"), 400

def open_browser():
    """Auto buka browser ketika server ready"""
    import time
    time.sleep(5)
    webbrowser.open("http://localhost:5000")

if __name__ == '__main__':
    # Create upload folder
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    print("🚀 Starting Pneumonia Detection App...")
    print("📍 Local URL: http://localhost:5000")
    print("📍 Alternative URL: http://127.0.0.1:5000")
    print("⏹️  Press CTRL+C to stop")
    
    # Auto buka browser
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
