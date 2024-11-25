# app.py (or main.py if that's what you're using)
from flask import Flask, request, jsonify
from app.utils.cnn_api import CNN
import os
from flask_cors import CORS

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['UPLOAD'] = UPLOAD_FOLDER

# Initialize the CNN model
cnn = CNN(architecture='deep-wide',
           tensors=["app/utils/data/image_tensors.pt", "app/utils/data/label_tensors.pt"],
           model_path='app/utils/data/model_11_15_dw.pth')

@app.route('/api/process_image', methods=['POST'])
def upload_image():
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = os.path.join(app.config['UPLOAD'], file.filename)
    file.save(file_path)

    category = cnn.predict_image(file_path)
    return jsonify({'category': category, 'file_path': file_path})

@app.route('/')
def home():
    return jsonify({"message": "Image processing API is running!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
