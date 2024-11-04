# app.py (or main.py if that's what you're using)
from flask import Flask, request, jsonify
from app.utils.cnn_api import CNN

app = Flask(__name__)

# Initialize the CNN model
cnn = CNN(architecture='wide',
           tensors=["app/utils/data/image_tensors.pt", "app/utils/data/label_tensors.pt"],
           model_path='app/utils/data/model_11_4.pth')

@app.route('/api/process_image', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    category = cnn.predict_image_tensor(file)
    return jsonify({'category': category})

@app.route('/')
def home():
    return jsonify({"message": "Image processing API is running!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
