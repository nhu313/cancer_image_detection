from flask import Flask, request, jsonify
from utils.cnn_api import CNN

app = Flask(__name__)


@app.route('/api/process_image', methods=['POST'])
def upload_image():
    '''
    processed in utils/neural_net.py
    '''
    file = request.files.get('file')
    category = cnn.predict_image_tensor(file)

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    
   
    return jsonify({'category': category})


@app.route('/')
def home():
    '''
    Is the API running?
    '''
    return jsonify({"message": "Image processing API is running!"})


if __name__ == '__main__':
    # TODO: Where to load model/tensors?
    cnn = CNN(architecture='wide',
              tensors=["utils/image_tensors.pt", "utils/label_tensors.pt"],
              model_path=['utils/model_11_4.pth'])

    app.run(debug=True, host='0.0.0.0', port=5001)
