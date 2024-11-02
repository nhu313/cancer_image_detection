from flask import Flask, request, jsonify
# from utils.neural_net import process_image

app = Flask(__name__)


@app.route('/api/process_image', methods=['POST'])
def upload_image():
    '''
    processed in utils/neural_net.py
    '''
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Process the image using the neural network and return the result
    # category = process_image(file)
    # return jsonify({'category': category})
    return jsonify({'message': 'processed'})


@app.route('/')
def home():
    '''
    Is the API running?
    '''
    return jsonify({"message": "Image processing API is running!"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
