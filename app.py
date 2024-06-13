from flask import Flask, render_template, request, jsonify
from utility import predict_denomination
import os


app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/uploadImage', methods=['POST'])
def image_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join("image", file.filename)
    file.save("image/" + file.filename)
    denomination = predict_denomination(file_path)
    print(denomination)
    os.remove(file_path)

    return jsonify({'message': 'File uploaded successfully','denomination':denomination}), 200

if __name__ == '__main__':
    app.run(debug=True)
