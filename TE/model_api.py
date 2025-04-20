from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np

app = Flask(__name__)

# Load model once during startup
model = load_model('dyslexia_model.h5')


# Define your preprocessing logic here

def preprocess_input(data):
    # Example: reshape, scale, etc.
    return np.array(data).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')  # your frontend HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']  # input sent as JSON
        processed = preprocess_input(data)
        prediction = model.predict(processed)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
