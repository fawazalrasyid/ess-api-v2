from datetime import datetime
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

try:
    interpreter = tf.lite.Interpreter(model_path="model/modelVGG.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_image_class(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(predictions)
        if predicted_class == 0:
            return "0-25 ton/ha"
        elif predicted_class == 1:
            return "25-50 ton/ha"
        else:
            return "50 ton++"
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Generate a new filename based on the current datetime, including milliseconds
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{timestamp}{file_extension}"
        file_path = os.path.join('uploads', new_filename)

        file.save(file_path)
        
        result = predict_image_class(file_path)
        # os.remove(file_path)
        
        if result:
            return jsonify({
                'message': 'Image predicted',
                'predictedClass': result
            }), 200
        else:
            return jsonify({
                'message': 'Prediction failed'
            }), 500
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({
            'message': 'Internal Server Error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)