import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='1.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    try:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0  # Normalize pixel values
        img = np.reshape(img, (1, 224, 224, 3))  # Reshape to (1, 224, 224, 3)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_with_tflite(interpreter, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Image to test
image_path = 'image.jpg'
processed_image = preprocess_image(image_path)
if processed_image is not None:
    prediction = predict_with_tflite(interpreter, processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_label = class_mapping[predicted_class_index[0]]

    print(f"Image: {image_path}, Predicted Label: {predicted_label}")
