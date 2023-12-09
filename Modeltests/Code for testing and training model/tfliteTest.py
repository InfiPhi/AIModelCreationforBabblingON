import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
#from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model10.tflite')
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
        img = np.reshape(img, (1, 224, 224, 3))  # Reshape to (1, 64, 64, 3)
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

test_folder_path = 'test'
results = []
actual_labels = []
predicted_labels = []

for image_file in os.listdir(test_folder_path):
    if image_file.lower().endswith('.jpg'):
        actual_label = image_file.split('_')[0].upper()

        image_path = os.path.join(test_folder_path, image_file)
        processed_image = preprocess_image(image_path)
        if processed_image is not None:
            prediction = predict_with_tflite(interpreter, processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)
            predicted_label = class_mapping[predicted_class_index[0]]

            results.append([image_file, actual_label, predicted_label])
            actual_labels.append(actual_label)
            predicted_labels.append(predicted_label)

accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

results_df = pd.DataFrame(results, columns=['Image', 'Actual', 'Predicted'])
results_df.to_csv('test_results.csv', index=False)
print("Results have been saved to test_results.csv.")
