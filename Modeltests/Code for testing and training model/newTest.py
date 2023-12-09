import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf

model_path = '1.tflite'

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, target_size=(224, 224)):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = img / 255.0
    img = tf.reshape(img, (1, target_size[0], target_size[1], 3))
    return img

def predict_with_tflite(interpreter, image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123'

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
