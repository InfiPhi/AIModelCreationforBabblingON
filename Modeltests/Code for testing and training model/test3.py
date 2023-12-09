import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    smoothed_img = cv2.medianBlur(img, 5)
    _, thresh_img = cv2.threshold(smoothed_img, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized_img = cv2.resize(thresh_img, (28, 28))
    normalized_img = resized_img / 255.0
    img_array = np.reshape(normalized_img, (1, 28, 28, 1))
    return img_array
    
model = load_model('asl_model.h5')
print("Model loaded successfully.")

test_folder_path = 'testing'

class_mapping = 'ABCDEFGHIKLMNOPQRSTUVWXY'

results = []
actual_labels = []
predicted_labels = []

for image_file in os.listdir(test_folder_path):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')): 
        actual_label = image_file.split('_')[0].upper()
        image_path = os.path.join(test_folder_path, image_file)
        processed_image = preprocess_image(image_path)
        if processed_image is not None:
            prediction = model.predict(processed_image)
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
