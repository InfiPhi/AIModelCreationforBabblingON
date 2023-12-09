import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set the path to your HDF5 model
model_path = 'resnet50v2_30_ep.hdf5'  # Make sure this is the correct path

# Load the model
model = load_model(model_path)

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img = image.img_to_array(img)

        # Normalize pixel values to [0, 1] as was done during training
        img = img / 255.0

        # Reshape the image to include a batch dimension
        img = np.expand_dims(img, axis=0)

        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_with_model(model, img):
    prediction = model.predict(img)
    return prediction

# The class mapping should be the same as the index of the classes used during training
class_mapping = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1'

# Set the path to your test folder
test_folder_path = 'testing/'  # Update this path to the folder where your test images are located
results = []

# Iterate over images in the test folder
for image_file in os.listdir(test_folder_path):
    # Make sure to process files with the .jpeg extension as in your original code
    if image_file.lower().endswith('.jpeg'):
        actual_label = image_file[0].upper()  # The first character of the filename is the label

        image_path = os.path.join(test_folder_path, image_file)
        img = preprocess_image(image_path)
        if img is not None:
            prediction = predict_with_model(model, img)
            predicted_class_index = np.argmax(prediction, axis=1)
            predicted_label = class_mapping[predicted_class_index[0]]

            results.append([image_file, actual_label, predicted_label])

# Calculate accuracy
actual_labels = [r[1] for r in results]
predicted_labels = [r[2] for r in results]
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save results to CSV
results_df = pd.DataFrame(results, columns=['Image', 'Actual', 'Predicted'])
csv_path = 'test_results.csv'  # Update this path if you want to save the CSV elsewhere
results_df.to_csv(csv_path, index=False)
print(f"Results have been saved to {csv_path}.")
