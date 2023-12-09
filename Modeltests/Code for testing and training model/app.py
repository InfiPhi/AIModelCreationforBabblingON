import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
import pandas as pd

# Load the model
model = load_model('ASL.h5')
print("Model loaded successfully...")

# Define the imageSize as used during training
imageSize = 64

# Helper function to preprocess the image
def preprocess_image(image_path, imageSize):
    img = cv2.imread(image_path)
    img = resize(img, (imageSize, imageSize, 3))
    img = np.asarray(img)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)  # Model expects batches, so we add a batch dimension
    return img

# Mapping from the image name to class indices
class_mapping = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'NOTHING': 27, 'SPACE': 28
}

# Directory where the test images are located
test_dir = 'test/asl_alphabet_test'

# Initialize variables to store test images and labels
test_images = []
test_labels = []
image_name_to_actual_class = {}

# Load and preprocess each image in the test directory
for image_name in sorted(os.listdir(test_dir)):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
        image_path = os.path.join(test_dir, image_name)
        image = preprocess_image(image_path, imageSize)
        test_images.append(image)
        
        label_name = image_name.split('_')[0].upper()
        label = class_mapping.get(label_name, 29)  # Default to 29 if label_name is not in class_mapping
        test_labels.append(label)
        
        # Store the mapping from image name to actual class
        image_name_to_actual_class[image_name] = label

# Check if there are any images to evaluate
if test_images and test_labels:
    # Convert lists to numpy arrays
    test_images = np.vstack(test_images)
    test_labels = np.array(test_labels)

    # Make predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Convert labels to one-hot encoding to compare with predictions
    test_labels_one_hot = to_categorical(test_labels, num_classes=29)

    # Evaluate the model
    score = model.evaluate(test_images, test_labels_one_hot, verbose=0)
    print('Test accuracy:', score[1])

    # Print the classification report
    from sklearn.metrics import classification_report
    print(classification_report(test_labels, predicted_classes))

    # Create a DataFrame to store results
    results = []

    for i, prediction in enumerate(predicted_classes):
        test_image_name = sorted(os.listdir(test_dir))[i]
        if test_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            actual_label_name = test_image_name.split('_')[0].upper()
            actual_label = class_mapping.get(actual_label_name, 'Unknown')
            predicted_label_name = list(class_mapping.keys())[list(class_mapping.values()).index(prediction)]
            
            results.append({
                'Image': test_image_name,
                'Predicted': predicted_label_name,
                'Actual': actual_label_name
            })

    # Create a DataFrame from the results list and print it
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Optionally, save the DataFrame to a CSV file
    results_df.to_csv('prediction_results.csv', index=False)

else:
    print("No valid test images or labels found for evaluation.")

print("Testing completed...")
