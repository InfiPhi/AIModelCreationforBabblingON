import numpy as np
import cv2
from tensorflow.keras.models import load_model
from skimage.transform import resize

# Load the model
model = load_model('asl_predictor.h5')
print("Model loaded successfully...")

# Define the imageSize as used during training
imageSize = 64

# Helper function to preprocess the image
def preprocess_image(image_path, imageSize):
    img = cv2.imread(image_path)
    img = resize(img, (imageSize, imageSize, 3), anti_aliasing=True)
    img = np.asarray(img)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)  # Model expects batches, so we add a batch dimension
    return img

# Mapping from class indices to class labels
class_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 
    27: 'NOTHING', 28: 'SPACE'
}

# Path to the image you want to predict
image_path = 'testimage1.jpeg'

# Preprocess the image
processed_image = preprocess_image(image_path, imageSize)

# Make prediction
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction, axis=1)

# Get the corresponding label from the class_mapping
predicted_label = class_mapping.get(predicted_class[0], "Unknown")

print(f"The predicted class for '{image_path}' is: {predicted_label}")
