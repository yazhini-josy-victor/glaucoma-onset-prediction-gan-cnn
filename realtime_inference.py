import tensorflow as tf
import numpy as np
import cv2
import os

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/baseline_refuge2_cnn.h5"
IMAGE_PATH = r"C:\Users\Admin\OneDrive\Desktop\Glaucoma_GAN_project\Phase_3_CNN_Dataset\train\glaucoma\EyePACS-DEV-RG-1.jpg"
 # one image at a time
IMG_SIZE = 224
THRESHOLD = 0.75

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# REAL-TIME INFERENCE
# -------------------------------
image = preprocess_image(IMAGE_PATH)

prediction = model.predict(image)[0][0]

# -------------------------------
# DECISION LOGIC
# -------------------------------
if prediction >= THRESHOLD:
    result = "Suspected Glaucoma"
    referral = "Refer to Ophthalmologist"
else:
    result = "Normal"
    referral = "No Immediate Referral"

# -------------------------------
# OUTPUT
# -------------------------------
print("🩺 Screening Result")
print("----------------------------")
print(f"Prediction          : {result}")
print(f"Confidence Score    : {prediction:.2f}")
print(f"Clinical Decision   : {referral}")
