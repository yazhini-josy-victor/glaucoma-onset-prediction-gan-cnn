import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "models/baseline_refuge2_cnn.h5"
IMG_SIZE = 224
THRESHOLD = 0.75
# ----------------------------------------

st.set_page_config(
    page_title="Glaucoma Screening System",
    layout="centered"
)

st.title("🩺 Glaucoma Screening System")
st.subheader("AI-assisted Early Detection (Decision Support Tool)")

st.markdown(
"""
**Disclaimer:**  
This system is intended for screening support only.  
Final diagnosis must be made by a qualified ophthalmologist.
"""
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader(
    "Upload Retinal Fundus Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fundus Image", use_column_width=True)

    processed_img = preprocess_image(image)

    if st.button("Run Screening"):
        prob = model.predict(processed_img)[0][0]

        if prob >= 0.75:
            prediction = "🔴 High Risk Glaucoma"
            decision = "Urgent Referral to Ophthalmologist"

        elif 0.45 <= prob < 0.75:
            prediction = "🟠 Suspected Glaucoma"
            decision = "Schedule Specialist Review"

        else:
            prediction = "🟢 Normal"
            decision = "Routine Checkup"

        st.markdown("### 🧠 Screening Result")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence Score:** {prob:.2f}")
        st.write(f"**Clinical Decision:** {decision}")
