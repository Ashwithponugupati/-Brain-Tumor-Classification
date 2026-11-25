from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Classification",
    page_icon="🧠",
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_CANDIDATES = [
    Path(__file__).resolve().parent / "brain_tumor_cnn.keras",
    Path("/content/brain_tumor_cnn.keras"),
]


@st.cache_resource
def load_tumor_model():
    """Load and cache the trained Keras CNN model."""
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return tf.keras.models.load_model(candidate), candidate
    raise FileNotFoundError(
        "Could not find 'brain_tumor_cnn.keras' in the app directory or /content/."
    )


st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI image to classify it into one of four categories.")

try:
    model, model_path = load_tumor_model()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image")

    if st.button("Predict"):
        resized = image.resize((128, 128))
        img_array = np.array(resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        predicted_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_idx]) * 100.0

        st.success(f"Predicted Class: {CLASS_NAMES[predicted_idx]}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")
        for name, prob in zip(CLASS_NAMES, prediction):
            st.progress(float(prob), text=f"{name}: {prob * 100:.2f}%")

        st.warning(
            "This is an experimental AI prediction, not a medical diagnosis."
        )
