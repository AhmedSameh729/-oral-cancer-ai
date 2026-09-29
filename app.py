import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("oral_cancer_model.keras")

# Page configuration
st.set_page_config(
    page_title="OralCancer AI",
    page_icon="🦷",
    layout="centered"
)

# Title
st.title("🦷 OralCancer AI")
st.write("AI-assisted oral image classification")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an oral image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    with st.spinner("Analyzing image..."):
        prediction = model.predict(image_array, verbose=0)[0][0]

    # Class 0 = Oral Cancer
    # Class 1 = Normal
    if prediction < 0.5:
        label = "Oral Cancer"
        confidence = 1 - prediction
    else:
        label = "Normal"
        confidence = prediction

    # Display result
    st.subheader("Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    st.warning(
        "This AI system is a research prototype and is not a substitute "
        "for professional medical diagnosis."
    )
