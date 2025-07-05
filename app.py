# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

# 🌐 Page Config
st.set_page_config(
    page_title="🩺 Medical Diagnosis",
    layout="centered",
    page_icon="🧠"
)

# 🎨 Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .title {
            text-align: center;
            color: #262730;
            font-size: 2.3rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #5a5a5a;
            margin-bottom: 2rem;
        }
        .confidence-box {
            background-color: #eaf4ff;
            border-left: 6px solid #1f77b4;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 6px solid #ffa500;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# 📦 Load Models
@st.cache_resource
def load_models():
    pneumonia_model = tf.keras.models.load_model("models/pneumonia_model2.keras")
    pneumonia_model._name = "pneumonia"
    tumor_model = tf.keras.models.load_model("models/tumor_model2.keras")
    tumor_model._name = "tumor"
    return pneumonia_model, tumor_model

# 🔥 Grad-CAM Utility
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        output = predictions[:, pred_index]
    grads = tape.gradient(output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 🖼️ Display Grad-CAM
def display_gradcam(img, model, last_conv_layer_name="top_conv"):
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(preprocess_input(img_array), axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(img_resized)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, caption="🔍 Grad-CAM Explanation", use_container_width=True)

# 🧠 App Header
st.markdown('<div class="title">Medical Imaging Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect Pneumonia or Brain Tumors with Explainability</div>', unsafe_allow_html=True)

# Sidebar for task
st.sidebar.title("Navigation")
task = st.sidebar.radio("Select Diagnosis Task", ("Pneumonia Detection", "Tumor Detection"))
uploaded_file = st.sidebar.file_uploader("📤 Upload an X-ray or MRI Image", type=["jpg", "jpeg", "png"])

# Load Models
pneumonia_model, tumor_model = load_models()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(preprocess_input(img_array), axis=0)

    # Task-specific logic
    if task == "Pneumonia Detection":
        model = pneumonia_model
        label = "PNEUMONIA"
        no_label = "NORMAL"
    else:
        model = tumor_model
        label = "TUMOR"
        no_label = "NO TUMOR"

    # Prediction
    pred = model.predict(img_array)[0][0]
    confidence = round(pred * 100, 2) if pred > 0.5 else round((1 - pred) * 100, 2)
    prediction = label if pred > 0.5 else no_label

    # Results
    st.markdown(f"### ✅ **Prediction**: `{prediction}`")
    st.markdown(f'<div class="confidence-box">🧪 <b>Confidence:</b> {confidence}%</div>', unsafe_allow_html=True)

    if confidence < 60:
        st.markdown('<div class="warning-box">⚠️ <b>Low confidence</b>: Consider reviewing the scan or consulting a specialist.</div>', unsafe_allow_html=True)

    st.subheader("🧠 Model Explainability")
    display_gradcam(image, model)
