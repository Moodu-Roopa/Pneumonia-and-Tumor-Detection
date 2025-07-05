# 🩺 Medical Imaging Diagnosis with Explainability

A deep learning-powered diagnostic tool that detects **Pneumonia** from chest X-rays and **Brain Tumors** from MRI scans. Built with a user-friendly **Streamlit interface**, the application also includes **Grad-CAM** visualizations to interpret predictions and highlight regions of medical concern.

---

## 🚀 Key Features

- 📂 Upload Chest X-ray or Brain MRI images
- 🧠 Predict medical conditions:
  - ✅ **Pneumonia Detection**
  - ✅ **Brain Tumor Detection**
- 🎯 **Model Confidence Score** for reliability
- 🔍 Visual explanations via **Grad-CAM** overlays
- 🖥️ Interactive **Streamlit Web App** for real-time diagnosis

---

## 🧠 Models & Architecture

- 🧬 **Model Backbone**: [EfficientNetV2S](https://arxiv.org/abs/2104.00298) (pretrained & fine-tuned)
- 🎓 **Classification**: Custom binary classifiers for:
  - `Pneumonia vs Normal` (Chest X-rays)
  - `Tumor vs No Tumor` (Brain MRIs)
- 🔥 **Explainability**: Grad-CAM implemented to highlight decisive regions in the input images
- 📏 **Input Shape**: All images are resized to 224×224

---

## 📊 Example Outputs

✅ **Prediction**: Pneumonia Detected  

📈 **Confidence**: 98.76%  

📍 **Grad-CAM Overlay**: Highlights infected regions in X-ray or MRI

## 🖥️ Run the App Locally

### 1. Create a virtual environment (optional but recommended)

python -m venv venv

source venv/bin/activate  

For Windows: venv\Scripts\activate

2. Install the dependencies

pip install -r requirements.txt

3. Launch the Streamlit app

streamlit run app.py

📦 **Datasets Used**

📌 Pneumonia Detection Dataset:

📁 Source: Chest X-ray dataset (Kaggle)

📊 Classes: Pneumonia, Normal

📥 Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

📌 Brain Tumor Detection Dataset:

📁 Source: Brain MRI images (Kaggle)

📊 Classes: Tumor, No Tumor

📥 Link: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

⚙️ **Tech Stack**

| Technology   | Version |
| ------------ | ------- |
| Python       | ≥ 3.8   |
| TensorFlow   | ≥ 2.13  |
| Keras        | ≥ 2.15  |
| Streamlit    | ≥ 1.34  |
| OpenCV       | ≥ 4.8   |
| Pillow (PIL) | ≥ 10.0  |
| Matplotlib   | ≥ 3.8   |
| Scikit-learn | ≥ 1.4   |

📬 **Contact**

Developer: Moodu Roopa

📧 Email: moodroopa1169@gmail.com

🌐 GitHub: Moodu-Roopa