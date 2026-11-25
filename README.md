# Brain Tumor Classification 🧠

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify brain MRI scans into 4 categories:

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `notumor` | No tumor detected |
| `pituitary` | Pituitary tumor |

## 📊 Model Performance
- **Test Accuracy:** ~82%
- **Architecture:** 3× Conv2D + BatchNorm + MaxPool → Dense(256) + Dropout(0.5) → Softmax(4)
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 5,600 training / 1,600 test images

## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashwithponugupati/-Brain-Tumor-Classification/blob/main/Brain_Tumor_Classification.ipynb)

## 📁 Project Structure
```
brain-tumor-classification/
├── Brain_Tumor_Classification.ipynb   # Full Colab notebook
├── app.py                             # Streamlit web app
├── requirements.txt                   # Dependencies
└── README.md
```

## 🖥️ Run the Streamlit App Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer
This is an experimental AI project for educational purposes only. **Not a medical diagnosis tool.**
