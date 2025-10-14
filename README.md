---
title: IC Counterfeit Detection System
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🔬 IC Counterfeit Detection System

Advanced AI-powered system for detecting counterfeit integrated circuits (ICs) using computer vision and deep learning.

## 🌟 Features

- **11-Layer Verification System**
  - Logo Detection (Template Matching)
  - AI Agent OEM Verification
  - Text & Serial Number OCR
  - QR/DMC Code Detection
  - Surface Defect Detection
  - Edge Detection (Canny)
  - IC Outline/Geometry Analysis
  - Angle Detection
  - Color Surface Verification
  - Texture Verification
  - Font Verification

- **AI-Powered Analysis**
  - Hugging Face BLIP Vision-Language Model
  - Automatic part number extraction
  - Manufacturer verification
  - Confidence scoring

- **Professional UI**
  - Modern, responsive design
  - Real-time results
  - Detailed test breakdowns
  - JSON export for API integration

## 🚀 How to Use

1. Upload a clear image of the IC chip you want to verify
2. Optionally upload a reference (genuine) IC image
3. Click "Start Verification" to begin analysis
4. View comprehensive results with confidence scores

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, scikit-image
- **OCR**: Tesseract, EasyOCR
- **AI Models**: Hugging Face Transformers, BLIP
- **Web Interface**: Gradio
- **Image Processing**: PIL, NumPy

## 📊 Verification Layers

Each IC undergoes 11 different tests:
- Visual inspection (logo, text, geometry)
- AI-powered OEM verification
- Surface quality analysis
- Code detection (QR, DMC)
- Font and texture verification

## ⚡ Performance

- Processing time: ~10-15 seconds per image (CPU)
- Accuracy: High confidence scoring system
- Real-time feedback with detailed breakdowns

## 🔒 Privacy

All processing is done on the server. Images are not stored permanently.

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

For questions or support, please open an issue on the repository.
