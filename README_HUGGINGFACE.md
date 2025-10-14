# IC Counterfeit Detection System

An AI-powered system for detecting counterfeit integrated circuits using computer vision and Hugging Face BLIP vision-language model.

## Features

- 🤖 **AI-Powered Analysis**: Uses Salesforce BLIP model for intelligent IC verification
- 🔍 **11-Layer Verification**: Comprehensive testing including logo detection, OCR, defect detection
- 📊 **Real-time Results**: Get instant verification with confidence scores
- 🌐 **Web Interface**: Easy-to-use Gradio interface
- 📱 **API Ready**: JSON output for integration with other systems

## How It Works

1. **Upload IC Image**: Provide an image of the IC chip to verify
2. **AI Analysis**: BLIP model analyzes the image and extracts information
3. **Multi-Layer Verification**: 11 different tests are performed
4. **Get Results**: Receive verdict with detailed confidence scores

## Technology Stack

- **AI Model**: Salesforce/blip-image-captioning-large
- **Framework**: Hugging Face Transformers, PyTorch
- **Computer Vision**: OpenCV, scikit-image
- **OCR**: Tesseract
- **Interface**: Gradio

## Usage

Simply upload an IC chip image and click "Verify IC". The system will:
- Extract part number and manufacturer
- Detect logos and markings
- Analyze surface defects
- Verify geometry and fonts
- Provide overall authenticity verdict

## Processing Time

- **CPU**: 10-15 seconds per image
- **GPU**: 2-3 seconds per image (if available)

## Verification Layers

1. Logo Detection
2. AI Agent OEM Verification
3. Text & Serial Number OCR
4. QR/DMC Code Detection
5. Surface Defect Detection
6. Edge Detection
7. IC Outline/Geometry
8. Angle Detection
9. Color Surface Verification
10. Texture Verification
11. Font Verification

## License

MIT License
