# Setup Guide - Counterfeit Detection AI Agent

## Quick Start Guide

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Tesseract OCR

#### Windows
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.3.0.exe`)
3. During installation, note the installation path (default: `C:\Program Files\Tesseract-OCR`)
4. Add Tesseract to your PATH:
   - Right-click "This PC" → Properties → Advanced System Settings
   - Click "Environment Variables"
   - Under "System Variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\Program Files\Tesseract-OCR`
   - Click OK on all dialogs

**Alternative**: If you don't want to add to PATH, update the code:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

#### macOS
```bash
brew install tesseract
```

### Step 3: Verify Installation

```bash
python -c "import cv2, pytesseract, pyzbar; print('All dependencies installed!')"
```

### Step 4: Run Test Suite

```bash
python test_agent.py
```

This will:
- Create synthetic test images
- Run all detection modules
- Generate test reports
- Verify the agent is working correctly

### Step 5: Try Example Usage

```bash
python example_usage.py
```

## Using with Real Images

### Prepare Reference Data

1. **Create reference directory structure:**
```
AI_PROect/
├── reference/
│   ├── genuine_logo.png      # Clear image of genuine logo
│   ├── golden_product.png    # Reference image of genuine product
│   └── ...
└── test_images/
    ├── product1.jpg          # Products to verify
    ├── product2.jpg
    └── ...
```

2. **Capture reference images:**
   - **Logo**: High-resolution, clear image of the genuine logo
   - **Golden Product**: Well-lit, straight-on photo of a genuine product
   - **Test Images**: Photos of products you want to verify

### Image Quality Guidelines

#### Logo Template
- ✅ High resolution (at least 200x200 pixels)
- ✅ Clear, sharp image
- ✅ Good contrast
- ✅ Isolated logo (minimal background)

#### Golden Reference Image
- ✅ Good lighting (no shadows)
- ✅ Straight-on view (not tilted)
- ✅ High resolution
- ✅ All features visible

#### Test Images
- ✅ Similar lighting to golden image
- ✅ Similar angle/perspective
- ✅ Clear, not blurry
- ✅ Entire product visible

## Configuration

### Basic Configuration

Edit `config.json` to customize:

```json
{
  "thresholds": {
    "logo_match": 0.7,        // Logo similarity threshold (0-1)
    "ssim": 0.85,             // Structural similarity threshold (0-1)
    "color_tolerance": 15,    // Color difference tolerance (0-255)
    "angle_tolerance": 5.0    // Angle deviation tolerance (degrees)
  }
}
```

### Adjusting Sensitivity

**More Strict (fewer false positives):**
```json
{
  "thresholds": {
    "logo_match": 0.85,
    "ssim": 0.90,
    "color_tolerance": 10,
    "angle_tolerance": 3.0
  }
}
```

**More Lenient (fewer false negatives):**
```json
{
  "thresholds": {
    "logo_match": 0.6,
    "ssim": 0.75,
    "color_tolerance": 25,
    "angle_tolerance": 10.0
  }
}
```

## Common Issues & Solutions

### Issue 1: Tesseract Not Found

**Error:** `TesseractNotFoundError`

**Solution:**
```python
# Add to the top of your script
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Issue 2: QR Code Not Detected

**Possible causes:**
- QR code is too small
- Poor image quality
- Low contrast

**Solutions:**
- Increase image resolution
- Improve lighting
- Ensure QR code is clearly visible
- Try preprocessing: `cv2.threshold()` or `cv2.GaussianBlur()`

### Issue 3: Low Logo Match Score

**Possible causes:**
- Logo size mismatch
- Different perspective/angle
- Logo partially obscured

**Solutions:**
- Ensure template logo is similar size to test image logo
- Use multiple template sizes
- Lower the `logo_match_threshold`
- Crop logo region before matching

### Issue 4: OCR Not Reading Text

**Possible causes:**
- Text is too small
- Poor contrast
- Unusual font

**Solutions:**
- Increase image resolution
- Preprocess image (threshold, denoise)
- Use `--psm` parameter in Tesseract:
  ```python
  pytesseract.image_to_string(image, config='--psm 7')  # Single line
  pytesseract.image_to_string(image, config='--psm 6')  # Uniform block
  ```

### Issue 5: High False Positive Rate

**Solution:** Increase thresholds
```python
agent.logo_match_threshold = 0.85
agent.ssim_threshold = 0.90
```

### Issue 6: High False Negative Rate

**Solution:** Decrease thresholds
```python
agent.logo_match_threshold = 0.6
agent.ssim_threshold = 0.75
```

## Advanced Usage

### Custom Detection Pipeline

```python
from counterfeit_detection_agent import CounterfeitDetectionAgent

agent = CounterfeitDetectionAgent()

# Load image
import cv2
image = cv2.imread('test.jpg')

# Run individual checks
logo_result = agent.detect_logo(image, 'reference/logo.png')
text_result = agent.detect_and_read_text(image, 'EXPECTED_TEXT')
qr_result = agent.detect_qr_code(image, 'EXPECTED_QR_DATA')

# Check results
if logo_result.status == "PASS":
    print("Logo verified!")
```

### Batch Processing

```python
import os
from counterfeit_detection_agent import CounterfeitDetectionAgent

agent = CounterfeitDetectionAgent()
reference_data = {...}

for filename in os.listdir('test_images'):
    if filename.endswith(('.jpg', '.png')):
        image_path = os.path.join('test_images', filename)
        report = agent.process_image(image_path, reference_data)
        print(f"{filename}: {report['verdict']}")
```

### Integration with Web API

```python
from flask import Flask, request, jsonify
from counterfeit_detection_agent import CounterfeitDetectionAgent

app = Flask(__name__)
agent = CounterfeitDetectionAgent()

@app.route('/verify', methods=['POST'])
def verify_product():
    file = request.files['image']
    file.save('temp.jpg')
    
    reference_data = {...}
    report = agent.process_image('temp.jpg', reference_data)
    
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)
```

## Performance Optimization

### For Faster Processing

1. **Resize images before processing:**
```python
image = cv2.imread('large_image.jpg')
image = cv2.resize(image, (800, 600))  # Resize to reasonable size
```

2. **Skip optional checks:**
```python
reference_data = {
    'logo_path': 'logo.png',
    'expected_text': None,  # Skip text verification
    'expected_qr_data': None,  # Skip QR verification
    'golden_image_path': None,  # Skip defect detection
    'color_reference': None  # Skip color verification
}
```

3. **Use GPU acceleration (if available):**
```python
# OpenCV with CUDA support
import cv2
cv2.cuda.getCudaEnabledDeviceCount()  # Check if CUDA is available
```

## Next Steps

1. ✅ Run `test_agent.py` to verify installation
2. ✅ Add your reference images to `reference/` folder
3. ✅ Add test images to `test_images/` folder
4. ✅ Adjust thresholds in `config.json`
5. ✅ Run `example_usage.py` with your images
6. ✅ Integrate into your application

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example scripts
3. Check OpenCV documentation: https://docs.opencv.org/
4. Check Tesseract documentation: https://github.com/tesseract-ocr/tesseract

## License

MIT License - Free to use and modify
