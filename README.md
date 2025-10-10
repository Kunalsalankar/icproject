# Counterfeit Detection AI Agent

A comprehensive AI-powered system for product authentication and counterfeit detection using computer vision and OpenCV.

## Features

The agent implements a **7-step detection pipeline**:

1. **Logo Detection** - Match genuine logo using template matching and ORB feature matching
2. **Text Area + OCR** - Detect and read serial text using Tesseract OCR
3. **QR/DMC Detection** - Detect and decode QR codes and Data Matrix codes
4. **Defect Detection** - Compare with golden IC using SSIM and absolute difference
5. **Angle/Edge Alignment** - Check rotation and tilt using Canny edge detection and Hough transform
6. **Color Calibration** - Verify ink/surface color uniformity using LAB color space
7. **Counterfeit Check** - Generate final verdict (GENUINE/COUNTERFEIT)

## Installation

### Prerequisites

1. **Python 3.8+** installed
2. **Tesseract OCR** installed:
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
     - Add Tesseract to PATH or set `pytesseract.pytesseract.tesseract_cmd` in code
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Mac**: `brew install tesseract`

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from counterfeit_detection_agent import CounterfeitDetectionAgent

# Initialize agent
agent = CounterfeitDetectionAgent()

# Prepare reference data
reference_data = {
    'logo_path': 'reference/genuine_logo.png',
    'expected_text': 'SN123456789',
    'expected_qr_data': 'GENUINE-PRODUCT-CODE-12345',
    'golden_image_path': 'reference/golden_product.png',
    'color_reference': {
        'bgr': [180, 180, 180],
    }
}

# Process test image
report = agent.process_image('test_images/product.jpg', reference_data)

# Save report
agent.save_report(report, 'detection_report.json')

print(f"Verdict: {report['verdict']}")
```

### Running the Example

```bash
python counterfeit_detection_agent.py
```

## Project Structure

```
AI_PROect/
├── counterfeit_detection_agent.py  # Main agent implementation
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── reference/                       # Reference images (create this)
│   ├── genuine_logo.png
│   ├── golden_product.png
│   └── ...
├── test_images/                     # Test images (create this)
│   └── product_to_verify.jpg
└── detection_report.json            # Output report (generated)
```

## Reference Data Format

### Logo Template
- PNG/JPG image of the genuine logo
- Should be clear and high-resolution

### Golden Image
- Reference image of a genuine product
- Used for defect detection comparison

### Expected Text
- String containing the expected serial number or text

### Expected QR Data
- String containing the expected QR/DMC code data

### Color Reference
- Dictionary with BGR color values: `{'bgr': [B, G, R]}`

## Output Report

The agent generates a JSON report with:

```json
{
  "timestamp": "2025-10-10T15:22:10",
  "verdict": "GENUINE" or "COUNTERFEIT",
  "pipeline_results": [
    {
      "step": "1. Logo Detection",
      "status": "PASS",
      "confidence": 0.95,
      "details": {...}
    },
    ...
  ],
  "summary": {
    "total_checks": 7,
    "passed": 6,
    "failed": 1,
    "skipped": 0,
    "errors": 0
  }
}
```

## OpenCV Methods Used

### Step 1: Logo Detection
- `cv2.matchTemplate()` - Template matching
- `cv2.ORB_create()` - ORB feature detector
- `cv2.BFMatcher()` - Brute-force matcher

### Step 2: Text Area + OCR
- `cv2.threshold()` - Image thresholding
- `cv2.findContours()` - Contour detection
- `pytesseract.image_to_string()` - OCR

### Step 3: QR/DMC Detection
- `pyzbar.decode()` - QR/barcode decoder
- `cv2.QRCodeDetector()` - QR detection

### Step 4: Defect Detection
- `cv2.absdiff()` - Absolute difference
- `skimage.metrics.ssim()` - Structural similarity
- `cv2.threshold()` - Thresholding

### Step 5: Angle/Edge Alignment
- `cv2.Canny()` - Edge detection
- `cv2.HoughLinesP()` - Line detection
- `cv2.goodFeaturesToTrack()` - Corner detection

### Step 6: Color Calibration
- `cv2.cvtColor()` - Color space conversion (LAB)
- `cv2.calcHist()` - Histogram calculation
- `cv2.compareHist()` - Histogram comparison

## Configuration

You can customize thresholds by modifying the agent initialization:

```python
agent = CounterfeitDetectionAgent()
agent.logo_match_threshold = 0.7    # Logo match threshold
agent.ssim_threshold = 0.85         # SSIM threshold for defects
agent.color_tolerance = 15          # Color difference tolerance
agent.angle_tolerance = 5.0         # Angle deviation tolerance (degrees)
```

## Verdict Logic

A product is classified as **GENUINE** if:
1. Average confidence across all checks ≥ 0.75
2. No more than 1 failed check
3. Critical checks (Logo, QR, Defect) must pass

Otherwise, it's classified as **COUNTERFEIT**.

## Troubleshooting

### Tesseract Not Found
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### QR Code Not Detected
- Ensure image quality is good
- Check lighting and contrast
- Try preprocessing (blur, threshold)

### Low Logo Match Score
- Ensure template image is similar in size
- Check if logo is clearly visible
- Adjust `logo_match_threshold`

## License

MIT License

## Author

Counterfeit Detection AI Agent
Created: 2025-10-10
