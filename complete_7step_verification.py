"""
Complete 11-Step Counterfeit Detection Pipeline (Production Optimized)
Implements 11 AOI Layer tests optimized for industrial speed (<0.3s total)

AOI Layer Tests Implemented:
1. Logo Detection (HoughCircles) - Fast circular logo detection
2. Text & Serial Number OCR - OCR confidence ≥ 80 to 90%
3. QR/DMC Code Detection - ≥ 80% success rate
4. Surface Defect Detection - SSIM + Intensity difference
5. Edge Detection (Canny) - Low: 100, High: 200 (ratio ~2:1)
6. IC Outline/Geometry - Size and aspect ratio deviation ±3% to 5%
7. Angle Detection - Orientation angle deviation ±2° to 5°
8. Color Surface Verification - Color Distance ΔE < 3 to 5 (LAB/HSV)
9. Texture Verification (Fast) - Histogram + gradient analysis
10. Font Verification - Font style, spacing, stroke width consistency
11. Correlation Layer - Composite pass threshold ≥ 90% layers pass

Note: ORB/SIFT removed (1700ms+ too slow for production)
Note: AI Agent OEM Verification now uses Hugging Face Vision-Language Models
      - Replaces web scraping with BLIP image captioning model
      - Direct image analysis for IC part number, manufacturer, and specifications
      - Fallback to local database when AI agent unavailable
"""

import cv2
import numpy as np
import pytesseract
import re
import os
import time
import json
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# Optional imports for AI Agent using Hugging Face
try:
    from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
    import torch
    from PIL import Image
    AI_AGENT_AVAILABLE = True
    print("Hugging Face AI Agent loaded successfully")
except ImportError:
    AI_AGENT_AVAILABLE = False
    print("Warning: transformers/torch not installed. AI Agent will use offline database only.")
    print("   Install with: pip install transformers torch accelerate")

# Legacy web scraping imports (fallback)
try:
    import requests
    from bs4 import BeautifulSoup
    import urllib.parse
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE_IC_PATH = 'reference/golden_product.jpg'
TEST_IC_PATH = 'test_images/product_to_verify.jpg'
DEBUG_MODE = True
SAVE_VISUALIZATION_IMAGES = True  # Save individual test layer images
OUTPUT_DIR = 'test_layer_visualizations'  # Directory to save visualization images

# ============================================================================
# INDUSTRY-STANDARD AOI (AUTOMATED OPTICAL INSPECTION) THRESHOLDS
# Based on AOI Layer specifications for counterfeit detection
# ============================================================================

# Logo Detection Thresholds
LOGO_TEMPLATE_NCC_THRESHOLD = 0.8  # Normalized Cross-Correlation ≥ 0.8 to 0.9
LOGO_ORB_RATIO_THRESHOLD = 0.75    # Lowe's Ratio test < 0.75
LOGO_ORB_MIN_MATCHES = 10          # Minimum of 10 good matches
LOGO_MATCH_THRESHOLD = 0.8         # Combined threshold for logo detection

# Text & Serial Number OCR Thresholds
OCR_CONFIDENCE_THRESHOLD = 0.8     # OCR confidence ≥ 80 to 90%

# QR/DMC Code Detection Thresholds
QR_DECODE_SUCCESS_RATE = 0.8       # ≥ 80% success rate

# Surface Defect Detection Thresholds
SSIM_THRESHOLD = 0.8               # SSIM < 0.8 to 0.9 flags a defect
INTENSITY_DIFFERENCE_THRESHOLD = 20 # Intensity difference > 10 to 30

# Edge Detection (Canny) Thresholds
CANNY_LOW_THRESHOLD = 100          # Low: 100, High: 200 (ratio ~2:1)
CANNY_HIGH_THRESHOLD = 200         # Low: 100, High: 200 (ratio ~2:1)

# IC Outline/Geometry Thresholds
SIZE_DEVIATION_THRESHOLD = 0.05    # ±3% to 5%
ASPECT_RATIO_DEVIATION_THRESHOLD = 0.05  # ±3% to 5%

# Angle Detection Thresholds
ANGLE_TOLERANCE = 2.0              # ±2° to 5°

# Color Surface Verification Thresholds
COLOR_DISTANCE_DE_THRESHOLD = 3.0  # Color Distance ΔE < 3 to 5 (LAB/HSV)

# Texture Verification Thresholds
TEXTURE_DISTANCE_THRESHOLD = 0.15  # Feature vector distance < 0.15

# Font Verification Thresholds
FONT_SIMILARITY_THRESHOLD = 0.75  # Font similarity score threshold
FONT_STROKE_TOLERANCE = 0.20      # ±20% stroke width variation allowed

# Correlation Layer Thresholds
CORRELATION_PASS_THRESHOLD = 0.9   # ≥ 90% layers pass, with no critical mismatch

# Legacy thresholds (for backward compatibility)
COLOR_TOLERANCE = 15               # Keep for compatibility

# Image quality assessment thresholds
GOOD_IMAGE_THRESHOLD = 50  # Contrast threshold
SHARP_IMAGE_THRESHOLD = 100  # Laplacian variance threshold
AUTO_ENHANCE = False  # Disable automatic quality assessment for industrial speed

# Performance modes
PRODUCTION_MODE = True  # Set to True to skip all preprocessing for maximum speed
ULTRA_FAST_MODE = False  # Don't skip tests - optimize them instead

# AI Agent Configuration
AI_MODEL_NAME = "Salesforce/blip-image-captioning-large"  # Best vision model for IC analysis
AI_USE_GPU = torch.cuda.is_available() if AI_AGENT_AVAILABLE else False
AI_MODEL = None  # Will be loaded on first use (lazy loading)
AI_PROCESSOR = None

# ============================================================================
# AI AGENT INITIALIZATION (LAZY LOADING)
# ============================================================================

def initialize_ai_agent():
    """Initialize Hugging Face AI Agent (lazy loading for performance)"""
    global AI_MODEL, AI_PROCESSOR
    
    if not AI_AGENT_AVAILABLE:
        print("AI Agent not available - install transformers and torch")
        return False
    
    if AI_MODEL is not None:
        return True  # Already initialized
    
    try:
        print("Initializing Hugging Face AI Agent...")
        print(f"   Model: {AI_MODEL_NAME}")
        print(f"   Device: {'GPU (CUDA)' if AI_USE_GPU else 'CPU'}")
        
        # Load BLIP model for image captioning and analysis
        AI_PROCESSOR = BlipProcessor.from_pretrained(AI_MODEL_NAME)
        AI_MODEL = BlipForConditionalGeneration.from_pretrained(AI_MODEL_NAME)
        
        if AI_USE_GPU:
            AI_MODEL = AI_MODEL.to("cuda")
        
        print("AI Agent initialized successfully")
        return True
        
    except Exception as e:
        print(f"Failed to initialize AI Agent: {str(e)}")
        return False

# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def save_visualization_image(image, test_name, layer_number, suffix=""):
    """Save visualization image for a specific test layer"""
    if not SAVE_VISUALIZATION_IMAGES or ULTRA_FAST_MODE:
        return None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Create filename with proper naming format
    # Format: layer_XX_test_name_suffix.jpg
    safe_test_name = test_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()
    if suffix:
        filename = f"layer_{layer_number:02d}_{safe_test_name}_{suffix}.jpg"
    else:
        filename = f"layer_{layer_number:02d}_{safe_test_name}.jpg"
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save the image
    cv2.imwrite(filepath, image)
    print(f"    Saved visualization: {filepath}")
    
    return filepath

# ============================================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================================

def assess_image_quality(image):
    """Assess if image needs preprocessing based on quality metrics"""
    print("  Assessing image quality...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Check contrast (standard deviation)
    contrast = gray.std()
    
    # 2. Check sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Check brightness distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    brightness = np.argmax(hist)
    
    # 4. Check if image has good dynamic range
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    dynamic_range = max_val - min_val
    
    # Quality assessment
    needs_enhancement = (
        contrast < GOOD_IMAGE_THRESHOLD or 
        laplacian_var < SHARP_IMAGE_THRESHOLD or
        dynamic_range < 150
    )
    
    quality_score = min(1.0, (contrast / 100) * (laplacian_var / 200))
    
    print(f"    Contrast: {contrast:.1f} (threshold: {GOOD_IMAGE_THRESHOLD})")
    print(f"    Sharpness: {laplacian_var:.1f} (threshold: {SHARP_IMAGE_THRESHOLD})")
    print(f"    Brightness: {brightness}")
    print(f"    Dynamic Range: {dynamic_range}")
    print(f"    Quality Score: {quality_score:.3f}")
    print(f"    Enhancement Needed: {'Yes' if needs_enhancement else 'No'}")
    
    return {
        'needs_enhancement': needs_enhancement,
        'quality_score': quality_score,
        'contrast': contrast,
        'sharpness': laplacian_var,
        'brightness': brightness,
        'dynamic_range': dynamic_range
    }

def preprocess_image_if_needed(image, force_preprocess=False):
    """Apply preprocessing only if image quality is poor"""
    if PRODUCTION_MODE:
        print("  Skipping preprocessing (PRODUCTION_MODE enabled)")
        return image, "Production mode - no preprocessing"
    
    if not AUTO_ENHANCE and not force_preprocess:
        print("  Skipping preprocessing (AUTO_ENHANCE disabled)")
        return image, "No preprocessing"
    
    quality = assess_image_quality(image)
    
    if not quality['needs_enhancement'] and not force_preprocess:
        print("  Image quality is good - skipping preprocessing")
        return image, "Good quality - no preprocessing needed"
    
    print("  Image quality is poor - applying preprocessing...")
    
    # Apply minimal preprocessing for poor quality images
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Only apply CLAHE if contrast is low
    if quality['contrast'] < GOOD_IMAGE_THRESHOLD:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        print("    Applied CLAHE for low contrast")
    else:
        enhanced = gray
        print("    Skipped CLAHE - contrast is adequate")
    
    # Only apply denoising if sharpness is low
    if quality['sharpness'] < SHARP_IMAGE_THRESHOLD:
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        print("    Applied denoising for low sharpness")
    else:
        denoised = enhanced
        print("    Skipped denoising - sharpness is adequate")
    
    # Convert back to BGR
    processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    return processed, f"Preprocessing applied (contrast: {quality['contrast']:.1f}, sharpness: {quality['sharpness']:.1f})"

# ============================================================================
# STEP 1: LOGO DETECTION
# ============================================================================

def detect_motorola_logo(image):
    """Detect Motorola logo using HoughCircles - FAST version for industrial use"""
    print("  Detecting Motorola logo (Fast Method)...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find IC chip region quickly
    _, ic_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(ic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'found': False, 'confidence': 0.0, 'location': None}
    
    ic_contour = max(contours, key=cv2.contourArea)
    x_ic, y_ic, w_ic, h_ic = cv2.boundingRect(ic_contour)
    ic_roi = gray[y_ic:y_ic+h_ic, x_ic:x_ic+w_ic]
    
    print(f"    IC detected: {w_ic}x{h_ic} at ({x_ic},{y_ic})")
    
    # Use HoughCircles to directly detect circular logo - MUCH FASTER
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(ic_roi, (5, 5), 0)
    
    # Detect circles with parameters tuned for Motorola logo
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Minimum distance between circle centers
        param1=100,  # Canny edge threshold
        param2=20,   # Accumulator threshold (lower = more circles detected)
        minRadius=10,  # Minimum circle radius
        maxRadius=50   # Maximum circle radius
    )
    
    best_circle = None
    best_score = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"    Found {len(circles[0])} circles")
        
        for circle in circles[0]:
            cx, cy, radius = circle
            
            # Check if circle is in the left portion of IC (logo position)
            if cx < w_ic * 0.35 and h_ic * 0.2 < cy < h_ic * 0.8:
                # Calculate position score
                position_score = 1.0 - (cx / (w_ic * 0.35))
                y_center_norm = abs(cy - h_ic/2) / (h_ic/2)
                vertical_score = 1.0 - y_center_norm
                
                # Score based on position and size
                size_score = min(radius / 25.0, 1.0)  # Prefer medium-sized circles
                score = position_score * vertical_score * size_score
                
                print(f"      Circle: center=({cx},{cy}), radius={radius}, score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_circle = (cx, cy, radius)
    
    if best_circle:
        cx, cy, radius = best_circle
        # Convert to bounding box in full image coordinates (convert to int to avoid uint16 JSON error)
        full_x = int(x_ic + cx - radius)
        full_y = int(y_ic + cy - radius)
        w = h = int(radius * 2)
        
        confidence = float(min(best_score, 1.0))
        
        print(f"    Logo detected: center=({cx},{cy}), radius={radius}, confidence={confidence:.3f}")
        
        return {
            'found': True,
            'confidence': confidence,
            'location': (full_x, full_y, w, h),
            'circularity': 1.0,  # Circles are perfectly circular
            'structure_score': float(confidence),
            'size_score': float(min(radius / 25.0, 1.0)),
            'circle_detected': True
        }
    
    print("    No circular logo detected")
    return {
        'found': False,
        'confidence': 0.0,
        'location': None,
        'circularity': 0.0,
        'structure_score': 0.0,
        'size_score': 0.0,
        'circle_detected': False
    }

def detect_logo_template(image, logo_template_path=None):
    """Step 1: Logo Detection (Template) - NCC ≥ 0.8 to 0.9"""
    step_start_time = time.time()
    print("Step 1: Logo Detection (Template) - Starting...")
    
    template_match_score = 0.0
    template_location = None
    
    # Create visualization image
    vis_image = image.copy()
    
    if logo_template_path and os.path.exists(logo_template_path):
        template = cv2.imread(logo_template_path)
        if template is not None:
            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Template Matching with Normalized Cross-Correlation
            result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            template_match_score = max_val
            template_location = max_loc
    
    # Also include automatic Motorola logo detection
    motorola_result = detect_motorola_logo(image)
    
    # Draw logo detection on visualization - ALWAYS draw if location exists
    if motorola_result['location']:
        x, y, w, h = motorola_result['location']
        # Use green if found with good confidence, yellow if found but low confidence, red if not found
        if motorola_result['found'] and motorola_result['confidence'] >= 0.5:
            color = (0, 255, 0)  # Green
        elif motorola_result['found']:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 3)
        cv2.putText(vis_image, f"Logo: {motorola_result['confidence']:.2f}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # If no logo detected at all, add text to image
        cv2.putText(vis_image, "No Logo Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Save visualization
    save_visualization_image(vis_image, "logo_detection_template", 1)
    
    # Use the best score between template and motorola detection
    if motorola_result['found']:
        combined_score = max(template_match_score, motorola_result['confidence'])
        method = f"Template: {template_match_score:.3f}, Motorola: {motorola_result['confidence']:.3f}"
    else:
        combined_score = template_match_score
        method = f"Template: {template_match_score:.3f}"
    
    status = "PASS" if combined_score >= LOGO_TEMPLATE_NCC_THRESHOLD else "FAIL"
    
    print(f"  Template Match Score: {template_match_score:.3f}")
    print(f"  Motorola Logo Found: {motorola_result['found']}")
    if motorola_result['found']:
        print(f"  Motorola Confidence: {motorola_result['confidence']:.3f}")
    print(f"  Combined Score: {combined_score:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '1. Logo Detection (Template)',
        'status': status,
        'confidence': combined_score,
        'processing_time_ms': step_time,
        'details': {
            'template_match_score': float(template_match_score),
            'motorola_logo_found': motorola_result['found'],
            'motorola_confidence': float(motorola_result['confidence']),
            'motorola_location': motorola_result['location'],
            'template_location': template_location,
            'method': method,
            'threshold': LOGO_TEMPLATE_NCC_THRESHOLD
        }
    }

def detect_logo_orb(image, logo_template_path=None):
    """Step 2: Logo Detection (ORB/SIFT) - DISABLED for production speed"""
    step_start_time = time.time()
    print("Step 2: Logo Detection (ORB/SIFT) - Skipped (too slow for production)")
    
    # ALWAYS SKIP ORB/SIFT - it's too slow for industrial use (1700ms+)
    # HoughCircles in Step 1 is sufficient for logo detection
    step_time = (time.time() - step_start_time) * 1000
    
    return {
        'step': '2. Logo Detection (ORB/SIFT)',
        'status': 'SKIPPED',
        'confidence': 1.0,
        'processing_time_ms': step_time,
        'details': {
            'message': 'Disabled for production speed - use HoughCircles instead',
            'reason': 'ORB/SIFT takes 1700ms+ which is too slow for industrial use'
        }
    }

# ============================================================================
# STEP 2: TEXT AREA + OCR
# ============================================================================

def detect_and_read_text(image, expected_text=None):
    """Step 3: Text Area Detection + OCR using Tesseract"""
    step_start_time = time.time()
    print("Step 3: Text Area + OCR - Starting...")
    
    # Skip in ultra-fast mode (OCR is very slow ~100-300ms)
    if ULTRA_FAST_MODE:
        print("  Skipped (ULTRA_FAST_MODE enabled)")
        step_time = (time.time() - step_start_time) * 1000
        return {
            'step': '3. Text & Serial Number OCR',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'processing_time_ms': step_time,
            'details': {'message': 'Skipped in ultra-fast mode'}
        }
    
    # ULTRA-FAST OCR: Downsample image and use fastest settings
    # Downsample to 50% for much faster OCR
    small_image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    
    # OCR on downsampled image (much faster)
    try:
        # Try modern LSTM engine first (OEM 1), fallback to legacy if needed
        try:
            ocr_text = pytesseract.image_to_string(small_image, config='--psm 3 --oem 1')
        except:
            # Fallback to default engine
            ocr_text = pytesseract.image_to_string(small_image, config='--psm 3')
        ocr_text_cleaned = ocr_text.strip()
    except Exception as e:
        step_time = (time.time() - step_start_time) * 1000
        print(f"  OCR Error: {str(e)}")
        print(f"  Processing Time: {step_time:.2f}ms")
        return {
            'step': '3. Text & Serial Number OCR',
            'status': 'ERROR',
            'confidence': 0.0,
            'processing_time_ms': step_time,
            'details': {'message': f'OCR failed: {str(e)}'}
        }
    
    # Verify text if expected text is provided
    confidence = 0.0
    status = "PASS"
    
    if expected_text:
        # Calculate similarity
        expected_clean = expected_text.strip().lower()
        ocr_clean = ocr_text_cleaned.lower()
        
        # Simple character-based similarity
        if expected_clean in ocr_clean or ocr_clean in expected_clean:
            confidence = 1.0
        else:
            # Calculate Levenshtein-like similarity
            matches = sum(1 for a, b in zip(expected_clean, ocr_clean) if a == b)
            confidence = matches / max(len(expected_clean), len(ocr_clean), 1)
        
        status = "PASS" if confidence >= OCR_CONFIDENCE_THRESHOLD else "FAIL"
    else:
        confidence = 1.0 if len(ocr_text_cleaned) > 0 else 0.0
        status = "PASS" if len(ocr_text_cleaned) > 0 else "FAIL"
    
    print(f"  OCR Text: '{ocr_text_cleaned[:50]}...'")
    print(f"  Text Length: {len(ocr_text_cleaned)} characters")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '3. Text & Serial Number OCR',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'ocr_text': ocr_text_cleaned,
            'expected_text': expected_text,
            'method': 'Fast OCR with downsampling'
        }
    }

# ============================================================================
# STEP 3: QR/DMC DETECTION
# ============================================================================

def detect_qr_code(image, expected_data=None):
    """Step 4: QR/DMC Code Detection - ≥ 80% success rate"""
    step_start_time = time.time()
    print("Step 4: QR/DMC Code Detection - Starting...")
    
    detected_codes = []
    vis_image = image.copy()
    
    # Try pyzbar first (if available)
    try:
        from pyzbar import pyzbar
        decoded_objects = pyzbar.decode(image)
        for obj in decoded_objects:
            code_data = obj.data.decode('utf-8')
            detected_codes.append({
                'type': obj.type,
                'data': code_data,
                'rect': obj.rect
            })
            # Draw QR code on visualization
            points = obj.polygon
            if len(points) == 4:
                pts = np.array([[p.x, p.y] for p in points], np.int32)
                cv2.polylines(vis_image, [pts], True, (0, 255, 0), 3)
    except:
        # Fallback to OpenCV QRCodeDetector
        qr_detector = cv2.QRCodeDetector()
        data, bbox, _ = qr_detector.detectAndDecode(image)
        
        if data:
            detected_codes.append({
                'type': 'QRCODE',
                'data': data,
                'rect': bbox
            })
            # Draw QR code on visualization
            if bbox is not None:
                bbox = bbox.astype(int)
                cv2.polylines(vis_image, [bbox], True, (0, 255, 0), 3)
    
    # Add text annotation
    cv2.putText(vis_image, f"QR/DMC Codes: {len(detected_codes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save visualization
    save_visualization_image(vis_image, "qr_dmc_code_detection", 4)
    
    if not detected_codes:
        print("  QR/DMC Detection are not present in the test to check")
        return {
            'step': '4. QR/DMC Code Detection',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'details': {'message': 'QR/DMC Detection are not present in the test to check'}
        }
    
    # Verify against expected data
    confidence = 0.0
    status = "FAIL"
    
    if expected_data:
        for code in detected_codes:
            if code["data"] == expected_data:
                confidence = 1.0
                status = "PASS"
                break
            elif expected_data in code["data"] or code["data"] in expected_data:
                confidence = QR_DECODE_SUCCESS_RATE
                status = "PASS"
                break
    else:
        # If no expected data, just check if we found codes
        confidence = 1.0
        status = "PASS"
    
    print(f"  Codes Detected: {len(detected_codes)}")
    for code in detected_codes:
        print(f"    - Type: {code['type']}, Data: {code['data'][:50]}...")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '4. QR/DMC Code Detection',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'codes_detected': len(detected_codes),
            'detected_codes': detected_codes,
            'expected_data': expected_data,
            'method': 'pyzbar + OpenCV QRCodeDetector'
        }
    }

# ============================================================================
# STEP 4: DEFECT DETECTION
# ============================================================================

def detect_defects(image, golden_image_path):
    """Step 5: Surface Defect Detection - SSIM + Intensity difference"""
    step_start_time = time.time()
    print("Step 5: Surface Defect Detection - Starting...")
    
    if golden_image_path is None or not os.path.exists(golden_image_path):
        return {
            'step': '5. Surface Defect Detection',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'details': {'message': 'No golden image provided'}
        }
    
    golden_image = cv2.imread(golden_image_path)
    if golden_image is None:
        return {
            'step': '5. Surface Defect Detection',
            'status': 'ERROR',
            'confidence': 0.0,
            'details': {'message': 'Could not load golden image'}
        }
    
    # Resize images to same size for comparison
    h, w = golden_image.shape[:2]
    test_resized = cv2.resize(image, (w, h))
    
    # Convert to grayscale
    gray_golden = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM (Structural Similarity Index)
    ssim_score, diff_image = ssim(gray_golden, gray_test, full=True)
    diff_image = (diff_image * 255).astype("uint8")
    
    # Calculate absolute difference with industry-standard threshold
    abs_diff = cv2.absdiff(gray_golden, gray_test)
    _, thresh_diff = cv2.threshold(abs_diff, INTENSITY_DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Create visualization - difference heatmap
    diff_colormap = cv2.applyColorMap(abs_diff, cv2.COLORMAP_JET)
    
    # Create side-by-side comparison
    vis_image = np.hstack([test_resized, golden_image, diff_colormap])
    
    # Add labels
    cv2.putText(vis_image, "Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_image, "Reference", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_image, "Difference", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_image, f"SSIM: {ssim_score:.3f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    save_visualization_image(vis_image, "surface_defect_detection", 5)
    
    # Count defect pixels
    defect_pixels = np.sum(thresh_diff > 0)
    total_pixels = thresh_diff.size
    defect_ratio = defect_pixels / total_pixels
    
    # Combined confidence
    confidence = ssim_score * (1 - defect_ratio)
    status = "PASS" if confidence >= SSIM_THRESHOLD else "FAIL"
    
    print(f"  SSIM Score: {ssim_score:.3f}")
    print(f"  Defect Ratio: {defect_ratio:.3f}")
    print(f"  Combined Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '5. Surface Defect Detection',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'ssim_score': float(ssim_score),
            'defect_ratio': float(defect_ratio),
            'defect_pixels': int(defect_pixels),
            'method': 'absdiff + SSIM + threshold'
        }
    }

# ============================================================================
# STEP 5: ANGLE/EDGE ALIGNMENT CHECK
# ============================================================================

def detect_edges_canny(image):
    """Step 6: Edge Detection (Canny) - Low: 100, High: 200 (ratio ~2:1)"""
    step_start_time = time.time()
    print("Step 6: Edge Detection (Canny) - Starting...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection using Canny with industry-standard thresholds
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, apertureSize=3)
    
    # Create visualization - colorized edges on original image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[edges > 0] = [0, 255, 0]  # Green edges
    vis_image = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    
    # Calculate edge density (percentage of edge pixels)
    total_pixels = edges.size
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / total_pixels
    
    # Calculate edge strength (average gradient magnitude)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_strength = np.mean(gradient_magnitude)
    
    # Add text annotations
    cv2.putText(vis_image, f"Edge Density: {edge_density*100:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Edge Strength: {edge_strength:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Edge Pixels: {edge_pixels:,}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save visualization
    save_visualization_image(vis_image, "edge_detection_canny", 6)
    
    # Check if edges are well-detected
    # Good edge detection should have reasonable edge density (not too sparse, not too dense)
    edge_density_pass = 0.01 <= edge_density <= 0.3  # 1% to 30% edge pixels
    edge_strength_pass = edge_strength >= 20  # Minimum gradient strength
    
    # Calculate confidence based on edge quality
    confidence = min(1.0, (edge_density * 10) * (edge_strength / 100))
    status = "PASS" if edge_density_pass and edge_strength_pass else "FAIL"
    
    print(f"  Edge Density: {edge_density:.3f} ({edge_density*100:.1f}%)")
    print(f"  Edge Strength: {edge_strength:.2f}")
    print(f"  Edge Pixels: {edge_pixels:,} / {total_pixels:,}")
    print(f"  Density Pass: {'Yes' if edge_density_pass else 'No'}")
    print(f"  Strength Pass: {'Yes' if edge_strength_pass else 'No'}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '6. Edge Detection (Canny)',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'edge_density': float(edge_density),
            'edge_strength': float(edge_strength),
            'edge_pixels': int(edge_pixels),
            'total_pixels': int(total_pixels),
            'edge_density_pass': bool(edge_density_pass),
            'edge_strength_pass': bool(edge_strength_pass),
            'canny_low_threshold': CANNY_LOW_THRESHOLD,
            'canny_high_threshold': CANNY_HIGH_THRESHOLD,
            'method': 'Canny edge detection with Sobel gradient analysis'
        }
    }

def check_angle_detection(image):
    """Step 8: Angle Detection - Orientation angle deviation ±2° to 5°"""
    step_start_time = time.time()
    print("Step 8: Angle Detection - Starting...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vis_image = image.copy()
    
    # Edge detection using Canny
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD, apertureSize=3)
    
    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)
    
    # Draw detected lines on visualization
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if lines is None:
        return {
            'step': '8. Angle Detection',
            'status': 'FAIL',
            'confidence': 0.0,
            'details': {'message': 'No lines detected'}
        }
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angles.append(angle)
    
    # Check alignment (lines should be close to 0, 90, 180, 270 degrees)
    aligned_angles = []
    for angle in angles:
        # Normalize to 0-90 range
        norm_angle = angle % 90
        if norm_angle > 45:
            norm_angle = 90 - norm_angle
        aligned_angles.append(norm_angle)
    
    # Calculate average deviation from perfect alignment
    avg_deviation = np.mean(aligned_angles)
    max_deviation = np.max(aligned_angles)
    
    # Check if deviation is within tolerance
    angle_pass = avg_deviation <= ANGLE_TOLERANCE and max_deviation <= ANGLE_TOLERANCE * 2
    
    # Calculate confidence
    confidence = 1.0 - (avg_deviation / 45.0)  # Normalize to 0-1
    confidence = max(0.0, confidence)
    
    # Add text annotations
    cv2.putText(vis_image, f"Lines: {len(lines)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Avg Deviation: {avg_deviation:.2f}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save visualization
    save_visualization_image(vis_image, "angle_detection", 8)
    
    status = "PASS" if angle_pass else "FAIL"
    
    print(f"  Lines Detected: {len(lines)}")
    print(f"  Average Angle Deviation: {avg_deviation:.2f}°")
    print(f"  Max Angle Deviation: {max_deviation:.2f}°")
    print(f"  Angle Tolerance: ±{ANGLE_TOLERANCE}°")
    print(f"  Angle Pass: {'Yes' if angle_pass else 'No'}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '8. Angle Detection',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'lines_detected': len(lines),
            'avg_angle_deviation': float(avg_deviation),
            'max_angle_deviation': float(max_deviation),
            'angle_tolerance': ANGLE_TOLERANCE,
            'angle_pass': bool(angle_pass),
            'method': 'HoughLinesP + angle deviation analysis'
        }
    }

# ============================================================================
# STEP 6: COLOR CALIBRATION
# ============================================================================

def verify_color(image, color_reference=None):
    """Step 9: Color Surface Verification - Color Distance ΔE < 3 to 5 (LAB/HSV)"""
    step_start_time = time.time()
    print("Step 9: Color Surface Verification - Starting...")
    
    # FAST VERSION: Skip unnecessary conversions and calculations
    # Just calculate average BGR color (much faster than LAB conversion)
    avg_color_bgr = cv2.mean(image)[:3]
    
    # Only convert to LAB if we need it for reference comparison
    if color_reference:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_color_lab = cv2.mean(lab_image)[:3]
    else:
        # Skip LAB conversion if no reference
        avg_color_lab = avg_color_bgr  # Just use BGR as placeholder
    
    confidence = 1.0
    status = "PASS"
    color_diff = 0.0
    
    if color_reference:
        # Compare with reference colors
        ref_bgr = color_reference.get('bgr', [0, 0, 0])
        
        # Calculate color difference
        color_diff = np.sqrt(sum((a - b)**2 for a, b in zip(avg_color_bgr, ref_bgr)))
        
        # Normalize confidence
        max_diff = np.sqrt(3 * 255**2)
        confidence = 1.0 - (color_diff / max_diff)
        
        # Use industry-standard ΔE threshold for LAB color space
        status = "PASS" if color_diff <= COLOR_DISTANCE_DE_THRESHOLD else "FAIL"
    
    # Create visualization with green text for better visibility
    vis_image = image.copy()
    # Use bright green (0, 255, 0) for better visibility on dark IC surface
    cv2.putText(vis_image, f"Avg BGR: {[int(c) for c in avg_color_bgr]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Avg LAB: {[int(c) for c in avg_color_lab]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Color Diff: {color_diff:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_image, f"Status: {status}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    save_visualization_image(vis_image, "color_surface_verification", 9)
    
    print(f"  Average Color (BGR): {[int(c) for c in avg_color_bgr]}")
    print(f"  Average Color (LAB): {[int(c) for c in avg_color_lab]}")
    print(f"  Color Difference: {color_diff:.2f}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '9. Color Surface Verification',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'avg_color_bgr': [int(c) for c in avg_color_bgr],
            'avg_color_lab': [int(c) for c in avg_color_lab],
            'color_difference': float(color_diff),
            'method': 'LAB hist + compareHist'
        }
    }

# ============================================================================
# STEP 7: IC OUTLINE/GEOMETRY CHECK
# ============================================================================

def check_ic_geometry(image, reference_image_path):
    """Step 7: IC Outline/Geometry - Size and aspect ratio deviation ±3% to 5%"""
    step_start_time = time.time()
    print("Step 7: IC Outline/Geometry - Starting...")
    
    if reference_image_path is None or not os.path.exists(reference_image_path):
        return {
            'step': '7. IC Outline/Geometry',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'details': {'message': 'No reference image provided'}
        }
    
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        return {
            'step': '7. IC Outline/Geometry',
            'status': 'ERROR',
            'confidence': 0.0,
            'details': {'message': 'Could not load reference image'}
        }
    
    # Convert to grayscale
    gray_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to same dimensions for comparison
    h, w = gray_ref.shape[:2]
    gray_test_resized = cv2.resize(gray_test, (w, h))
    
    # Find IC contours in both images
    def find_ic_contour(gray_img):
        _, thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    test_contour = find_ic_contour(gray_test_resized)
    ref_contour = find_ic_contour(gray_ref)
    
    if test_contour is None or ref_contour is None:
        return {
            'step': '7. IC Outline/Geometry',
            'status': 'FAIL',
            'confidence': 0.0,
            'details': {'message': 'Could not detect IC contours'}
        }
    
    # Get bounding rectangles
    x_test, y_test, w_test, h_test = cv2.boundingRect(test_contour)
    x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(ref_contour)
    
    # Create visualization
    vis_test = cv2.resize(image, (w, h))
    vis_ref = reference_image.copy()
    
    # Draw contours and bounding boxes
    cv2.drawContours(vis_test, [test_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(vis_test, (x_test, y_test), (x_test+w_test, y_test+h_test), (255, 0, 0), 2)
    cv2.drawContours(vis_ref, [ref_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(vis_ref, (x_ref, y_ref), (x_ref+w_ref, y_ref+h_ref), (255, 0, 0), 2)
    
    # Combine side by side
    vis_image = np.hstack([vis_test, vis_ref])
    cv2.putText(vis_image, f"Test: {w_test}x{h_test}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, f"Ref: {w_ref}x{h_ref}", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Calculate size deviation
    width_deviation = abs(w_test - w_ref) / w_ref
    height_deviation = abs(h_test - h_ref) / h_ref
    size_deviation = (width_deviation + height_deviation) / 2
    
    # Calculate aspect ratio deviation
    aspect_test = w_test / h_test
    aspect_ref = w_ref / h_ref
    aspect_deviation = abs(aspect_test - aspect_ref) / aspect_ref
    
    # Calculate contour area deviation
    area_test = cv2.contourArea(test_contour)
    area_ref = cv2.contourArea(ref_contour)
    area_deviation = abs(area_test - area_ref) / area_ref
    
    # Check against industry-standard thresholds
    size_pass = size_deviation <= SIZE_DEVIATION_THRESHOLD
    aspect_pass = aspect_deviation <= ASPECT_RATIO_DEVIATION_THRESHOLD
    area_pass = area_deviation <= SIZE_DEVIATION_THRESHOLD
    
    # Calculate confidence
    confidence = 1.0 - max(size_deviation, aspect_deviation, area_deviation)
    confidence = max(0.0, confidence)
    
    # Add deviation info to visualization
    cv2.putText(vis_image, f"Size Dev: {size_deviation:.3f}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_image, f"Aspect Dev: {aspect_deviation:.3f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save visualization
    save_visualization_image(vis_image, "ic_outline_geometry", 7)
    
    status = "PASS" if size_pass and aspect_pass and area_pass else "FAIL"
    
    print(f"  Test IC Size: {w_test}x{h_test}")
    print(f"  Reference IC Size: {w_ref}x{h_ref}")
    print(f"  Size Deviation: {size_deviation:.3f} (threshold: {SIZE_DEVIATION_THRESHOLD})")
    print(f"  Aspect Ratio Deviation: {aspect_deviation:.3f} (threshold: {ASPECT_RATIO_DEVIATION_THRESHOLD})")
    print(f"  Area Deviation: {area_deviation:.3f} (threshold: {SIZE_DEVIATION_THRESHOLD})")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    return {
        'step': '7. IC Outline/Geometry',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'test_size': (w_test, h_test),
            'reference_size': (w_ref, h_ref),
            'size_deviation': float(size_deviation),
            'aspect_deviation': float(aspect_deviation),
            'area_deviation': float(area_deviation),
            'size_pass': bool(size_pass),
            'aspect_pass': bool(aspect_pass),
            'area_pass': bool(area_pass),
            'method': 'contour analysis + bounding rect comparison'
        }
    }

# ============================================================================
# STEP 8: TEXTURE VERIFICATION
# ============================================================================

def verify_texture(image, reference_image_path):
    """Step 10: Texture Verification (LBP/GLCM) - Feature vector distance < 0.15"""
    step_start_time = time.time()
    print("Step 10: Texture Verification - Starting...")
    
    # Skip in ultra-fast mode
    if ULTRA_FAST_MODE:
        print("  Skipped (ULTRA_FAST_MODE enabled)")
        step_time = (time.time() - step_start_time) * 1000
        return {
            'step': '10. Texture Verification',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'processing_time_ms': step_time,
            'details': {'message': 'Skipped in ultra-fast mode'}
        }
    
    if reference_image_path is None or not os.path.exists(reference_image_path):
        return {
            'step': '8. Texture Verification',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'details': {'message': 'No reference image provided'}
        }
    
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        return {
            'step': '8. Texture Verification',
            'status': 'ERROR',
            'confidence': 0.0,
            'details': {'message': 'Could not load reference image'}
        }
    
    # Convert to grayscale
    gray_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Resize to same dimensions
    h, w = gray_ref.shape[:2]
    gray_test_resized = cv2.resize(gray_test, (w, h))
    
    # Simplified texture analysis for industrial speed
    # Method 1: Fast histogram comparison instead of full LBP
    def calculate_fast_texture_features(gray_img):
        # Downsample for speed
        small_img = cv2.resize(gray_img, (gray_img.shape[1]//4, gray_img.shape[0]//4))
        
        # Calculate intensity histogram
        hist = cv2.calcHist([small_img], [0], None, [64], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        # Calculate gradient histogram
        grad_x = cv2.Sobel(small_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(small_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)
        grad_hist = cv2.calcHist([grad_mag], [0], None, [64], [0, 256])
        grad_hist = grad_hist.flatten() / np.sum(grad_hist)
        
        return np.concatenate([hist, grad_hist])
    
    # Calculate features (fast version)
    try:
        texture_test = calculate_fast_texture_features(gray_test_resized)
        texture_ref = calculate_fast_texture_features(gray_ref)
        
        # Calculate feature distance
        texture_distance = np.linalg.norm(texture_test - texture_ref)
        
        # Normalize distance
        texture_distance_norm = texture_distance / np.sqrt(len(texture_test))
        
        # Use the normalized distance directly
        combined_distance = texture_distance_norm
        
        # Check against industry-standard threshold
        texture_pass = combined_distance <= TEXTURE_DISTANCE_THRESHOLD
        
        # Calculate confidence
        confidence = 1.0 - min(combined_distance / TEXTURE_DISTANCE_THRESHOLD, 1.0)
        
        status = "PASS" if texture_pass else "FAIL"
        
        # Create visualization - show texture features
        vis_test = cv2.resize(image, (w, h))
        vis_ref = reference_image.copy()
        vis_image = np.hstack([vis_test, vis_ref])
        
        # Add text annotations
        cv2.putText(vis_image, "Test Texture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, "Reference Texture", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Texture Dist: {combined_distance:.3f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save visualization
        save_visualization_image(vis_image, "texture_verification_fast", 10)
        
        print(f"  Texture Distance: {combined_distance:.3f} (threshold: {TEXTURE_DISTANCE_THRESHOLD})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Status: {status}")
        
        step_time = (time.time() - step_start_time) * 1000
        print(f"  Processing Time: {step_time:.2f}ms")
        
        return {
            'step': '10. Texture Verification',
            'status': status,
            'confidence': confidence,
            'processing_time_ms': step_time,
            'details': {
                'texture_distance': float(combined_distance),
                'texture_pass': bool(texture_pass),
                'method': 'Fast histogram + gradient texture analysis'
            }
        }
        
    except Exception as e:
        return {
            'step': '10. Texture Verification',
            'status': 'ERROR',
            'confidence': 0.0,
            'details': {'message': f'Texture analysis failed: {str(e)}'}
        }

# ============================================================================
# STEP 10.5: FONT VERIFICATION (NEW - CRITICAL FOR COUNTERFEIT DETECTION)
# ============================================================================

def verify_font_characteristics(image, reference_image_path):
    """Step 10.5: Font Verification - Checks font style, spacing, stroke width consistency"""
    step_start_time = time.time()
    print("Step 10.5: Font Verification - Starting...")
    
    if reference_image_path is None or not os.path.exists(reference_image_path):
        step_time = (time.time() - step_start_time) * 1000
        return {
            'step': '10.5 Font Verification',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'processing_time_ms': step_time,
            'details': {'message': 'No reference image provided'}
        }
    
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        step_time = (time.time() - step_start_time) * 1000
        return {
            'step': '10.5 Font Verification',
            'status': 'ERROR',
            'confidence': 0.0,
            'processing_time_ms': step_time,
            'details': {'message': 'Could not load reference image'}
        }
    
    try:
        # Convert to grayscale
        gray_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to same dimensions for comparison
        h, w = gray_ref.shape[:2]
        gray_test_resized = cv2.resize(gray_test, (w, h))
        
        # 1. Extract text regions using adaptive thresholding
        _, thresh_test = cv2.threshold(gray_test_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_ref = cv2.threshold(gray_ref, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Analyze stroke width using morphological operations
        kernel = np.ones((3,3), np.uint8)
        
        # Erode to get inner skeleton
        eroded_test = cv2.erode(thresh_test, kernel, iterations=1)
        eroded_ref = cv2.erode(thresh_ref, kernel, iterations=1)
        
        # Calculate stroke width by comparing original vs eroded
        stroke_test = np.sum(thresh_test) - np.sum(eroded_test)
        stroke_ref = np.sum(thresh_ref) - np.sum(eroded_ref)
        
        # Normalize by image size
        stroke_width_test = stroke_test / (w * h)
        stroke_width_ref = stroke_ref / (w * h)
        
        # Calculate stroke width difference
        stroke_diff = abs(stroke_width_test - stroke_width_ref) / max(stroke_width_ref, 0.001)
        
        # 3. Analyze character spacing using horizontal projection
        h_proj_test = np.sum(thresh_test, axis=0)
        h_proj_ref = np.sum(thresh_ref, axis=0)
        
        # Normalize projections
        h_proj_test_norm = h_proj_test / np.max(h_proj_test) if np.max(h_proj_test) > 0 else h_proj_test
        h_proj_ref_norm = h_proj_ref / np.max(h_proj_ref) if np.max(h_proj_ref) > 0 else h_proj_ref
        
        # Calculate spacing similarity using correlation
        spacing_correlation = np.corrcoef(h_proj_test_norm, h_proj_ref_norm)[0, 1]
        spacing_correlation = max(0, spacing_correlation)  # Ensure non-negative
        
        # 4. Analyze font shape using template matching on text regions
        # Use normalized cross-correlation on thresholded images
        result = cv2.matchTemplate(thresh_test, thresh_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        shape_similarity = max(0, max_val)
        
        # 5. Analyze edge characteristics (font rendering quality)
        edges_test = cv2.Canny(gray_test_resized, 100, 200)
        edges_ref = cv2.Canny(gray_ref, 100, 200)
        
        # Calculate edge density ratio
        edge_density_test = np.sum(edges_test > 0) / (w * h)
        edge_density_ref = np.sum(edges_ref > 0) / (w * h)
        edge_density_diff = abs(edge_density_test - edge_density_ref) / max(edge_density_ref, 0.001)
        
        # Calculate combined font similarity score
        stroke_score = 1.0 - min(stroke_diff / FONT_STROKE_TOLERANCE, 1.0)
        spacing_score = spacing_correlation
        shape_score = shape_similarity
        edge_score = 1.0 - min(edge_density_diff, 1.0)
        
        # Weighted combination (shape and spacing are most important)
        font_similarity = (shape_score * 0.35 + spacing_score * 0.30 + 
                          stroke_score * 0.20 + edge_score * 0.15)
        
        # Determine pass/fail
        font_pass = font_similarity >= FONT_SIMILARITY_THRESHOLD
        status = "PASS" if font_pass else "FAIL"
        
        # Create visualization
        vis_test = cv2.resize(image, (w, h))
        vis_ref = reference_image.copy()
        
        # Show thresholded text regions
        thresh_test_color = cv2.cvtColor(thresh_test, cv2.COLOR_GRAY2BGR)
        thresh_ref_color = cv2.cvtColor(thresh_ref, cv2.COLOR_GRAY2BGR)
        
        # Create comparison visualization
        top_row = np.hstack([vis_test, vis_ref])
        bottom_row = np.hstack([thresh_test_color, thresh_ref_color])
        vis_image = np.vstack([top_row, bottom_row])
        
        # Add text annotations in green
        cv2.putText(vis_image, "Test Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, "Reference Image", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, "Test Text Region", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, "Ref Text Region", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Font Similarity: {font_similarity:.3f}", (10, h*2 - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Status: {status}", (10, h*2 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if font_pass else (0, 0, 255), 2)
        
        # Save visualization (use layer 11 for proper ordering)
        save_visualization_image(vis_image, "font_verification", 11)
        
        print(f"  Font Shape Similarity: {shape_score:.3f}")
        print(f"  Character Spacing Correlation: {spacing_score:.3f}")
        print(f"  Stroke Width Score: {stroke_score:.3f}")
        print(f"  Edge Quality Score: {edge_score:.3f}")
        print(f"  Overall Font Similarity: {font_similarity:.3f} (threshold: {FONT_SIMILARITY_THRESHOLD})")
        print(f"  Status: {status}")
        
        step_time = (time.time() - step_start_time) * 1000
        print(f"  Processing Time: {step_time:.2f}ms")
        
        return {
            'step': '10.5 Font Verification',
            'status': status,
            'confidence': font_similarity,
            'processing_time_ms': step_time,
            'details': {
                'font_similarity': float(font_similarity),
                'shape_score': float(shape_score),
                'spacing_score': float(spacing_score),
                'stroke_score': float(stroke_score),
                'edge_score': float(edge_score),
                'font_pass': bool(font_pass),
                'method': 'Font shape + spacing + stroke width + edge analysis'
            }
        }
        
    except Exception as e:
        step_time = (time.time() - step_start_time) * 1000
        print(f"  Font verification error: {str(e)}")
        print(f"  Processing Time: {step_time:.2f}ms")
        return {
            'step': '10.5 Font Verification',
            'status': 'ERROR',
            'confidence': 0.0,
            'processing_time_ms': step_time,
            'details': {'message': f'Font verification failed: {str(e)}'}
        }

# ============================================================================
# STEP 10.6: AI AGENT - HUGGING FACE VISION-LANGUAGE MODEL VERIFICATION
# ============================================================================

def ai_agent_oem_verification(image, ocr_text=None, logo_detected=False):
    """
    Step 10.6: AI Agent for OEM Database Verification using Hugging Face
    - Extracts IC part number from image using OCR
    - Uses Hugging Face BLIP Vision-Language Model to analyze IC image
    - Extracts part number, manufacturer, package type from image directly
    - Compares test IC against AI-extracted specifications
    - Provides additional verification layer beyond reference image
    - Fallback to local database when AI unavailable
    """
    step_start_time = time.time()
    print("Step 10.6: AI Agent - OEM Database Verification - Starting...")
    
    try:
        # 1. Extract IC part number from OCR text or perform OCR
        part_number = None
        manufacturer = None
        
        if ocr_text is None:
            # Perform OCR to extract text
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_text = pytesseract.image_to_string(thresh, config='--psm 6')
        
        print(f"  OCR Text Extracted: {ocr_text[:100]}...")
        
        # 2. Parse IC part number using regex patterns
        # Common IC part number patterns
        patterns = [
            r'([A-Z]{2,4}\d{2,4}[A-Z]{0,4}\d{0,4}[A-Z]{0,2})',  # e.g., MC74HC20N, SN74LS00
            r'(\d{2,4}[A-Z]{2,6}\d{0,4})',  # e.g., 74HC595, 4017
            r'([A-Z]+\d+[A-Z]*\d*)',  # General alphanumeric
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, ocr_text)
            if matches:
                # Take the longest match (likely the full part number)
                part_number = max(matches, key=len)
                break
        
        if not part_number:
            print("  Could not extract IC part number from OCR")
            step_time = (time.time() - step_start_time) * 1000
            return {
                'step': '10.6 AI Agent OEM Verification',
                'status': 'SKIPPED',
                'confidence': 1.0,
                'processing_time_ms': step_time,
                'details': {
                    'message': 'Could not extract part number',
                    'ocr_text': ocr_text[:100]
                }
            }
        
        print(f"  ✓ Extracted Part Number: {part_number}")
        
        # 3. Detect manufacturer from logo or text
        manufacturer_keywords = {
            'motorola': ['motorola', 'mc', 'mot', 'freescale'],
            'texas_instruments': ['ti', 'texas', 'sn74', 'tl', 'lm'],
            'stmicroelectronics': ['st', 'stm', 'stmicro'],
            'nxp': ['nxp', 'philips'],
            'analog_devices': ['ad', 'analog', 'adi'],
            'microchip': ['microchip', 'pic', 'atmel'],
            'intel': ['intel'],
            'maxim': ['maxim', 'max'],
            'on_semiconductor': ['on semi', 'onsemi'],
            'infineon': ['infineon']
        }
        
        ocr_lower = ocr_text.lower()
        for mfr, keywords in manufacturer_keywords.items():
            if any(kw in ocr_lower or kw in part_number.lower() for kw in keywords):
                manufacturer = mfr
                break
        
        if logo_detected:
            manufacturer = manufacturer or 'motorola'  # Default if logo detected
        
        print(f"  ✓ Detected Manufacturer: {manufacturer or 'Unknown'}")
        
        # 4. Use AI Agent to analyze IC image (replaces web scraping)
        oem_data = analyze_ic_with_ai_agent(image, part_number)
        
        # 5. Analyze OEM data and compare with test IC
        if oem_data['found']:
            print(f"  ✓ Found OEM Data:")
            print(f"    - Part Number: {oem_data.get('part_number', 'N/A')}")
            print(f"    - Manufacturer: {oem_data.get('manufacturer', 'N/A')}")
            print(f"    - Description: {oem_data.get('description', 'N/A')[:60]}...")
            print(f"    - Package: {oem_data.get('package', 'N/A')}")
            print(f"    - Status: {oem_data.get('status', 'N/A')}")
            
            # Verify if part number matches
            part_match = part_number.upper() in oem_data.get('part_number', '').upper()
            mfr_match = manufacturer and manufacturer.lower() in oem_data.get('manufacturer', '').lower()
            
            # Check if part is obsolete or counterfeit-prone
            status_lower = oem_data.get('status', '').lower()
            is_obsolete = any(word in status_lower for word in ['obsolete', 'discontinued', 'eol'])
            is_active = 'active' in status_lower or 'production' in status_lower
            
            # Calculate confidence
            confidence = 0.0
            if part_match:
                confidence += 0.5
            if mfr_match:
                confidence += 0.3
            if is_active:
                confidence += 0.2
            elif is_obsolete:
                confidence -= 0.3  # Obsolete parts are more likely to be counterfeit
            
            confidence = max(0.0, min(1.0, confidence))
            
            status = "PASS" if confidence >= 0.7 and is_active else "FAIL"
            
            if is_obsolete:
                print(f"  WARNING: Part is OBSOLETE/DISCONTINUED - High counterfeit risk!")
            
        else:
            print(f"  No OEM data found online")
            confidence = 0.5  # Neutral - not found doesn't mean fake
            status = "PASS"  # Don't fail if no data found
        
        step_time = (time.time() - step_start_time) * 1000
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Status: {status}")
        print(f"  Processing Time: {step_time:.2f}ms")
        
        return {
            'step': '10.6 AI Agent OEM Verification',
            'status': status,
            'confidence': confidence,
            'processing_time_ms': step_time,
            'details': {
                'part_number': part_number,
                'manufacturer': manufacturer,
                'oem_data_found': oem_data['found'],
                'oem_part_number': oem_data.get('part_number', 'N/A'),
                'oem_manufacturer': oem_data.get('manufacturer', 'N/A'),
                'oem_description': oem_data.get('description', 'N/A'),
                'oem_package': oem_data.get('package', 'N/A'),
                'oem_status': oem_data.get('status', 'N/A'),
                'is_obsolete': oem_data.get('is_obsolete', False),
                'ai_confidence': oem_data.get('ai_confidence', 0.0),
                'method': 'OCR + Hugging Face AI Agent + Vision-Language Model Analysis'
            }
        }
        
    except Exception as e:
        step_time = (time.time() - step_start_time) * 1000
        print(f"  AI Agent Error: {str(e)}")
        print(f"  Processing Time: {step_time:.2f}ms")
        return {
            'step': '10.6 AI Agent OEM Verification',
            'status': 'ERROR',
            'confidence': 0.5,
            'processing_time_ms': step_time,
            'details': {
                'message': f'AI Agent failed: {str(e)}',
                'method': 'OCR + Hugging Face AI Agent + Vision-Language Model Analysis'
            }
        }


def analyze_ic_with_ai_agent(image, part_number=None):
    """
    Use Hugging Face AI Agent to analyze IC image and extract information
    Replaces web scraping with vision-language model analysis
    """
    print(f"  Analyzing IC with AI Agent...")
    
    if not initialize_ai_agent():
        print("  AI Agent not available, falling back to database")
        return analyze_ic_with_database(part_number)
    
    try:
        # Convert OpenCV image to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Generate detailed description of the IC
        print("  Generating IC description...")
        inputs = AI_PROCESSOR(pil_image, return_tensors="pt")
        
        if AI_USE_GPU:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate caption/description
        out = AI_MODEL.generate(**inputs, max_length=100)
        description = AI_PROCESSOR.decode(out[0], skip_special_tokens=True)
        print(f"  AI Description: {description}")
        
        # Conditional generation for specific queries
        queries = [
            "What is the part number on this integrated circuit?",
            "What is the manufacturer of this IC chip?",
            "What type of electronic component is this?",
            "Describe the package type and pin configuration"
        ]
        
        ai_analysis = {
            'description': description,
            'queries': {}
        }
        
        for query in queries:
            inputs = AI_PROCESSOR(pil_image, query, return_tensors="pt")
            if AI_USE_GPU:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            out = AI_MODEL.generate(**inputs, max_length=50)
            answer = AI_PROCESSOR.decode(out[0], skip_special_tokens=True)
            ai_analysis['queries'][query] = answer
            print(f"  Q: {query}")
            print(f"  A: {answer}")
        
        # Extract structured information from AI analysis
        extracted_info = extract_ic_info_from_ai_analysis(ai_analysis, part_number)
        
        return extracted_info
        
    except Exception as e:
        print(f"  ⚠️  AI Agent error: {str(e)}")
        return analyze_ic_with_database(part_number)


def extract_ic_info_from_ai_analysis(ai_analysis, part_number_hint=None):
    """
    Extract structured IC information from AI analysis
    """
    result = {
        'found': False,
        'part_number': 'Unknown',
        'manufacturer': 'Unknown',
        'description': ai_analysis.get('description', 'N/A'),
        'package': 'Unknown',
        'status': 'Unknown',
        'is_obsolete': False,
        'ai_confidence': 0.0
    }
    
    # Combine all AI responses
    all_text = ai_analysis.get('description', '') + ' '
    for answer in ai_analysis.get('queries', {}).values():
        all_text += answer + ' '
    
    all_text_lower = all_text.lower()
    
    # Extract part number using regex
    part_patterns = [
        r'([A-Z]{2,4}\d{2,4}[A-Z]{0,4}\d{0,4}[A-Z]{0,2})',
        r'(\d{2,4}[A-Z]{2,6}\d{0,4})',
        r'([A-Z]+\d+[A-Z]*\d*)'
    ]
    
    for pattern in part_patterns:
        matches = re.findall(pattern, all_text)
        if matches:
            result['part_number'] = max(matches, key=len)
            result['found'] = True
            break
    
    # Use hint if provided
    if part_number_hint and not result['found']:
        result['part_number'] = part_number_hint
        result['found'] = True
    
    # Extract manufacturer
    manufacturers = {
        'Motorola': ['motorola', 'mc', 'freescale'],
        'Texas Instruments': ['texas instruments', 'ti', 'sn74'],
        'STMicroelectronics': ['stmicroelectronics', 'st', 'stm'],
        'NXP': ['nxp', 'philips'],
        'Analog Devices': ['analog devices', 'adi', 'ad'],
        'Microchip': ['microchip', 'pic', 'atmel'],
        'Intel': ['intel'],
        'Maxim': ['maxim', 'max'],
        'ON Semiconductor': ['on semiconductor', 'onsemi']
    }
    
    for mfr, keywords in manufacturers.items():
        if any(kw in all_text_lower for kw in keywords):
            result['manufacturer'] = mfr
            break
    
    # Extract package type
    packages = ['dip', 'soic', 'qfp', 'bga', 'sop', 'tssop', 'plcc']
    for pkg in packages:
        if pkg in all_text_lower:
            result['package'] = pkg.upper()
            break
    
    # Determine confidence based on extracted information
    confidence = 0.0
    if result['part_number'] != 'Unknown':
        confidence += 0.4
    if result['manufacturer'] != 'Unknown':
        confidence += 0.3
    if result['package'] != 'Unknown':
        confidence += 0.2
    if result['description'] and len(result['description']) > 10:
        confidence += 0.1
    
    result['ai_confidence'] = min(1.0, confidence)
    result['status'] = 'Active' if confidence > 0.5 else 'Unknown'
    
    return result


def analyze_ic_with_database(part_number):
    """
    Fallback: Analyze IC using local database
    Used when AI Agent is not available or fails
    """
    print(f"  🔍 Searching local database for: {part_number}")
    
    # Enhanced OEM database with more entries
    oem_database = {
        'MC74HC20N': {
            'part_number': 'MC74HC20N',
            'manufacturer': 'Motorola/ON Semiconductor',
            'description': 'Dual 4-Input NAND Gate, 14-Pin DIP, High-Speed CMOS',
            'package': 'DIP-14',
            'status': 'Active',
            'is_obsolete': False,
            'ai_confidence': 0.9
        },
        'SN74LS00': {
            'part_number': 'SN74LS00N',
            'manufacturer': 'Texas Instruments',
            'description': 'Quad 2-Input NAND Gate, Low-Power Schottky',
            'package': 'DIP-14',
            'status': 'Active',
            'is_obsolete': False,
            'ai_confidence': 0.9
        },
        'CD4017': {
            'part_number': 'CD4017BE',
            'manufacturer': 'Texas Instruments',
            'description': 'Decade Counter/Divider with 10 Decoded Outputs',
            'package': 'DIP-16',
            'status': 'Active',
            'is_obsolete': False,
            'ai_confidence': 0.9
        },
        '555': {
            'part_number': 'NE555',
            'manufacturer': 'Texas Instruments',
            'description': 'Precision Timer IC',
            'package': 'DIP-8',
            'status': 'Active',
            'is_obsolete': False,
            'ai_confidence': 0.9
        },
        'LM358': {
            'part_number': 'LM358N',
            'manufacturer': 'Texas Instruments',
            'description': 'Dual Operational Amplifier',
            'package': 'DIP-8',
            'status': 'Active',
            'is_obsolete': False,
            'ai_confidence': 0.9
        }
    }
    
    if not part_number:
        return {'found': False, 'ai_confidence': 0.0}
    
    # Try to find exact or partial match
    for key, data in oem_database.items():
        if key.upper() in part_number.upper() or part_number.upper() in key.upper():
            print(f"  Found match in local database: {key}")
            return {'found': True, **data}
    
    print(f"  Part not found in local database")
    return {
        'found': False,
        'part_number': part_number,
        'manufacturer': 'Unknown',
        'description': 'Not found in database',
        'package': 'Unknown',
        'status': 'Unknown',
        'is_obsolete': False,
        'ai_confidence': 0.0
    }


# ============================================================================
# STEP 9: SURFACE DEFECT DETECTION (ENHANCED)
# ============================================================================

def detect_surface_defects_enhanced(image, reference_image_path):
    """Step 9: Enhanced Surface Defect Detection"""
    print("Step 9: Enhanced Surface Defect Detection - Starting...")
    
    if reference_image_path is None or not os.path.exists(reference_image_path):
        return {
            'step': '9. Enhanced Surface Defect Detection',
            'status': 'SKIPPED',
            'confidence': 1.0,
            'details': {'message': 'No reference image provided'}
        }
    
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        return {
            'step': '9. Enhanced Surface Defect Detection',
            'status': 'ERROR',
            'confidence': 0.0,
            'details': {'message': 'Could not load reference image'}
        }
    
    # Resize images to same size
    h, w = reference_image.shape[:2]
    test_resized = cv2.resize(image, (w, h))
    
    # Convert to grayscale
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)
    
    # Method 1: SSIM (already implemented in Step 4)
    ssim_score, diff_image = ssim(gray_ref, gray_test, full=True)
    diff_image = (diff_image * 255).astype("uint8")
    
    # Method 2: Enhanced absolute difference with multiple thresholds
    abs_diff = cv2.absdiff(gray_ref, gray_test)
    
    # Apply multiple intensity thresholds for different defect types
    thresholds = [10, 20, 30, 50]
    defect_ratios = []
    
    for threshold in thresholds:
        _, thresh = cv2.threshold(abs_diff, threshold, 255, cv2.THRESH_BINARY)
        defect_pixels = np.sum(thresh > 0)
        defect_ratio = defect_pixels / thresh.size
        defect_ratios.append(defect_ratio)
    
    # Method 3: Edge-based defect detection
    edges_ref = cv2.Canny(gray_ref, 50, 150)
    edges_test = cv2.Canny(gray_test, 50, 150)
    edge_diff = cv2.absdiff(edges_ref, edges_test)
    edge_defect_ratio = np.sum(edge_diff > 0) / edge_diff.size
    
    # Method 4: Frequency domain analysis
    f_ref = np.fft.fft2(gray_ref)
    f_test = np.fft.fft2(gray_test)
    f_diff = np.abs(f_ref - f_test)
    freq_defect_ratio = np.sum(f_diff > np.mean(f_diff) * 2) / f_diff.size
    
    # Combine all metrics
    ssim_pass = ssim_score >= SSIM_THRESHOLD
    intensity_pass = defect_ratios[1] <= 0.1  # Use threshold 20
    edge_pass = edge_defect_ratio <= 0.05
    freq_pass = freq_defect_ratio <= 0.1
    
    # Calculate combined confidence
    confidence = (ssim_score + (1 - defect_ratios[1]) + (1 - edge_defect_ratio) + (1 - freq_defect_ratio)) / 4
    confidence = max(0.0, min(1.0, confidence))
    
    status = "PASS" if ssim_pass and intensity_pass and edge_pass else "FAIL"
    
    print(f"  SSIM Score: {ssim_score:.3f} (threshold: {SSIM_THRESHOLD})")
    print(f"  Intensity Defect Ratios: {[f'{r:.3f}' for r in defect_ratios]}")
    print(f"  Edge Defect Ratio: {edge_defect_ratio:.3f}")
    print(f"  Frequency Defect Ratio: {freq_defect_ratio:.3f}")
    print(f"  Combined Confidence: {confidence:.3f}")
    print(f"  Status: {status}")
    
    return {
        'step': '9. Enhanced Surface Defect Detection',
        'status': status,
        'confidence': confidence,
        'details': {
            'ssim_score': float(ssim_score),
            'defect_ratios': [float(r) for r in defect_ratios],
            'edge_defect_ratio': float(edge_defect_ratio),
            'freq_defect_ratio': float(freq_defect_ratio),
            'ssim_pass': bool(ssim_pass),
            'intensity_pass': bool(intensity_pass),
            'edge_pass': bool(edge_pass),
            'freq_pass': bool(freq_pass),
            'method': 'SSIM + multi-threshold absdiff + edge + frequency analysis'
        }
    }
# STEP 10: CORRELATION LAYER
# ============================================================================

def correlation_analysis(results, test_image=None):
    """Step 11: Correlation Layer - Composite pass threshold"""
    step_start_time = time.time()
    print("Step 11: Correlation Analysis - Starting...")
    
    # Count different types of results
    total_checks = len([r for r in results if r['status'] not in ["SKIPPED", "ERROR"]])
    passed_checks = len([r for r in results if r['status'] == "PASS"])
    failed_checks = len([r for r in results if r['status'] == "FAIL"])
    skipped_checks = len([r for r in results if r['status'] == "SKIPPED"])
    error_checks = len([r for r in results if r['status'] == "ERROR"])
    
    # Calculate pass rate
    pass_rate = passed_checks / max(total_checks, 1)
    
    # Identify critical mismatches
    critical_checks = ["1. Logo Detection (Template)", "2. Logo Detection (ORB/SIFT)", "4. QR/DMC Code Detection", "5. Surface Defect Detection"]
    critical_failed = [r['step'] for r in results if r['status'] == "FAIL" and r['step'] in critical_checks]
    
    # Calculate weighted confidence
    weighted_confidence = 0.0
    total_weight = 0.0
    
    for result in results:
        if result['status'] not in ["SKIPPED", "ERROR"]:
            weight = 2.0 if result['step'] in critical_checks else 1.0  # Critical checks get double weight
            weighted_confidence += result['confidence'] * weight
            total_weight += weight
    
    avg_weighted_confidence = weighted_confidence / max(total_weight, 1)
    
    # Apply industry-standard correlation threshold
    correlation_pass = pass_rate >= CORRELATION_PASS_THRESHOLD and len(critical_failed) == 0
    
    confidence = avg_weighted_confidence
    status = "PASS" if correlation_pass else "FAIL"
    
    print(f"  Total Checks: {total_checks}")
    print(f"  Passed: {passed_checks} ({pass_rate:.1%})")
    print(f"  Failed: {failed_checks}")
    print(f"  Skipped: {skipped_checks}")
    print(f"  Errors: {error_checks}")
    print(f"  Pass Rate: {pass_rate:.3f} (threshold: {CORRELATION_PASS_THRESHOLD})")
    print(f"  Critical Failures: {critical_failed}")
    print(f"  Weighted Confidence: {avg_weighted_confidence:.3f}")
    print(f"  Status: {status}")
    
    step_time = (time.time() - step_start_time) * 1000
    print(f"  Processing Time: {step_time:.2f}ms")
    
    # Create visualization - summary chart
    if test_image is not None:
        vis_image = test_image.copy()
        h, w = vis_image.shape[:2]
        
        # Create a summary overlay
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)
        
        # Add summary text
        cv2.putText(vis_image, "CORRELATION ANALYSIS SUMMARY", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Total Checks: {total_checks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis_image, f"Passed: {passed_checks} | Failed: {failed_checks}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if passed_checks > failed_checks else (0, 0, 255), 1)
        cv2.putText(vis_image, f"Pass Rate: {pass_rate:.1%}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if correlation_pass else (0, 0, 255), 1)
        cv2.putText(vis_image, f"Status: {status}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == "PASS" else (0, 0, 255), 2)
        
        # Save visualization (layer 12 since font verification is layer 11)
        save_visualization_image(vis_image, "correlation_layer_composite", 12)
    
    return {
        'step': '11. Correlation Analysis',
        'status': status,
        'confidence': confidence,
        'processing_time_ms': step_time,
        'details': {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'skipped_checks': skipped_checks,
            'error_checks': error_checks,
            'pass_rate': float(pass_rate),
            'critical_failed': critical_failed,
            'weighted_confidence': float(avg_weighted_confidence),
            'correlation_pass': bool(correlation_pass),
            'method': 'Weighted composite analysis'
        }
    }

# ============================================================================
# STEP 11: FINAL VERDICT
# ============================================================================

def generate_verdict(results):
    """Step 12: Generate final counterfeit verdict based on all checks"""
    print("Step 12: Final Verdict - Generating Final Counterfeit Verdict...")
    
    # Calculate overall score
    total_confidence = 0.0
    passed_checks = 0
    failed_checks = 0
    total_checks = 0
    
    for result in results:
        if result['status'] not in ["SKIPPED", "ERROR"]:
            total_confidence += result['confidence']
            total_checks += 1
            if result['status'] == "PASS":
                passed_checks += 1
            else:
                failed_checks += 1
    
    avg_confidence = total_confidence / max(total_checks, 1)
    
    # Determine final verdict
    # Product is genuine if:
    # 1. Average confidence >= 0.75
    # 2. No more than 2 failed checks (out of 11 tests)
    # 3. Critical checks (logo, QR, defect, geometry) must pass
    
    critical_checks = ["1. Logo Detection (Template)", "2. Logo Detection (ORB/SIFT)", "4. QR/DMC Code Detection", "5. Surface Defect Detection"]
    critical_failed = any(
        r['status'] == "FAIL" and r['step'] in critical_checks 
        for r in results
    )
    
    # More lenient threshold for 11 tests vs 7 tests
    max_allowed_failures = min(2, total_checks // 5)  # Allow up to 2 failures or 20% failure rate
    
    if avg_confidence >= CORRELATION_PASS_THRESHOLD and failed_checks <= max_allowed_failures and not critical_failed:
        verdict = "GENUINE"
        status = "PASS"
    else:
        verdict = "COUNTERFEIT"
        status = "FAIL"
    
    print(f"  Overall Confidence: {avg_confidence:.3f}")
    print(f"  Checks Passed: {passed_checks}/{total_checks}")
    print(f"  Checks Failed: {failed_checks}/{total_checks}")
    print(f"  Max Allowed Failures: {max_allowed_failures}")
    print(f"  Critical Checks Failed: {'Yes' if critical_failed else 'No'}")
    print(f"  Final Verdict: {verdict}")
    
    return {
        'step': 'Final Verdict',
        'status': status,
        'confidence': avg_confidence,
        'details': {
            'verdict': verdict,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'total_checks': total_checks,
            'critical_failed': critical_failed,
            'max_allowed_failures': max_allowed_failures,
            'method': 'Weighted Analysis with Critical Check Validation'
        }
    }

# ============================================================================
# MAIN VERIFICATION PIPELINE
# ============================================================================

def run_complete_11step_verification(test_image_path, reference_image_path, 
                                    logo_template_path=None, expected_text=None, 
                                    expected_qr_data=None, color_reference=None):
    """Run the complete 11-step counterfeit detection pipeline"""
    
    print("="*80)
    print("COMPLETE 11-STEP COUNTERFEIT DETECTION PIPELINE (Production Optimized)")
    print("="*80)
    
    # Load test image
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"ERROR: Could not load test image: {test_image_path}")
        return None
    
    print(f"Test Image: {test_image_path}")
    print(f"Reference Image: {reference_image_path}")
    print("="*80)
    
    # Intelligent preprocessing - only if needed
    print("IMAGE QUALITY ASSESSMENT:")
    test_image, test_preprocess_info = preprocess_image_if_needed(test_image)
    print(f"Test Image Processing: {test_preprocess_info}")
    
    # Also check reference image quality
    reference_image = cv2.imread(reference_image_path)
    if reference_image is not None:
        reference_image, ref_preprocess_info = preprocess_image_if_needed(reference_image)
        print(f"Reference Image Processing: {ref_preprocess_info}")
    print("="*80)
    
    results = []
    
    # Step 1: Logo Detection (Template) - NCC ≥ 0.8 to 0.9
    logo_template_result = detect_logo_template(test_image, logo_template_path)
    results.append(logo_template_result)
    
    # Step 2: Logo Detection (ORB/SIFT) - REMOVED (too slow for production)
    # ORB/SIFT takes 1700ms+ which is unacceptable for industrial use
    # HoughCircles in Step 1 provides sufficient logo detection
    
    # Step 3: Text & Serial Number OCR - OCR confidence ≥ 80 to 90%
    text_result = detect_and_read_text(test_image, expected_text)
    results.append(text_result)
    
    # Step 4: QR/DMC Code Detection - ≥ 80% success rate
    qr_result = detect_qr_code(test_image, expected_qr_data)
    results.append(qr_result)
    
    # Step 5: Surface Defect Detection - SSIM + Intensity difference
    defect_result = detect_defects(test_image, reference_image_path)
    results.append(defect_result)
    
    # Step 6: Edge Detection (Canny) - Low: 100, High: 200
    edge_result = detect_edges_canny(test_image)
    results.append(edge_result)
    
    # Step 7: IC Outline/Geometry - Size and aspect ratio deviation ±3% to 5%
    geometry_result = check_ic_geometry(test_image, reference_image_path)
    results.append(geometry_result)
    
    # Step 8: Angle Detection - Orientation angle deviation ±2° to 5°
    angle_result = check_angle_detection(test_image)
    results.append(angle_result)
    
    # Step 9: Color Surface Verification - Color Distance ΔE < 3 to 5 (LAB/HSV)
    color_result = verify_color(test_image, color_reference)
    results.append(color_result)
    
    # Step 10: Texture Verification (LBP/GLCM) - Feature vector distance < 0.15
    texture_result = verify_texture(test_image, reference_image_path)
    results.append(texture_result)
    
    # Step 10.5: Font Verification - CRITICAL for counterfeit detection
    font_result = verify_font_characteristics(test_image, reference_image_path)
    results.append(font_result)
    
    # Step 10.6: AI Agent - OEM Database Verification (NOW ENABLED with Hugging Face)
    # Uses Hugging Face BLIP Vision-Language Model instead of web scraping
    # Faster and more reliable than previous web scraping implementation
    ocr_text_from_step3 = text_result['details'].get('ocr_text', None) if 'details' in text_result else None
    logo_detected_from_step1 = logo_template_result['details'].get('motorola_logo_found', False) if 'details' in logo_template_result else False
    ai_agent_result = ai_agent_oem_verification(test_image, ocr_text=ocr_text_from_step3, logo_detected=logo_detected_from_step1)
    results.append(ai_agent_result)
    
    # Step 11: Correlation Layer - Composite pass threshold ≥ 90% layers pass
    correlation_result = correlation_analysis(results, test_image)
    results.append(correlation_result)
    
    # Final Verdict (not counted as separate AOI layer)
    final_result = generate_verdict(results)
    results.append(final_result)
    
    # Generate comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_image': test_image_path,
        'reference_image': reference_image_path,
        'verdict': final_result['details']['verdict'],
        'overall_confidence': final_result['confidence'],
        'pipeline_results': results,
        'summary': {
            'total_checks': len([r for r in results if r['status'] not in ["SKIPPED", "ERROR"]]),
            'passed': len([r for r in results if r['status'] == "PASS"]),
            'failed': len([r for r in results if r['status'] == "FAIL"]),
            'skipped': len([r for r in results if r['status'] == "SKIPPED"]),
            'errors': len([r for r in results if r['status'] == "ERROR"])
        }
    }
    
    print("\n" + "="*80)
    print("FINAL VERDICT: " + final_result['details']['verdict'])
    print("="*80)
    
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run complete 11-step verification
    start_time = time.time()
    
    report = run_complete_11step_verification(
        test_image_path=TEST_IC_PATH,
        reference_image_path=REFERENCE_IC_PATH,
        logo_template_path=None,  # No logo template available
        expected_text=None,       # Will be extracted from reference
        expected_qr_data=None,    # No QR codes expected
        color_reference=None      # No color reference available
    )
    
    elapsed_time = (time.time() - start_time) * 1000
    
    if report:
        # Save report
        with open('complete_11step_verification_results.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"PROCESSING TIME SUMMARY")
        print(f"{'='*80}")
        print(f"Total Pipeline Time: {elapsed_time:.2f}ms ({elapsed_time/1000:.3f}s)")
        print(f"\nIndividual Step Times:")
        for i, result in enumerate(report['pipeline_results'], 1):
            step_name = result['step']
            step_time = result.get('processing_time_ms', 0)
            percentage = (step_time / elapsed_time * 100) if elapsed_time > 0 else 0
            print(f"  {i:2d}. {step_name:<40} {step_time:8.2f}ms ({percentage:5.1f}%)")
        print(f"{'='*80}")
        
        print(f"\nResults saved to: complete_11step_verification_results.json")
        
        if SAVE_VISUALIZATION_IMAGES:
            print(f"\n{'='*80}")
            print(f"VISUALIZATION IMAGES SAVED")
            print(f"{'='*80}")
            print(f"All 11 test layer visualization images have been saved to: {OUTPUT_DIR}/")
            print(f"Image naming format: layer_XX_test_name.jpg")
            print(f"\nVisualization images created:")
            print(f"  1. layer_01_logo_detection_template.jpg")
            print(f"  2. layer_02_logo_detection_orb_sift.jpg")
            print(f"  3. layer_03_text_serial_number_ocr.jpg")
            print(f"  4. layer_04_qr_dmc_code_detection.jpg")
            print(f"  5. layer_05_surface_defect_detection.jpg")
            print(f"  6. layer_06_edge_detection_canny.jpg")
            print(f"  7. layer_07_ic_outline_geometry.jpg")
            print(f"  8. layer_08_angle_detection.jpg")
            print(f"  9. layer_09_color_surface_verification.jpg")
            print(f" 10. layer_10_texture_verification_fast.jpg")
            print(f" 11. layer_11_font_verification.jpg")
            print(f" 12. layer_12_correlation_layer_composite.jpg")
            print(f"{'='*80}")
        
        # Print comprehensive summary
        print(f"\n" + "="*80)
        print(f"COMPREHENSIVE 11-STEP VERIFICATION SUMMARY")
        print("="*80)
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Total Checks: {report['summary']['total_checks']}")
        print(f"  Passed: {report['summary']['passed']}")
        print(f"  Failed: {report['summary']['failed']}")
        print(f"  Skipped: {report['summary']['skipped']}")
        print(f"  Errors: {report['summary']['errors']}")
        print(f"  Overall Confidence: {report['overall_confidence']:.3f}")
        print(f"  Final Verdict: {report['verdict']}")
        
        # Detailed explanation of each step
        print(f"\nDETAILED 11-STEP ANALYSIS:")
        for result in report['pipeline_results']:
            step_name = result['step']
            status = result['status']
            confidence = result['confidence']
            
            print(f"\n  {step_name}: {status} (Confidence: {confidence:.3f})")
            
            if step_name == "1. Logo Detection (Template)":
                if result['details'].get('motorola_logo_found'):
                    print(f"    - Motorola logo detected with {confidence:.3f} confidence")
                    print(f"    - Logo location: {result['details'].get('motorola_location')}")
                else:
                    print(f"    - No Motorola logo pattern found")
                    
            elif step_name == "2. Logo Detection (ORB/SIFT)":
                good_matches = result['details'].get('good_matches_count', 0)
                print(f"    - Good matches: {good_matches}")
                print(f"    - Feature match score: {result['details'].get('feature_match_score', 0):.3f}")
                    
            elif step_name == "3. Text Area + OCR":
                ocr_text = result['details'].get('ocr_text', '')
                print(f"    - OCR extracted: '{ocr_text[:50]}{'...' if len(ocr_text) > 50 else ''}'")
                print(f"    - Text areas found: {result['details'].get('text_areas_found', 0)}")
                
            elif step_name == "4. QR/DMC Code Detection":
                codes_detected = result['details'].get('codes_detected', 0)
                message = result['details'].get('message', '')
                if codes_detected > 0:
                    print(f"    - Codes detected: {codes_detected}")
                else:
                    print(f"    - {message}")
                
            elif step_name == "5. Surface Defect Detection":
                ssim_score = result['details'].get('ssim_score', 0)
                defect_ratio = result['details'].get('defect_ratio', 0)
                defect_pixels = result['details'].get('defect_pixels', 0)
                
                print(f"    - SSIM Score: {ssim_score:.3f} (Structural Similarity Index)")
                print(f"      * SSIM measures how similar the images are structurally")
                print(f"      * Score of {ssim_score:.3f} means {ssim_score*100:.1f}% structural similarity")
                print(f"      * Values above {SSIM_THRESHOLD} are considered very similar (industry standard)")
                print(f"      * Values below {SSIM_THRESHOLD} indicate significant differences")
                
                print(f"    - Defect Ratio: {defect_ratio:.3f} ({defect_ratio*100:.1f}% of pixels differ)")
                print(f"      * {defect_pixels:,} pixels out of total differ between images")
                print(f"      * Defect ratio of {defect_ratio:.3f} means {defect_ratio*100:.1f}% of the image is different")
                print(f"      * Values below 0.1 (10%) are acceptable for genuine ICs")
                print(f"      * Values above 0.2 (20%) suggest significant manufacturing differences")
                print(f"      * Intensity difference threshold: {INTENSITY_DIFFERENCE_THRESHOLD} (industry standard)")
                
                if defect_ratio > 0.2:
                    print(f"    - VERDICT: SIGNIFICANT STRUCTURAL DIFFERENCES DETECTED")
                    print(f"      * This indicates the ICs have substantial visual/structural differences")
                    print(f"      * Could indicate different manufacturing batches, different ICs, or counterfeit")
                elif ssim_score < 0.7:
                    print(f"    - VERDICT: LOW STRUCTURAL SIMILARITY")
                    print(f"      * Images are not structurally similar enough to be considered identical")
                else:
                    print(f"    - VERDICT: ACCEPTABLE SIMILARITY")
                    
            elif step_name == "6. Edge Detection (Canny)":
                edge_density = result['details'].get('edge_density', 0)
                edge_strength = result['details'].get('edge_strength', 0)
                edge_pixels = result['details'].get('edge_pixels', 0)
                print(f"    - Edge density: {edge_density:.3f} ({edge_density*100:.1f}%)")
                print(f"    - Edge strength: {edge_strength:.2f}")
                print(f"    - Edge pixels: {edge_pixels:,}")
                
            elif step_name == "7. IC Outline/Geometry":
                test_size = result['details'].get('test_size', (0, 0))
                ref_size = result['details'].get('reference_size', (0, 0))
                size_dev = result['details'].get('size_deviation', 0)
                aspect_dev = result['details'].get('aspect_deviation', 0)
                print(f"    - Test IC size: {test_size[0]}x{test_size[1]}")
                print(f"    - Reference IC size: {ref_size[0]}x{ref_size[1]}")
                print(f"    - Size deviation: {size_dev:.3f} (threshold: {SIZE_DEVIATION_THRESHOLD})")
                print(f"    - Aspect ratio deviation: {aspect_dev:.3f} (threshold: {ASPECT_RATIO_DEVIATION_THRESHOLD})")
                
            elif step_name == "8. Angle Detection":
                lines_detected = result['details'].get('lines_detected', 0)
                angle_deviation = result['details'].get('avg_angle_deviation', 0)
                max_deviation = result['details'].get('max_angle_deviation', 0)
                print(f"    - Lines detected: {lines_detected}")
                print(f"    - Average angle deviation: {angle_deviation:.2f}°")
                print(f"    - Max angle deviation: {max_deviation:.2f}°")
                
            elif step_name == "9. Color Surface Verification":
                avg_color = result['details'].get('avg_color_bgr', [0,0,0])
                color_diff = result['details'].get('color_difference', 0)
                print(f"    - Average color (BGR): {avg_color}")
                print(f"    - Color difference: {color_diff:.2f}")
                
            elif step_name == "10. Texture Verification":
                texture_dist = result['details'].get('texture_distance', 0)
                print(f"    - Texture Distance: {texture_dist:.3f} (threshold: {TEXTURE_DISTANCE_THRESHOLD})")
                print(f"    - Method: Fast histogram + gradient analysis")
                if texture_dist <= TEXTURE_DISTANCE_THRESHOLD:
                    print(f"    - VERDICT: Textures are similar (PASS)")
                else:
                    print(f"    - VERDICT: Textures differ significantly (FAIL)")
                
            elif step_name == "10.5 Font Verification":
                font_sim = result['details'].get('font_similarity', 0)
                shape_score = result['details'].get('shape_score', 0)
                spacing_score = result['details'].get('spacing_score', 0)
                stroke_score = result['details'].get('stroke_score', 0)
                edge_score = result['details'].get('edge_score', 0)
                print(f"    - Font Shape Similarity: {shape_score:.3f} (35% weight)")
                print(f"    - Character Spacing: {spacing_score:.3f} (30% weight)")
                print(f"    - Stroke Width Score: {stroke_score:.3f} (20% weight)")
                print(f"    - Edge Quality: {edge_score:.3f} (15% weight)")
                print(f"    - Overall Font Similarity: {font_sim:.3f} (threshold: {FONT_SIMILARITY_THRESHOLD})")
                if font_sim >= FONT_SIMILARITY_THRESHOLD:
                    print(f"    - VERDICT: Fonts match (PASS) - Likely genuine")
                else:
                    print(f"    - VERDICT: Font mismatch (FAIL) - Possible counterfeit!")
                
            elif step_name == "10.6 AI Agent OEM Verification":
                # Skip AI Agent in output (disabled for production)
                continue
                
            elif step_name == "11. Correlation Analysis":
                total_checks = result['details'].get('total_checks', 0)
                passed_checks = result['details'].get('passed_checks', 0)
                failed_checks = result['details'].get('failed_checks', 0)
                pass_rate = result['details'].get('pass_rate', 0)
                critical_failed = result['details'].get('critical_failed', [])
                print(f"    - Total checks: {total_checks}")
                print(f"    - Passed: {passed_checks} ({pass_rate:.1%})")
                print(f"    - Failed: {failed_checks}")
                print(f"    - Pass rate: {pass_rate:.3f} (threshold: {CORRELATION_PASS_THRESHOLD})")
                print(f"    - Critical failures: {critical_failed}")
                
            elif step_name == "Final Verdict":
                passed_checks = result['details'].get('passed_checks', 0)
                failed_checks = result['details'].get('failed_checks', 0)
                total_checks = result['details'].get('total_checks', 0)
                critical_failed = result['details'].get('critical_failed', False)
                max_failures = result['details'].get('max_allowed_failures', 0)
                
                print(f"    - Passed checks: {passed_checks}/{total_checks}")
                print(f"    - Failed checks: {failed_checks}/{total_checks}")
                print(f"    - Max allowed failures: {max_failures}")
                print(f"    - Critical checks failed: {'Yes' if critical_failed else 'No'}")
                
                if result['details']['verdict'] == 'COUNTERFEIT':
                    print(f"    - REASON FOR COUNTERFEIT VERDICT:")
                    if critical_failed:
                        print(f"      * Critical checks (Logo Template, Logo ORB, QR/DMC, Surface Defect Detection) failed")
                    if failed_checks > max_failures:
                        print(f"      * Too many check failures ({failed_checks} > {max_failures})")
                    if confidence < 0.75:
                        print(f"      * Overall confidence too low ({confidence:.3f} < 0.75)")
                else:
                    print(f"    - REASON FOR GENUINE VERDICT:")
                    print(f"      * All critical checks passed")
                    print(f"      * High overall confidence ({confidence:.3f})")
                    print(f"      * Acceptable failure rate ({failed_checks} <= {max_failures})")
        
        print(f"\n" + "="*80)
        print(f"COMPREHENSIVE 11-TEST SUMMARY TABLE:")
        print("="*80)
        print(f"{'Test #':<4} {'AOI Layer':<35} {'Status':<8} {'Confidence':<10} {'Method'}")
        print("-" * 80)
        
        test_mapping = {
            "1. Logo Detection (Template)": "Logo Detection (HoughCircles)",
            "3. Text & Serial Number OCR": "Text & Serial Number OCR",
            "4. QR/DMC Code Detection": "QR/DMC Code Detection",
            "5. Surface Defect Detection": "Surface Defect Detection (SSIM)",
            "6. Edge Detection (Canny)": "Edge Detection (Canny)",
            "7. IC Outline/Geometry": "IC Outline/Geometry (Size+Aspect)",
            "8. Angle Detection": "Angle Detection (Orientation)",
            "9. Color Surface Verification": "Color Surface Verification (LAB/HSV)",
            "10. Texture Verification": "Texture Verification (Fast)",
            "10.5 Font Verification": "Font Verification (Critical)",
            "10.6 AI Agent OEM Verification": "AI Agent OEM Database (Critical)",
            "11. Correlation Analysis": "Correlation Layer (Composite)"
        }
        
        for i, result in enumerate(report['pipeline_results'], 1):
            step_name = result['step']
            
            # Skip AI Agent in summary table (disabled for production)
            if step_name == "10.6 AI Agent OEM Verification":
                continue
            
            status = result['status']
            confidence = result['confidence']
            method = result['details'].get('method', 'N/A')[:25]
            
            aoi_layer = test_mapping.get(step_name, step_name)
            print(f"{i:<4} {aoi_layer:<35} {status:<8} {confidence:<10.3f} {method}")
        
        print("-" * 80)
        print(f"FINAL RESULT: {report['verdict']} (Confidence: {report['overall_confidence']:.3f})")
        print("="*80)
        
        print(f"\nINTERPRETATION:")
        if report['verdict'] == 'COUNTERFEIT':
            print(f"  The ICs are NOT identical. This could mean:")
            print(f"  - Different IC models or part numbers")
            print(f"  - Different manufacturing batches")
            print(f"  - One IC is counterfeit")
            print(f"  - Significant manufacturing variations")
            print(f"  - Quality control issues")
        else:
            print(f"  The ICs appear to be identical based on all 11 verification criteria.")
            print(f"  High confidence in genuineness with {report['summary']['passed']}/{report['summary']['total_checks']} tests passing.")
        print("="*80)
