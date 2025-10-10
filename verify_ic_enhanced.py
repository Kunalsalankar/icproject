"""
Enhanced IC Verification with Image Preprocessing
Improves OCR accuracy by preprocessing the image
"""

from counterfeit_detection_agent import CounterfeitDetectionAgent
import cv2
import numpy as np
import pytesseract
import re

# IC Configuration
IC_PART_NUMBER = 'SN74LS266N'
IC_MANUFACTURER_MARK = 'HLF'
IC_DATE_CODE = '20A1'

test_image_path = 'test_images/product_to_verify.jpg'


def preprocess_for_ocr(image):
    """
    Enhance image for better OCR results
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Threshold - try multiple methods
    _, thresh1 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    thresh3 = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Save preprocessed images for debugging
    cv2.imwrite('debug_enhanced.jpg', enhanced)
    cv2.imwrite('debug_thresh1.jpg', thresh1)
    cv2.imwrite('debug_thresh2.jpg', thresh2)
    cv2.imwrite('debug_thresh3.jpg', thresh3)
    
    return [enhanced, thresh1, thresh2, thresh3]


def extract_text_multiple_methods(image):
    """
    Try multiple OCR configurations
    """
    preprocessed_images = preprocess_for_ocr(image)
    
    # Different Tesseract PSM modes
    psm_modes = [
        '--psm 6',  # Uniform block of text
        '--psm 7',  # Single line
        '--psm 11', # Sparse text
        '--psm 13', # Raw line
    ]
    
    all_texts = []
    
    for img in preprocessed_images:
        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(img, config=psm)
                all_texts.append(text.strip())
            except:
                pass
    
    return all_texts


def analyze_ic_markings(ocr_texts):
    """
    Analyze multiple OCR results to find IC markings
    """
    markings = {
        'part_number': None,
        'manufacturer_mark': None,
        'date_code': None
    }
    
    for text in ocr_texts:
        if not text:
            continue
            
        text_upper = text.upper().replace('\n', ' ').replace('\r', ' ')
        
        # Extract part number
        if not markings['part_number']:
            part_pattern = r'SN\d{2}[A-Z]{2,4}\d{2,4}[A-Z]?'
            part_match = re.search(part_pattern, text_upper)
            if part_match:
                markings['part_number'] = part_match.group()
        
        # Extract manufacturer
        if not markings['manufacturer_mark']:
            if 'HLF' in text_upper or 'HL' in text_upper:
                markings['manufacturer_mark'] = 'HLF'
        
        # Extract date code
        if not markings['date_code']:
            date_pattern = r'\d{2}[A-Z]\d'
            date_match = re.search(date_pattern, text_upper)
            if date_match:
                markings['date_code'] = date_match.group()
    
    return markings


print("="*70)
print("ENHANCED IC VERIFICATION WITH IMAGE PREPROCESSING")
print("="*70)
print(f"\n📋 Expected IC Markings:")
print(f"   Part Number:      {IC_PART_NUMBER}")
print(f"   Manufacturer:     {IC_MANUFACTURER_MARK}")
print(f"   Date/Batch Code:  {IC_DATE_CODE}")
print(f"\n📷 Testing Image: {test_image_path}")
print("="*70)

# Load image
image = cv2.imread(test_image_path)

if image is None:
    print(f"\n❌ Error: Could not load image: {test_image_path}")
    print("Please check the file path and ensure the image exists.")
else:
    print("\n🔄 Preprocessing image for better OCR...")
    
    # Extract text using multiple methods
    ocr_results = extract_text_multiple_methods(image)
    
    print(f"✓ Tried {len(ocr_results)} different OCR configurations")
    print("\n📝 All OCR Results:")
    for i, text in enumerate(ocr_results, 1):
        if text:
            print(f"   {i}. '{text[:50]}...'")
    
    # Analyze markings
    detected_markings = analyze_ic_markings(ocr_results)
    
    print("\n" + "="*70)
    print("DETECTED IC MARKINGS")
    print("="*70)
    print(f"🔍 Extracted Markings:")
    print(f"   Part Number:      {detected_markings['part_number'] or 'NOT DETECTED'}")
    print(f"   Manufacturer:     {detected_markings['manufacturer_mark'] or 'NOT DETECTED'}")
    print(f"   Date/Batch Code:  {detected_markings['date_code'] or 'NOT DETECTED'}")
    
    # Verification
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    
    part_match = detected_markings['part_number'] == IC_PART_NUMBER
    mfg_match = detected_markings['manufacturer_mark'] == IC_MANUFACTURER_MARK
    date_match = detected_markings['date_code'] == IC_DATE_CODE
    
    part_status = "✅ PASS" if part_match else "❌ FAIL"
    print(f"{part_status} Part Number: {IC_PART_NUMBER}")
    if detected_markings['part_number']:
        print(f"     Detected: {detected_markings['part_number']}")
    
    if IC_MANUFACTURER_MARK:
        mfg_status = "✅ PASS" if mfg_match else "⚠️  WARN"
        print(f"{mfg_status} Manufacturer: {IC_MANUFACTURER_MARK}")
        if detected_markings['manufacturer_mark']:
            print(f"     Detected: {detected_markings['manufacturer_mark']}")
    
    if IC_DATE_CODE:
        date_status = "✅ PASS" if date_match else "⚠️  INFO"
        print(f"{date_status} Date/Batch Code: {IC_DATE_CODE}")
        if detected_markings['date_code']:
            print(f"     Detected: {detected_markings['date_code']}")
    
    # Final verdict
    print("\n" + "="*70)
    final_verdict = "GENUINE" if part_match else "COUNTERFEIT/UNVERIFIED"
    print(f"🎯 FINAL VERDICT: {final_verdict}")
    print("="*70)
    
    if part_match:
        print("\n✅ IC part number matches! This appears to be genuine.")
        print("   IC Type: SN74LS266N - Quad 2-input XNOR Gate (74LS series)")
    else:
        print("\n❌ IC part number does NOT match or could not be detected!")
        print("\n💡 Troubleshooting:")
        print("   1. Check debug images: debug_enhanced.jpg, debug_thresh1.jpg, etc.")
        print("   2. Take a better photo (closer, better lighting, no glare)")
        print("   3. Ensure IC text is clearly visible and in focus")
        print("   4. Try different angles/lighting conditions")
    
    print("\n📁 Debug images saved:")
    print("   - debug_enhanced.jpg (contrast enhanced)")
    print("   - debug_thresh1.jpg (threshold method 1)")
    print("   - debug_thresh2.jpg (threshold method 2)")
    print("   - debug_thresh3.jpg (threshold method 3)")
