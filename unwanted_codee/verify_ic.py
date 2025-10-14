"""
IC Verification Script with Detailed Marking Analysis

IC Marking Interpretation:
- SN74LS266N: IC part number (Quad 2-input Exclusive NOR Gate with open-collector outputs)
  - SN: Texas Instruments prefix
  - 74LS: 74LS series TTL logic IC family
  - 266: Specific IC type (Quad XNOR Gate)
  - N: Package type (DIP - Dual In-line Package)
  
- HLF®: Manufacturer's logo/marking (indicates specific manufacturer or foundry)

- 20A1: Batch/Date code
  - 20: Year (2020) or Week 20
  - A1: Manufacturing lot identifier
"""

import time
from counterfeit_detection_agent import CounterfeitDetectionAgent
import cv2
import numpy as np
import pytesseract
import re

# Initialize the agent
agent = CounterfeitDetectionAgent()

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================

# Debug mode - set to False in production to skip debug image saving (saves 5-10ms per IC)
DEBUG_MODE = True

# ROI (Region of Interest) - crop to IC marking area for faster processing
# Format: (x, y, width, height) or None for full image
# Example: ROI = (100, 50, 800, 600) crops to 800x600 starting at (100,50)
ROI = None  # Set to crop coordinates or None for full image

# Target image height for OCR (smaller = faster, but keep readable)
# Recommended: 300-500 pixels for IC text
TARGET_OCR_HEIGHT = 400  # Downscale to this height (maintains aspect ratio)

# Conditional preprocessing - only apply if image quality is poor
AUTO_ENHANCE = True  # Automatically detect if enhancement is needed

# Tesseract optimization
TESSERACT_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Only alphanumeric
OPTIMIZED_PSM_MODES = ['--psm 6', '--psm 11']  # Reduced from 4 to 2 modes

# Early exit on high confidence
EARLY_EXIT_CONFIDENCE = True  # Stop processing if confident match found


def crop_roi(image, roi=None):
    """
    Crop image to Region of Interest (ROI) for faster processing
    """
    if roi is None:
        return image
    x, y, w, h = roi
    return image[y:y+h, x:x+w]


def downscale_image(image, target_height=400):
    """
    Downscale image to target height while maintaining aspect ratio
    Smaller images = faster OCR (2-5x speedup)
    """
    h, w = image.shape[:2]
    if h <= target_height:
        return image
    
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)


def assess_image_quality(gray):
    """
    Assess if image needs enhancement (blur detection, contrast check)
    Returns: True if enhancement needed, False if image is already good
    """
    # Check contrast
    contrast = gray.std()
    
    # Check sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # If low contrast or blurry, enhancement needed
    needs_enhancement = contrast < 50 or laplacian_var < 100
    
    return needs_enhancement


def preprocess_image_for_ocr(image_path, debug=True, auto_enhance=True, roi=None, target_height=400):
    """
    OPTIMIZED: Preprocess image for better OCR using conditional enhancement
    
    Args:
        image_path: Path to image
        debug: Save debug images (disable in production for 5-10ms speedup)
        auto_enhance: Only enhance if image quality is poor
        roi: Region of interest (x, y, w, h) to crop
        target_height: Downscale to this height for faster OCR
    
    Returns: Path to preprocessed image
    """
    start_time = time.time()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # OPTIMIZATION 1: Crop to ROI (reduces processing area)
    if roi:
        image = crop_roi(image, roi)
        if debug:
            print(f"   ⚡ ROI cropped: {roi}")
    
    # OPTIMIZATION 2: Downscale for faster processing
    image = downscale_image(image, target_height)
    if debug:
        print(f"   ⚡ Image downscaled to height: {image.shape[0]}px")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # OPTIMIZATION 3: Conditional enhancement (only if needed)
    if auto_enhance:
        needs_enhancement = assess_image_quality(gray)
        if not needs_enhancement:
            if debug:
                print("   ⚡ Image quality good, skipping enhancement")
            # Just apply basic threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            enhanced_path = 'temp_enhanced_for_ocr.jpg'
            cv2.imwrite(enhanced_path, enhanced_bgr)
            
            elapsed = (time.time() - start_time) * 1000
            if debug:
                print(f"   ⏱ Preprocessing time: {elapsed:.1f}ms (fast path)")
            return enhanced_path
    
    # Apply full enhancement if needed
    # Increase contrast using CLAHE (reduced clipLimit for speed)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # OPTIMIZATION 4: Skip denoising if image is small (already downscaled)
    if image.shape[0] > 500:
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    else:
        denoised = enhanced  # Skip denoising on small images
    
    # Sharpen (only if needed)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Apply threshold
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OPTIMIZATION 5: Only save debug images if debug mode enabled
    if debug:
        cv2.imwrite('debug_enhanced.jpg', enhanced)
        cv2.imwrite('debug_thresh.jpg', thresh)
    
    # Convert back to BGR for the agent
    enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Save the enhanced image
    enhanced_path = 'temp_enhanced_for_ocr.jpg'
    cv2.imwrite(enhanced_path, enhanced_bgr)
    
    elapsed = (time.time() - start_time) * 1000
    if debug:
        print(f"   ⏱ Preprocessing time: {elapsed:.1f}ms (full enhancement)")
    
    return enhanced_path


def extract_text_enhanced(image_path, whitelist=None, psm_modes=None, roi=None, target_height=400):
    """
    OPTIMIZED: Extract text using optimized OCR configuration
    
    Args:
        image_path: Path to image
        whitelist: Character whitelist (e.g., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        psm_modes: List of PSM modes to try (reduced for speed)
        roi: Region of interest to crop
        target_height: Target height for downscaling
    """
    start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        return ""
    
    # OPTIMIZATION 1: Crop ROI
    if roi:
        image = crop_roi(image, roi)
    
    # OPTIMIZATION 2: Downscale
    image = downscale_image(image, target_height)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Quick preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OPTIMIZATION 3: Reduced preprocessing variants (2 instead of 3)
    images_to_try = [enhanced, thresh1]
    
    # OPTIMIZATION 4: Use optimized PSM modes (default to 2 modes instead of 4)
    if psm_modes is None:
        psm_modes = OPTIMIZED_PSM_MODES
    
    # OPTIMIZATION 5: Add character whitelist to Tesseract config
    whitelist_config = f"-c tessedit_char_whitelist={whitelist}" if whitelist else ""
    
    all_texts = []
    for img in images_to_try:
        for psm in psm_modes:
            try:
                config = f"{psm} {whitelist_config}".strip()
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
                    
                    # OPTIMIZATION 6: Early exit if we found good text
                    if EARLY_EXIT_CONFIDENCE and len(text.strip()) > 8:
                        elapsed = (time.time() - start_time) * 1000
                        print(f"   ⚡ OCR early exit (confident match): {elapsed:.1f}ms")
                        return text.strip()
            except:
                pass
    
    elapsed = (time.time() - start_time) * 1000
    print(f"   ⏱ OCR time: {elapsed:.1f}ms")
    
    # Return the longest/best result
    return max(all_texts, key=len) if all_texts else ""

# ============================================================================
# IC VERIFICATION CONFIGURATION
# ============================================================================

# Expected IC markings (CHANGE THESE to match your genuine IC)
IC_PART_NUMBER = 'SN74LS266N'        # Main IC part number
IC_MANUFACTURER_MARK = 'HLF'         # Manufacturer logo/marking (optional)
IC_DATE_CODE = '20A1'                # Batch/Date code (optional, can be None)

# Configure reference data - FOCUS ON TEXT ONLY
reference_data = {
    'logo_path': None,                    # Skip logo matching
    'expected_text': IC_PART_NUMBER,      # Verify IC part number
    'expected_qr_data': None,             # Skip QR code
    'golden_image_path': None,            # Skip pixel-by-pixel comparison
    'color_reference': None               # Skip color verification
}

# Path to the IC image you want to verify
test_image_path = 'test_images/product_to_verify.jpg'  # CHANGE THIS to your image path

def analyze_ic_markings(ocr_text):
    """
    Analyze and extract IC markings from OCR text
    Returns: dict with part_number, manufacturer_mark, date_code
    """
    markings = {
        'part_number': None,
        'manufacturer_mark': None,
        'date_code': None
    }
    
    # Clean OCR text
    text = ocr_text.upper().replace('\n', ' ').replace('\r', ' ')
    
    # Extract part number (SN74LS266N pattern)
    part_pattern = r'SN\d{2}[A-Z]{2,4}\d{2,4}[A-Z]?'
    part_match = re.search(part_pattern, text)
    if part_match:
        markings['part_number'] = part_match.group()
    
    # Extract manufacturer mark (HLF pattern)
    if 'HLF' in text or 'HL' in text:
        markings['manufacturer_mark'] = 'HLF'
    
    # Extract date code (20A1 pattern - 2 digits + alphanumeric)
    date_pattern = r'\d{2}[A-Z]\d'
    date_match = re.search(date_pattern, text)
    if date_match:
        markings['date_code'] = date_match.group()
    
    return markings


def verify_ic_markings(detected_markings, expected_part, expected_mfg, expected_date):
    """
    Verify IC markings against expected values
    """
    results = {
        'part_number_match': False,
        'manufacturer_match': False,
        'date_code_match': False,
        'overall_authentic': False
    }
    
    # Check part number (CRITICAL)
    if detected_markings['part_number'] == expected_part:
        results['part_number_match'] = True
    
    # Check manufacturer mark (OPTIONAL)
    if expected_mfg:
        if detected_markings['manufacturer_mark'] == expected_mfg:
            results['manufacturer_match'] = True
    else:
        results['manufacturer_match'] = True  # Skip if not provided
    
    # Check date code (OPTIONAL - can vary for genuine ICs)
    if expected_date:
        if detected_markings['date_code'] == expected_date:
            results['date_code_match'] = True
    else:
        results['date_code_match'] = True  # Skip if not provided
    
    # Overall authenticity (part number is CRITICAL)
    results['overall_authentic'] = results['part_number_match']
    
    return results


print("="*70)
print("IC CHIP VERIFICATION - OPTIMIZED FOR SPEED")
print("="*70)
print(f"\n⚡ Performance Settings:")
print(f"   Debug Mode:       {'ON' if DEBUG_MODE else 'OFF (production)'}")
print(f"   ROI Cropping:     {'Enabled' if ROI else 'Disabled (full image)'}")
print(f"   Target Height:    {TARGET_OCR_HEIGHT}px")
print(f"   Auto Enhance:     {'ON' if AUTO_ENHANCE else 'OFF'}")
print(f"   Early Exit:       {'ON' if EARLY_EXIT_CONFIDENCE else 'OFF'}")
print(f"   Tesseract Modes:  {len(OPTIMIZED_PSM_MODES)} (optimized)")
print(f"\n📋 Expected IC Markings:")
print(f"   Part Number:      {IC_PART_NUMBER}")
print(f"   Manufacturer:     {IC_MANUFACTURER_MARK if IC_MANUFACTURER_MARK else 'Any'}")
print(f"   Date/Batch Code:  {IC_DATE_CODE if IC_DATE_CODE else 'Any'}")
print(f"\n📷 Testing Image: {test_image_path}")
print("="*70 + "\n")

# Run verification
try:
    overall_start = time.time()
    
    # STEP 1: Preprocess image for better OCR (OPTIMIZED)
    print("🔄 Preprocessing image for OCR (optimized)...")
    enhanced_image_path = preprocess_image_for_ocr(
        test_image_path, 
        debug=DEBUG_MODE,
        auto_enhance=AUTO_ENHANCE,
        roi=ROI,
        target_height=TARGET_OCR_HEIGHT
    )
    
    if enhanced_image_path is None:
        raise ValueError(f"Could not load image: {test_image_path}")
    
    print("✓ Image preprocessed and saved to: temp_enhanced_for_ocr.jpg")
    if DEBUG_MODE:
        print("✓ Debug images saved: debug_enhanced.jpg, debug_thresh.jpg\n")
    else:
        print("⚡ Debug images skipped (production mode)\n")
    
    # STEP 2: Run agent on ENHANCED image (skip if not needed)
    # Skip agent processing if we only need OCR
    # report = agent.process_image(enhanced_image_path, reference_data)
    
    # STEP 3: Run optimized OCR extraction
    print("🔍 Running optimized OCR extraction...")
    enhanced_text = extract_text_enhanced(
        test_image_path,
        whitelist=TESSERACT_WHITELIST,
        psm_modes=OPTIMIZED_PSM_MODES,
        roi=ROI,
        target_height=TARGET_OCR_HEIGHT
    )
    print(f"✓ Enhanced OCR result: '{enhanced_text[:100]}...'\n")
    
    detected_text = enhanced_text
    
    # Analyze IC markings (OPTIMIZED - direct analysis)
    detected_markings = analyze_ic_markings(detected_text)
    verification_results = verify_ic_markings(
        detected_markings, 
        IC_PART_NUMBER, 
        IC_MANUFACTURER_MARK, 
        IC_DATE_CODE
    )
    
    # Display detailed results
    print("\n" + "="*70)
    print("DETECTED IC MARKINGS")
    print("="*70)
    print(f"📝 Raw OCR Text: '{detected_text.strip()}'")
    print(f"\n🔍 Extracted Markings:")
    print(f"   Part Number:      {detected_markings['part_number'] or 'NOT DETECTED'}")
    print(f"   Manufacturer:     {detected_markings['manufacturer_mark'] or 'NOT DETECTED'}")
    print(f"   Date/Batch Code:  {detected_markings['date_code'] or 'NOT DETECTED'}")
    
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    
    # Part Number Check (CRITICAL)
    part_status = "✅ PASS" if verification_results['part_number_match'] else "❌ FAIL"
    print(f"{part_status} Part Number: {IC_PART_NUMBER}")
    if detected_markings['part_number']:
        print(f"     Detected: {detected_markings['part_number']}")
    
    # Manufacturer Check (OPTIONAL)
    if IC_MANUFACTURER_MARK:
        mfg_status = "✅ PASS" if verification_results['manufacturer_match'] else "⚠️  WARN"
        print(f"{mfg_status} Manufacturer: {IC_MANUFACTURER_MARK}")
        if detected_markings['manufacturer_mark']:
            print(f"     Detected: {detected_markings['manufacturer_mark']}")
    
    # Date Code Check (OPTIONAL)
    if IC_DATE_CODE:
        date_status = "✅ PASS" if verification_results['date_code_match'] else "⚠️  INFO"
        print(f"{date_status} Date/Batch Code: {IC_DATE_CODE}")
        if detected_markings['date_code']:
            print(f"     Detected: {detected_markings['date_code']}")
        print(f"     Note: Date codes can vary for genuine ICs from different batches")
    
    # Final Verdict
    print("\n" + "="*70)
    final_verdict = "GENUINE" if verification_results['overall_authentic'] else "COUNTERFEIT/UNVERIFIED"
    print(f"🎯 FINAL VERDICT: {final_verdict}")
    print("="*70)
    
    if verification_results['overall_authentic']:
        print("\n✅ IC part number matches! This appears to be a genuine IC.")
        print("   The IC is: SN74LS266N - Quad 2-input XNOR Gate (74LS series)")
    else:
        print("\n❌ IC part number does NOT match or could not be detected!")
        print("   This may be counterfeit, damaged, or OCR failed to read the text.")
        print("\n💡 Troubleshooting:")
        print("   1. Check debug images: debug_enhanced.jpg, debug_thresh.jpg")
        print("   2. Check enhanced image: temp_enhanced_for_ocr.jpg")
        print("   3. Take a better photo (closer, better lighting, no glare)")
        print("   4. Ensure IC text is clearly visible and in focus")
    
    # Calculate total processing time
    total_time = (time.time() - overall_start) * 1000
    print(f"\n⚡ TOTAL PROCESSING TIME: {total_time:.1f}ms")
    print(f"   Target: <30ms per IC (production)")
    if total_time < 30:
        print(f"   ✅ EXCELLENT - Meeting production target!")
    elif total_time < 50:
        print(f"   ⚠️  GOOD - Consider further optimization")
    else:
        print(f"   ❌ SLOW - Enable production mode (DEBUG_MODE=False)")
    
    # Save detailed report
    import datetime
    detailed_report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'processing_time_ms': total_time,
        'test_image': test_image_path,
        'performance_settings': {
            'debug_mode': DEBUG_MODE,
            'roi': ROI,
            'target_height': TARGET_OCR_HEIGHT,
            'auto_enhance': AUTO_ENHANCE,
            'early_exit': EARLY_EXIT_CONFIDENCE
        },
        'expected_markings': {
            'part_number': IC_PART_NUMBER,
            'manufacturer': IC_MANUFACTURER_MARK,
            'date_code': IC_DATE_CODE
        },
        'detected_markings': detected_markings,
        'verification_results': verification_results,
        'final_verdict': final_verdict,
        'ocr_raw_text': detected_text
    }
    
    agent.save_report(detailed_report, 'ic_verification_report.json')
    print("\n📄 Detailed report saved to: ic_verification_report.json")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
