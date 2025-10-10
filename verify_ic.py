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

from counterfeit_detection_agent import CounterfeitDetectionAgent
import cv2
import numpy as np
import pytesseract
import re

# Initialize the agent
agent = CounterfeitDetectionAgent()


def preprocess_image_for_ocr(image_path):
    """
    Preprocess image for better OCR using enhanced techniques
    Returns the best preprocessed image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
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
    
    # Apply threshold
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save debug images
    cv2.imwrite('debug_enhanced.jpg', enhanced)
    cv2.imwrite('debug_thresh.jpg', thresh)
    
    # Convert back to BGR for the agent
    enhanced_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Save the enhanced image
    enhanced_path = 'temp_enhanced_for_ocr.jpg'
    cv2.imwrite(enhanced_path, enhanced_bgr)
    
    return enhanced_path


def extract_text_enhanced(image_path):
    """
    Extract text using multiple OCR methods on preprocessed images
    """
    image = cv2.imread(image_path)
    if image is None:
        return ""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple preprocessing methods
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Try multiple PSM modes
    psm_modes = ['--psm 6', '--psm 7', '--psm 11', '--psm 13']
    
    all_texts = []
    for img in [enhanced, thresh1, thresh2]:
        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(img, config=psm)
                if text.strip():
                    all_texts.append(text.strip())
            except:
                pass
    
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
print("IC CHIP VERIFICATION - DETAILED MARKING ANALYSIS")
print("="*70)
print(f"\n📋 Expected IC Markings:")
print(f"   Part Number:      {IC_PART_NUMBER}")
print(f"   Manufacturer:     {IC_MANUFACTURER_MARK if IC_MANUFACTURER_MARK else 'Any'}")
print(f"   Date/Batch Code:  {IC_DATE_CODE if IC_DATE_CODE else 'Any'}")
print(f"\n📷 Testing Image: {test_image_path}")
print("="*70 + "\n")

# Run verification
try:
    # STEP 1: Preprocess image for better OCR
    print("🔄 Preprocessing image for better OCR...")
    enhanced_image_path = preprocess_image_for_ocr(test_image_path)
    
    if enhanced_image_path is None:
        raise ValueError(f"Could not load image: {test_image_path}")
    
    print("✓ Image enhanced and saved to: temp_enhanced_for_ocr.jpg")
    print("✓ Debug images saved: debug_enhanced.jpg, debug_thresh.jpg\n")
    
    # STEP 2: Run agent on ENHANCED image
    report = agent.process_image(enhanced_image_path, reference_data)
    
    # STEP 3: Also try direct enhanced OCR
    print("🔍 Running enhanced OCR extraction...")
    enhanced_text = extract_text_enhanced(test_image_path)
    print(f"✓ Enhanced OCR result: '{enhanced_text[:100]}...'\n")
    
    # Extract OCR text from agent
    detected_text = ""
    for result in report['pipeline_results']:
        if result['step'] == '2. Text Area + OCR':
            detected_text = result['details'].get('ocr_text', '')
            break
    
    # Use enhanced text if agent's OCR failed
    if not detected_text or len(detected_text) < 10:
        print("⚠️  Agent OCR failed, using enhanced OCR result instead\n")
        detected_text = enhanced_text
    
    # Analyze IC markings
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
    
    # Save detailed report
    detailed_report = {
        'timestamp': report['timestamp'],
        'test_image': test_image_path,
        'expected_markings': {
            'part_number': IC_PART_NUMBER,
            'manufacturer': IC_MANUFACTURER_MARK,
            'date_code': IC_DATE_CODE
        },
        'detected_markings': detected_markings,
        'verification_results': verification_results,
        'final_verdict': final_verdict,
        'ocr_raw_text': detected_text,
        'full_pipeline_report': report
    }
    
    agent.save_report(detailed_report, 'ic_verification_report.json')
    print("\n📄 Detailed report saved to: ic_verification_report.json")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
