"""
IC Verification System - Fixed Version
Comprehensive analysis of IC authenticity based on multiple parameters
"""

import cv2
import numpy as np
import pytesseract
import re
import os
import time
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE_IC_PATH = 'reference/golden_product.jpg'
TEST_IC_PATH = 'test_images/product_to_verify.jpg'
DEBUG_MODE = True

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def preprocess_for_ocr(image, prefix='debug', debug=True):
    """Enhanced preprocessing for IC text extraction"""
    start_time = time.time()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(bilateral, None, 10, 7, 21)
    
    # Multiple threshold methods
    _, thresh1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh3 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, thresh5 = cv2.threshold(denoised, 120, 255, cv2.THRESH_BINARY)
    _, thresh6 = cv2.threshold(denoised, 120, 255, cv2.THRESH_BINARY_INV)
    
    if debug:
        cv2.imwrite(f'{prefix}_enhanced.jpg', enhanced)
        cv2.imwrite(f'{prefix}_bilateral.jpg', bilateral)
        cv2.imwrite(f'{prefix}_thresh1_otsu.jpg', thresh1)
        cv2.imwrite(f'{prefix}_thresh2_otsu_inv.jpg', thresh2)
        cv2.imwrite(f'{prefix}_thresh3_adaptive.jpg', thresh3)
        cv2.imwrite(f'{prefix}_thresh4_adaptive_inv.jpg', thresh4)
        cv2.imwrite(f'{prefix}_thresh5_manual.jpg', thresh5)
        cv2.imwrite(f'{prefix}_thresh6_manual_inv.jpg', thresh6)
    
    return [enhanced, bilateral, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6]

def extract_text_multiple_methods(image_path, prefix='debug', debug=True):
    """Extract text using multiple OCR methods"""
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # Preprocess
    preprocessed_images = preprocess_for_ocr(image, prefix, debug)
    
    # OCR configurations
    psm_modes = ['--psm 11', '--psm 6']
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    all_texts = []
    
    for img in preprocessed_images:
        for psm in psm_modes:
            try:
                config = f"{psm} -c tessedit_char_whitelist={whitelist}"
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
            except:
                pass
    
    return all_texts

def analyze_ic_markings(ocr_texts):
    """Analyze OCR results to extract IC markings"""
    markings = {
        'part_number': None,
        'manufacturer_mark': None,
        'date_code': None,
        'additional_codes': [],
        'all_text': []
    }
    
    for text in ocr_texts:
        if not text:
            continue
        
        markings['all_text'].append(text)
        text_upper = text.upper().replace('\n', ' ').replace('\r', ' ')
        text_upper = ' '.join(text_upper.split())
        
        # OCR corrections
        corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'Z': '2', 'z': '2',
        }
        
        text_corrected = text_upper
        for old, new in corrections.items():
            text_corrected = text_corrected.replace(old, new)
        
        # Extract part number
        if not markings['part_number']:
            part_patterns = [
                r'MC74[A-Z]{0,4}\d{2,4}[A-Z]{0,2}',
                r'MCT[A-Z0-9]{2,8}[A-Z]?',
                r'SN74[A-Z]{0,4}\d{2,3}[A-Z]{0,2}',
                r'74[A-Z]{1,4}\d{2,4}[A-Z]{0,2}',
            ]
            
            for pattern in part_patterns:
                part_match = re.search(pattern, text_corrected)
                if part_match:
                    markings['part_number'] = part_match.group()
                    break
        
        # Extract manufacturer
        if not markings['manufacturer_mark']:
            mfg_patterns = [
                r'XXAC\d{4}',
                r'HLF',
                r'HSS\d{4}',
                r'TI',
                r'NXP',
            ]
            
            for pattern in mfg_patterns:
                mfg_match = re.search(pattern, text_upper)
                if mfg_match:
                    markings['manufacturer_mark'] = mfg_match.group()
                    break
        
        # Extract date code
        if not markings['date_code']:
            date_patterns = [
                r'\d{4}',
                r'\d{6}[A-Z]',
                r'\d{2}[A-Z]\d',
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, text_corrected)
                if date_match:
                    match_text = date_match.group()
                    if markings['part_number'] and match_text not in markings['part_number']:
                        markings['date_code'] = match_text
                        break
    
    return markings

def compare_visual_features(ref_img_path, test_img_path):
    """Compare visual characteristics between reference and test IC"""
    ref_img = cv2.imread(ref_img_path)
    test_img = cv2.imread(test_img_path)
    
    if ref_img is None or test_img is None:
        return {'similarity': 0.0, 'details': 'Failed to load images'}
    
    # Resize both images to the same dimensions
    target_size = (500, 500)  # Fixed size for comparison
    
    ref_resized = cv2.resize(ref_img, target_size)
    test_resized = cv2.resize(test_img, target_size)
    
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)
    
    try:
        # Calculate SSIM
        from skimage.metrics import structural_similarity as ssim
        ssim_score, _ = ssim(ref_gray, test_gray, full=True)
    except:
        ssim_score = 0.0
    
    # Calculate histogram correlation
    ref_hist = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
    test_hist = cv2.calcHist([test_gray], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(ref_hist, test_hist, cv2.HISTCMP_CORREL)
    
    # Combined similarity
    combined_similarity = (ssim_score + hist_corr) / 2
    
    return {
        'similarity': combined_similarity,
        'ssim_score': ssim_score,
        'histogram_correlation': hist_corr,
        'details': f'SSIM: {ssim_score:.3f}, Histogram: {hist_corr:.3f}'
    }

def comprehensive_ic_analysis(ref_path, test_path):
    """Comprehensive IC analysis based on multiple parameters"""
    print("="*80)
    print("COMPREHENSIVE IC VERIFICATION ANALYSIS")
    print("="*80)
    
    # Load and process reference IC
    print("\n[1] REFERENCE IC ANALYSIS")
    print("-"*50)
    
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference IC not found: {ref_path}")
        return False
    
    ref_start = time.time()
    ref_texts = extract_text_multiple_methods(ref_path, 'ref_debug', DEBUG_MODE)
    ref_time = (time.time() - ref_start) * 1000
    
    print(f"Reference IC processed in {ref_time:.1f}ms")
    print(f"OCR results: {len(ref_texts)} variants")
    
    ref_markings = analyze_ic_markings(ref_texts)
    
    print(f"Part Number: {ref_markings['part_number'] or 'NOT DETECTED'}")
    print(f"Manufacturer: {ref_markings['manufacturer_mark'] or 'NOT DETECTED'}")
    print(f"Date Code: {ref_markings['date_code'] or 'NOT DETECTED'}")
    
    # Load and process test IC
    print("\n[2] TEST IC ANALYSIS")
    print("-"*50)
    
    if not os.path.exists(test_path):
        print(f"ERROR: Test IC not found: {test_path}")
        return False
    
    test_start = time.time()
    test_texts = extract_text_multiple_methods(test_path, 'debug', DEBUG_MODE)
    test_time = (time.time() - test_start) * 1000
    
    print(f"Test IC processed in {test_time:.1f}ms")
    print(f"OCR results: {len(test_texts)} variants")
    
    test_markings = analyze_ic_markings(test_texts)
    
    print(f"Part Number: {test_markings['part_number'] or 'NOT DETECTED'}")
    print(f"Manufacturer: {test_markings['manufacturer_mark'] or 'NOT DETECTED'}")
    print(f"Date Code: {test_markings['date_code'] or 'NOT DETECTED'}")
    
    # Visual comparison
    print("\n[3] VISUAL CHARACTERISTICS COMPARISON")
    print("-"*50)
    
    visual_comparison = compare_visual_features(ref_path, test_path)
    print(f"Visual Similarity: {visual_comparison['similarity']:.3f}")
    print(f"Details: {visual_comparison['details']}")
    
    # Comprehensive verification
    print("\n[4] COMPREHENSIVE VERIFICATION")
    print("-"*50)
    
    # Part number comparison
    part_match = (ref_markings['part_number'] == test_markings['part_number'])
    print(f"Part Number Match: {'PASS' if part_match else 'FAIL'}")
    print(f"  Reference: {ref_markings['part_number']}")
    print(f"  Test:      {test_markings['part_number']}")
    
    # Manufacturer comparison
    mfg_match = (ref_markings['manufacturer_mark'] == test_markings['manufacturer_mark'])
    print(f"Manufacturer Match: {'PASS' if mfg_match else 'FAIL'}")
    print(f"  Reference: {ref_markings['manufacturer_mark']}")
    print(f"  Test:      {test_markings['manufacturer_mark']}")
    
    # Date code comparison (optional)
    date_match = (ref_markings['date_code'] == test_markings['date_code'])
    print(f"Date Code Match: {'PASS' if date_match else 'INFO (optional)'}")
    print(f"  Reference: {ref_markings['date_code']}")
    print(f"  Test:      {test_markings['date_code']}")
    
    # Visual similarity check
    visual_match = visual_comparison['similarity'] >= 0.7
    print(f"Visual Similarity: {'PASS' if visual_match else 'FAIL'}")
    print(f"  Score: {visual_comparison['similarity']:.3f}")
    
    # Final verdict
    print("\n[5] FINAL VERDICT")
    print("="*50)
    
    # Calculate overall score
    checks = [part_match, mfg_match, visual_match]
    passed_checks = sum(checks)
    total_checks = len(checks)
    
    overall_score = passed_checks / total_checks
    
    print(f"Overall Score: {overall_score:.2f} ({passed_checks}/{total_checks} checks passed)")
    
    if overall_score >= 0.67:  # At least 2 out of 3 checks pass
        verdict = "IDENTICAL"
        status = "PASS"
    else:
        verdict = "DIFFERENT"
        status = "FAIL"
    
    print(f"VERDICT: {verdict}")
    print(f"STATUS: {status}")
    
    # Detailed analysis
    print("\n[6] DETAILED ANALYSIS")
    print("-"*50)
    
    if verdict == "IDENTICAL":
        print("ANALYSIS: The two ICs appear to be identical based on:")
        if part_match:
            print("  - Part numbers match exactly")
        if mfg_match:
            print("  - Manufacturer codes match exactly")
        if visual_match:
            print("  - Visual characteristics are highly similar")
        print("This suggests they are the same type of IC from the same manufacturer.")
    else:
        print("ANALYSIS: The two ICs appear to be different based on:")
        if not part_match:
            print("  - Part numbers do not match")
        if not mfg_match:
            print("  - Manufacturer codes do not match")
        if not visual_match:
            print("  - Visual characteristics are dissimilar")
        print("This suggests they are different ICs or potentially counterfeit.")
    
    # Performance summary
    total_time = (time.time() - ref_start) * 1000
    print(f"\n[7] PERFORMANCE SUMMARY")
    print("-"*50)
    print(f"Reference IC processing: {ref_time:.1f}ms")
    print(f"Test IC processing: {test_time:.1f}ms")
    print(f"Total analysis time: {total_time:.1f}ms")
    
    if DEBUG_MODE:
        print(f"\nDebug images saved:")
        print(f"  Reference: ref_debug_*.jpg")
        print(f"  Test: debug_*.jpg")
    
    return {
        'verdict': verdict,
        'status': status,
        'overall_score': overall_score,
        'part_match': part_match,
        'mfg_match': mfg_match,
        'visual_match': visual_match,
        'visual_similarity': visual_comparison['similarity'],
        'processing_time': total_time,
        'reference_markings': ref_markings,
        'test_markings': test_markings
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        result = comprehensive_ic_analysis(REFERENCE_IC_PATH, TEST_IC_PATH)
        
        # Save results to JSON file
        timestamp = datetime.now().isoformat()
        result['timestamp'] = timestamp
        result['reference_path'] = REFERENCE_IC_PATH
        result['test_path'] = TEST_IC_PATH
        
        with open('ic_verification_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\nResults saved to: ic_verification_results.json")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
