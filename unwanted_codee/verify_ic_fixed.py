"""
FIXED IC Verification with Advanced OCR for Embossed Text
Specialized preprocessing for reading embossed/engraved IC markings on dark surfaces
"""

import cv2
import numpy as np
import pytesseract
import re
import os
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE_IC_PATH = 'reference/golden_product.jpg'
TEST_IC_PATH = 'test_images/product_to_verify.jpg'
DEBUG_MODE = True

# ============================================================================
# ADVANCED PREPROCESSING FOR EMBOSSED TEXT
# ============================================================================

def enhance_embossed_text(image):
    """
    Advanced preprocessing specifically for embossed/engraved text on dark IC surfaces
    Uses multiple techniques to make the text visible
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Upscale for better OCR
    scale = 3
    h, w = gray.shape
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    
    processed_images = []
    
    # ========================================================================
    # METHOD 1: CLAHE + Bilateral Filter + Adaptive Threshold
    # ========================================================================
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced1 = clahe.apply(gray)
    bilateral1 = cv2.bilateralFilter(enhanced1, 9, 75, 75)
    
    # Multiple adaptive thresholds
    adaptive1 = cv2.adaptiveThreshold(bilateral1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    adaptive2 = cv2.adaptiveThreshold(bilateral1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    adaptive3 = cv2.adaptiveThreshold(bilateral1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 3)
    adaptive4 = cv2.adaptiveThreshold(bilateral1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 3)
    
    processed_images.extend([adaptive1, adaptive2, adaptive3, adaptive4])
    
    # ========================================================================
    # METHOD 2: Morphological Gradient (Edge Detection)
    # ========================================================================
    # This is excellent for embossed text - detects edges/boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(enhanced1, cv2.MORPH_GRADIENT, kernel)
    _, gradient_thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(gradient_thresh)
    
    # ========================================================================
    # METHOD 3: Top-Hat and Black-Hat Transforms
    # ========================================================================
    # Top-hat: reveals bright text on dark background
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(enhanced1, cv2.MORPH_TOPHAT, kernel_large)
    _, tophat_thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(tophat_thresh)
    
    # Black-hat: reveals dark text on bright background
    blackhat = cv2.morphologyEx(enhanced1, cv2.MORPH_BLACKHAT, kernel_large)
    _, blackhat_thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(blackhat_thresh)
    
    # ========================================================================
    # METHOD 4: Unsharp Masking (Sharpening)
    # ========================================================================
    gaussian = cv2.GaussianBlur(enhanced1, (9, 9), 10.0)
    unsharp = cv2.addWeighted(enhanced1, 1.5, gaussian, -0.5, 0)
    _, unsharp_thresh1 = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, unsharp_thresh2 = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.extend([unsharp_thresh1, unsharp_thresh2])
    
    # ========================================================================
    # METHOD 5: Canny Edge Detection + Dilation
    # ========================================================================
    edges = cv2.Canny(enhanced1, 50, 150)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
    processed_images.append(edges_dilated)
    
    # ========================================================================
    # METHOD 6: Contrast Stretching + Multiple Thresholds
    # ========================================================================
    # Normalize to full range
    normalized = cv2.normalize(enhanced1, None, 0, 255, cv2.NORM_MINMAX)
    
    # Try multiple fixed thresholds
    for thresh_val in [80, 100, 120, 140, 160]:
        _, fixed_thresh = cv2.threshold(normalized, thresh_val, 255, cv2.THRESH_BINARY)
        processed_images.append(fixed_thresh)
        _, fixed_thresh_inv = cv2.threshold(normalized, thresh_val, 255, cv2.THRESH_BINARY_INV)
        processed_images.append(fixed_thresh_inv)
    
    # ========================================================================
    # METHOD 7: Sobel Edge Detection
    # ========================================================================
    sobelx = cv2.Sobel(enhanced1, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced1, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    _, sobel_thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(sobel_thresh)
    
    # ========================================================================
    # METHOD 8: Laplacian Edge Detection
    # ========================================================================
    laplacian = cv2.Laplacian(enhanced1, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, laplacian_thresh = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(laplacian_thresh)
    
    # Add the enhanced original for comparison
    processed_images.insert(0, enhanced1)
    
    return processed_images, enhanced1


def extract_text_advanced(image_path, prefix='debug'):
    """
    Extract text using advanced preprocessing and multiple OCR configurations
    """
    print(f"🔄 Loading and preprocessing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return []
    
    # Apply advanced preprocessing
    start_time = time.time()
    processed_images, enhanced = enhance_embossed_text(image)
    preprocess_time = (time.time() - start_time) * 1000
    print(f"✓ Preprocessing complete: {len(processed_images)} variants in {preprocess_time:.1f}ms")
    
    # Save debug images
    if DEBUG_MODE:
        cv2.imwrite(f'{prefix}_0_enhanced.jpg', enhanced)
        for i, img in enumerate(processed_images[1:11], 1):  # Save first 10
            cv2.imwrite(f'{prefix}_{i}_variant.jpg', img)
        print(f"📁 Saved {min(11, len(processed_images))} debug images with prefix '{prefix}_'")
    
    # OCR Configuration
    # PSM modes optimized for IC text
    psm_modes = [
        '--psm 6',   # Uniform block of text
        '--psm 7',   # Single line of text
        '--psm 11',  # Sparse text
        '--psm 12',  # Sparse text with OSD
        '--psm 13',  # Raw line
        '--psm 8',   # Single word
    ]
    
    # Character whitelist for IC markings
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    all_texts = []
    ocr_start = time.time()
    
    print(f"🔍 Running OCR on {len(processed_images)} preprocessed images...")
    
    for img_idx, img in enumerate(processed_images):
        for psm in psm_modes:
            try:
                # Try with whitelist
                config = f"{psm} -c tessedit_char_whitelist={whitelist}"
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
                
                # Try without whitelist (sometimes works better)
                text2 = pytesseract.image_to_string(img, config=psm)
                if text2.strip() and text2.strip() not in all_texts:
                    all_texts.append(text2.strip())
                
                # Try with different OEM modes
                config_oem = f"{psm} --oem 1 -c tessedit_char_whitelist={whitelist}"
                text3 = pytesseract.image_to_string(img, config=config_oem)
                if text3.strip() and text3.strip() not in all_texts:
                    all_texts.append(text3.strip())
                    
            except Exception as e:
                pass
    
    ocr_time = (time.time() - ocr_start) * 1000
    print(f"✓ OCR complete: {len(all_texts)} text variants extracted in {ocr_time:.1f}ms")
    
    return all_texts


def clean_ocr_text(text):
    """
    Clean and normalize OCR text
    """
    # Convert to uppercase
    text = text.upper()
    
    # Remove newlines and extra spaces
    text = ' '.join(text.split())
    
    # Common OCR corrections for IC markings
    corrections = {
        'O': '0', 'o': '0',           # O to 0
        'I': '1', 'l': '1', '|': '1',  # I/l to 1
        'Z': '2', 'z': '2',            # Z to 2
        'S': '5',                      # Sometimes S is 5
        'B': '8',                      # Sometimes B is 8
    }
    
    # Apply corrections carefully (only in numeric contexts)
    return text


def extract_part_number(texts):
    """
    Extract IC part number from OCR results with advanced pattern matching
    """
    all_part_numbers = []
    
    for text in texts:
        text_clean = clean_ocr_text(text)
        
        # Pattern 1: Direct match - SN74XXXXX format
        patterns = [
            r'SN74[A-Z]{0,4}\d{2,3}[A-Z]{0,2}',  # SN74LS90N, SN74HC00N
            r'SN\s*74\s*[A-Z]{0,4}\s*\d{2,3}\s*[A-Z]{0,2}',  # With spaces
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_clean)
            for match in matches:
                # Remove spaces
                clean_match = match.replace(' ', '')
                if len(clean_match) >= 7:  # Minimum valid length
                    all_part_numbers.append(clean_match)
        
        # Pattern 2: Fragmented text assembly
        # Look for "SN" "74" "LS" "90" "N" as separate fragments
        words = text_clean.split()
        
        # Try to find SN74 pattern in fragments
        for i, word in enumerate(words):
            if 'SN' in word or word == 'SN':
                # Look ahead for 74, LS/HC, numbers, N
                assembled = 'SN'
                
                # Look for 74
                for j in range(i, min(i+5, len(words))):
                    if '74' in words[j]:
                        assembled += '74'
                        break
                else:
                    # Assume 74 if we have LS or HC nearby
                    if any('LS' in words[k] or 'HC' in words[k] for k in range(i, min(i+5, len(words)))):
                        assembled += '74'
                
                # Look for LS/HC/HCT
                for j in range(i, min(i+8, len(words))):
                    if 'LS' in words[j]:
                        assembled += 'LS'
                        break
                    elif 'HC' in words[j]:
                        assembled += 'HC'
                        break
                    elif 'HCT' in words[j]:
                        assembled += 'HCT'
                        break
                
                # Look for 2-3 digit number
                for j in range(i, min(i+10, len(words))):
                    if re.match(r'^\d{2,3}$', words[j]):
                        assembled += words[j]
                        break
                
                # Look for ending letter (usually N)
                for j in range(i, min(i+12, len(words))):
                    if words[j] in ['N', 'AN', 'BN', 'DN'] or (len(words[j]) <= 2 and words[j].endswith('N')):
                        if not assembled.endswith('N'):
                            assembled += 'N'
                        break
                
                # Validate assembled part number
                if len(assembled) >= 8 and assembled.startswith('SN74'):
                    all_part_numbers.append(assembled)
    
    # Return most common part number (voting)
    if all_part_numbers:
        from collections import Counter
        counter = Counter(all_part_numbers)
        most_common = counter.most_common(1)[0][0]
        print(f"   Found part numbers: {list(set(all_part_numbers))}")
        print(f"   Selected (most common): {most_common}")
        return most_common
    
    return None


def extract_manufacturer(texts):
    """
    Extract manufacturer marking
    """
    for text in texts:
        text_upper = text.upper()
        if 'HLF' in text_upper:
            return 'HLF'
        if 'HL' in text_upper and 'F' in text_upper:
            return 'HLF'
    return None


def extract_date_code(texts):
    """
    Extract date/batch code
    """
    for text in texts:
        text_clean = clean_ocr_text(text)
        
        # Common date code patterns
        patterns = [
            r'\d{4}',  # 2081, 2024
            r'\d{2}[A-Z]\d{1,2}',  # 20A1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_clean)
            if match:
                return match.group()
    
    return None


def verify_ic(reference_path, test_path):
    """
    Main verification function
    """
    print("="*70)
    print("FIXED IC VERIFICATION - ADVANCED OCR FOR EMBOSSED TEXT")
    print("="*70)
    
    # Step 1: Extract reference IC markings
    print(f"\n📖 STEP 1: Extracting markings from REFERENCE IC")
    print(f"   Path: {reference_path}")
    print("-"*70)
    
    if not os.path.exists(reference_path):
        print(f"❌ Reference image not found: {reference_path}")
        return False
    
    ref_texts = extract_text_advanced(reference_path, prefix='ref_debug')
    
    print(f"\n📝 Reference IC - OCR Results (showing first 15):")
    for i, text in enumerate(ref_texts[:15], 1):
        display = text.replace('\n', ' ')[:100]
        print(f"   {i}. {display}")
    
    print(f"\n🔍 Analyzing reference IC markings...")
    ref_part_number = extract_part_number(ref_texts)
    ref_manufacturer = extract_manufacturer(ref_texts)
    ref_date_code = extract_date_code(ref_texts)
    
    print(f"\n✓ Reference IC Markings:")
    print(f"   Part Number:  {ref_part_number or '❌ NOT DETECTED'}")
    print(f"   Manufacturer: {ref_manufacturer or '❌ NOT DETECTED'}")
    print(f"   Date Code:    {ref_date_code or '(optional)'}")
    
    # CRITICAL: Fail if reference part number not detected
    if not ref_part_number:
        print(f"\n❌ CRITICAL ERROR: Could not detect part number from reference IC!")
        print(f"   Cannot proceed with verification.")
        print(f"\n💡 Suggestions:")
        print(f"   1. Check debug images: ref_debug_*.jpg")
        print(f"   2. Ensure reference IC has clear, visible markings")
        print(f"   3. Try better lighting or different angle")
        return False
    
    # Step 2: Extract test IC markings
    print(f"\n" + "="*70)
    print(f"📷 STEP 2: Extracting markings from TEST IC")
    print(f"   Path: {test_path}")
    print("-"*70)
    
    if not os.path.exists(test_path):
        print(f"❌ Test image not found: {test_path}")
        return False
    
    test_texts = extract_text_advanced(test_path, prefix='test_debug')
    
    print(f"\n📝 Test IC - OCR Results (showing first 15):")
    for i, text in enumerate(test_texts[:15], 1):
        display = text.replace('\n', ' ')[:100]
        print(f"   {i}. {display}")
    
    print(f"\n🔍 Analyzing test IC markings...")
    test_part_number = extract_part_number(test_texts)
    test_manufacturer = extract_manufacturer(test_texts)
    test_date_code = extract_date_code(test_texts)
    
    print(f"\n✓ Test IC Markings:")
    print(f"   Part Number:  {test_part_number or '❌ NOT DETECTED'}")
    print(f"   Manufacturer: {test_manufacturer or '❌ NOT DETECTED'}")
    print(f"   Date Code:    {test_date_code or '(optional)'}")
    
    # Step 3: Verification
    print(f"\n" + "="*70)
    print(f"🔍 STEP 3: VERIFICATION")
    print("="*70)
    
    # CRITICAL: Fail if test part number not detected
    if not test_part_number:
        print(f"\n❌ VERIFICATION FAILED: Could not detect part number from test IC!")
        print(f"\n   Expected: {ref_part_number}")
        print(f"   Detected: NOT DETECTED")
        print(f"\n💡 Suggestions:")
        print(f"   1. Check debug images: test_debug_*.jpg")
        print(f"   2. Ensure test IC has clear, visible markings")
        print(f"   3. Try better lighting or different angle")
        print(f"\n🎯 FINAL VERDICT: ❌ UNVERIFIED (Cannot read markings)")
        return False
    
    # Compare part numbers
    part_match = (test_part_number == ref_part_number)
    
    print(f"\n📊 Comparison Results:")
    print(f"   Part Number:")
    print(f"      Reference: {ref_part_number}")
    print(f"      Test:      {test_part_number}")
    print(f"      Match:     {'✅ YES' if part_match else '❌ NO'}")
    
    if ref_manufacturer and test_manufacturer:
        mfg_match = (test_manufacturer == ref_manufacturer)
        print(f"\n   Manufacturer:")
        print(f"      Reference: {ref_manufacturer}")
        print(f"      Test:      {test_manufacturer}")
        print(f"      Match:     {'✅ YES' if mfg_match else '⚠️  NO'}")
    
    # Final verdict
    print(f"\n" + "="*70)
    if part_match:
        print(f"🎯 FINAL VERDICT: ✅ GENUINE")
        print(f"   The IC part number matches the reference.")
        print("="*70)
        return True
    else:
        print(f"🎯 FINAL VERDICT: ❌ COUNTERFEIT or MISMATCH")
        print(f"   The IC part number does NOT match the reference!")
        print("="*70)
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    result = verify_ic(REFERENCE_IC_PATH, TEST_IC_PATH)
    
    total_time = (time.time() - start_time) * 1000
    print(f"\n⚡ Total execution time: {total_time:.1f}ms")
    
    if DEBUG_MODE:
        print(f"\n📁 Debug images saved:")
        print(f"   Reference: ref_debug_*.jpg")
        print(f"   Test:      test_debug_*.jpg")
