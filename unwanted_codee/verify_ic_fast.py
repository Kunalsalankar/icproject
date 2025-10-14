"""
FAST & ACCURATE IC Verification
Optimized for speed while maintaining high accuracy for embossed text
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
# OPTIMIZED PREPROCESSING (Only the best methods)
# ============================================================================

def enhance_embossed_text_fast(image):
    """
    Fast preprocessing - only the most effective methods for embossed text
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Upscale 2x (not 3x for speed)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    processed_images = []
    
    # Method 1: CLAHE + Adaptive Threshold (BEST for embossed text)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Multiple adaptive thresholds (most reliable)
    adaptive1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    adaptive2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    processed_images.extend([adaptive1, adaptive2])
    
    # Method 2: Morphological Gradient (EXCELLENT for edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
    _, gradient_thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(gradient_thresh)
    
    # Method 3: Unsharp Masking (Good for sharpening)
    gaussian = cv2.GaussianBlur(enhanced, (5, 5), 2.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    _, unsharp_thresh = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, unsharp_thresh_inv = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.extend([unsharp_thresh, unsharp_thresh_inv])
    
    # Method 4: Simple contrast normalization with fixed thresholds
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    for thresh_val in [100, 130]:
        _, fixed_thresh = cv2.threshold(normalized, thresh_val, 255, cv2.THRESH_BINARY_INV)
        processed_images.append(fixed_thresh)
    
    return processed_images, enhanced


def extract_text_fast(image_path, prefix='debug'):
    """
    Fast text extraction - optimized OCR configurations
    """
    print(f"🔄 Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image")
        return []
    
    # Preprocess
    start_time = time.time()
    processed_images, enhanced = enhance_embossed_text_fast(image)
    preprocess_time = (time.time() - start_time) * 1000
    print(f"✓ Preprocessed {len(processed_images)} variants in {preprocess_time:.0f}ms")
    
    # Save debug images
    if DEBUG_MODE:
        cv2.imwrite(f'{prefix}_enhanced.jpg', enhanced)
        for i, img in enumerate(processed_images, 1):
            cv2.imwrite(f'{prefix}_variant{i}.jpg', img)
    
    # OCR - only the best PSM modes
    psm_modes = [
        '--psm 11',  # Sparse text (BEST for ICs)
        '--psm 6',   # Uniform block
        '--psm 7',   # Single line
    ]
    
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    all_texts = []
    ocr_start = time.time()
    
    for img in processed_images:
        for psm in psm_modes:
            try:
                # With whitelist
                config = f"{psm} -c tessedit_char_whitelist={whitelist}"
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
                
                # Without whitelist (backup)
                text2 = pytesseract.image_to_string(img, config=psm)
                if text2.strip() and text2.strip() not in all_texts:
                    all_texts.append(text2.strip())
                    
            except:
                pass
    
    ocr_time = (time.time() - ocr_start) * 1000
    print(f"✓ OCR extracted {len(all_texts)} variants in {ocr_time:.0f}ms")
    
    return all_texts


def extract_part_number(texts):
    """
    Extract IC part number with smart pattern matching
    """
    all_candidates = []
    
    for text in texts:
        text_upper = text.upper().replace('\n', ' ')
        text_upper = ' '.join(text_upper.split())
        
        # Apply OCR corrections
        text_corrected = text_upper
        for old, new in [('O', '0'), ('I', '1'), ('l', '1'), ('|', '1')]:
            text_corrected = text_corrected.replace(old, new)
        
        # Pattern 1: Direct match SN74XXXXX
        patterns = [
            r'SN74[A-Z]{1,4}\d{2,3}[A-Z]?',  # SN74LS90N
            r'SN\s*74\s*[A-Z]{1,4}\s*\d{2,3}\s*[A-Z]?',  # With spaces
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_corrected)
            for match in matches:
                clean = match.replace(' ', '')
                if 7 <= len(clean) <= 12:
                    all_candidates.append(clean)
        
        # Pattern 2: Assemble from fragments
        words = text_corrected.split()
        
        for i, word in enumerate(words):
            if 'SN' in word:
                assembled = 'SN'
                
                # Find 74
                if '74' in ' '.join(words[i:i+3]):
                    assembled += '74'
                
                # Find LS/HC/HCT
                for j in range(i, min(i+5, len(words))):
                    if 'LS' in words[j]:
                        assembled += 'LS'
                        break
                    elif 'HC' in words[j] and 'HCT' not in words[j]:
                        assembled += 'HC'
                        break
                    elif 'HCT' in words[j]:
                        assembled += 'HCT'
                        break
                
                # Find 2-3 digit number
                for j in range(i, min(i+8, len(words))):
                    if re.match(r'^\d{2,3}$', words[j]) and words[j] != '74':
                        assembled += words[j]
                        break
                
                # Find N suffix
                for j in range(i, min(i+10, len(words))):
                    if words[j] in ['N', 'AN', 'BN', 'DN']:
                        assembled += 'N'
                        break
                
                if len(assembled) >= 8 and assembled.startswith('SN74'):
                    all_candidates.append(assembled)
    
    # Vote for most common
    if all_candidates:
        from collections import Counter
        counter = Counter(all_candidates)
        winner = counter.most_common(1)[0][0]
        print(f"   Candidates: {list(set(all_candidates))}")
        print(f"   Selected: {winner}")
        return winner
    
    return None


def extract_manufacturer(texts):
    """Extract manufacturer"""
    for text in texts:
        if 'HLF' in text.upper():
            return 'HLF'
    return None


def extract_date_code(texts):
    """Extract date code"""
    for text in texts:
        text_clean = text.upper().replace('\n', ' ')
        match = re.search(r'\d{4}', text_clean)
        if match:
            return match.group()
    return None


def verify_ic(reference_path, test_path):
    """
    Main verification
    """
    print("="*70)
    print("FAST IC VERIFICATION - OPTIMIZED FOR EMBOSSED TEXT")
    print("="*70)
    
    # Extract reference
    print(f"\n📖 REFERENCE IC: {reference_path}")
    print("-"*70)
    
    if not os.path.exists(reference_path):
        print(f"❌ File not found")
        return False
    
    ref_texts = extract_text_fast(reference_path, 'ref')
    
    print(f"\n📝 OCR Results (top 10):")
    for i, text in enumerate(ref_texts[:10], 1):
        print(f"   {i}. {text.replace(chr(10), ' ')[:80]}")
    
    print(f"\n🔍 Extracting markings...")
    ref_part = extract_part_number(ref_texts)
    ref_mfg = extract_manufacturer(ref_texts)
    ref_date = extract_date_code(ref_texts)
    
    print(f"\n✓ Reference Markings:")
    print(f"   Part Number:  {ref_part or '❌ NOT FOUND'}")
    print(f"   Manufacturer: {ref_mfg or '❌ NOT FOUND'}")
    print(f"   Date Code:    {ref_date or '(optional)'}")
    
    if not ref_part:
        print(f"\n❌ FAILED: Cannot read reference IC")
        print(f"💡 Check ref_*.jpg debug images")
        return False
    
    # Extract test
    print(f"\n" + "="*70)
    print(f"📷 TEST IC: {test_path}")
    print("-"*70)
    
    if not os.path.exists(test_path):
        print(f"❌ File not found")
        return False
    
    test_texts = extract_text_fast(test_path, 'test')
    
    print(f"\n📝 OCR Results (top 10):")
    for i, text in enumerate(test_texts[:10], 1):
        print(f"   {i}. {text.replace(chr(10), ' ')[:80]}")
    
    print(f"\n🔍 Extracting markings...")
    test_part = extract_part_number(test_texts)
    test_mfg = extract_manufacturer(test_texts)
    test_date = extract_date_code(test_texts)
    
    print(f"\n✓ Test Markings:")
    print(f"   Part Number:  {test_part or '❌ NOT FOUND'}")
    print(f"   Manufacturer: {test_mfg or '❌ NOT FOUND'}")
    print(f"   Date Code:    {test_date or '(optional)'}")
    
    if not test_part:
        print(f"\n❌ FAILED: Cannot read test IC")
        print(f"💡 Check test_*.jpg debug images")
        return False
    
    # Compare
    print(f"\n" + "="*70)
    print(f"🔍 VERIFICATION")
    print("="*70)
    
    match = (test_part == ref_part)
    
    print(f"\n📊 Part Number Comparison:")
    print(f"   Reference: {ref_part}")
    print(f"   Test:      {test_part}")
    print(f"   Match:     {'✅ YES' if match else '❌ NO'}")
    
    if ref_mfg and test_mfg:
        mfg_match = (test_mfg == ref_mfg)
        print(f"\n   Manufacturer: {'✅ Match' if mfg_match else '⚠️  Mismatch'}")
    
    print(f"\n" + "="*70)
    if match:
        print(f"🎯 VERDICT: ✅ GENUINE")
        print(f"   Part numbers match!")
    else:
        print(f"🎯 VERDICT: ❌ COUNTERFEIT/MISMATCH")
        print(f"   Part numbers do NOT match!")
    print("="*70)
    
    return match


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    start = time.time()
    
    result = verify_ic(REFERENCE_IC_PATH, TEST_IC_PATH)
    
    elapsed = (time.time() - start) * 1000
    print(f"\n⚡ Total time: {elapsed:.0f}ms")
    
    if DEBUG_MODE:
        print(f"\n📁 Debug images: ref_*.jpg, test_*.jpg")
