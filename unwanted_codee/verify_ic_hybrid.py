"""
Hybrid IC Verification - EasyOCR + Tesseract
Uses EasyOCR (better for industrial text) with Tesseract fallback
"""

import cv2
import numpy as np
import re
import os
import time

# Try to import EasyOCR, fallback to Tesseract only
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✓ EasyOCR available")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not available, using Tesseract only")

import pytesseract

# ============================================================================
# CONFIGURATION
# ============================================================================

REFERENCE_IC_PATH = 'reference/golden_product.jpg'
TEST_IC_PATH = 'test_images/product_to_verify.jpg'
DEBUG_MODE = True

# ============================================================================
# PREPROCESSING
# ============================================================================

def detect_and_crop_ic(image):
    """Auto-detect and crop IC chip"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    
    return image[y:y+h, x:x+w]


def preprocess_ic(image):
    """Optimized preprocessing for IC text"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Upscale 3x
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    
    variants = []
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Variant 1: Adaptive threshold inverted (best for embossed text)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    variants.append(('adaptive_inv', adaptive))
    
    # Variant 2: Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
    _, gradient_thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(('gradient', gradient_thresh))
    
    # Variant 3: High contrast
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    _, high_contrast = cv2.threshold(normalized, 100, 255, cv2.THRESH_BINARY_INV)
    variants.append(('contrast', high_contrast))
    
    return variants, enhanced


# ============================================================================
# OCR EXTRACTION
# ============================================================================

def extract_with_easyocr(image):
    """Extract text using EasyOCR"""
    if not EASYOCR_AVAILABLE:
        return []
    
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(image)
        
        texts = []
        for (bbox, text, conf) in results:
            if conf > 0.1:  # Low threshold to catch everything
                texts.append(text)
        
        return texts
    except Exception as e:
        print(f"   EasyOCR error: {e}")
        return []


def extract_with_tesseract(image):
    """Extract text using Tesseract"""
    texts = []
    
    psm_modes = ['--psm 11', '--psm 6', '--psm 7']
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    for psm in psm_modes:
        try:
            # With whitelist
            config = f"{psm} -c tessedit_char_whitelist={whitelist}"
            text = pytesseract.image_to_string(image, config=config)
            if text.strip():
                texts.append(text.strip())
            
            # Without whitelist
            text2 = pytesseract.image_to_string(image, config=psm)
            if text2.strip() and text2.strip() not in texts:
                texts.append(text2.strip())
        except:
            pass
    
    return texts


def extract_text_from_ic(image_path, prefix='debug'):
    """Main text extraction"""
    print(f"📖 Loading: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load")
        return []
    
    # Crop to IC
    start = time.time()
    cropped = detect_and_crop_ic(image)
    print(f"✓ Cropped in {(time.time()-start)*1000:.0f}ms")
    
    # Preprocess
    start = time.time()
    variants, enhanced = preprocess_ic(cropped)
    print(f"✓ Preprocessed in {(time.time()-start)*1000:.0f}ms")
    
    # Save debug
    if DEBUG_MODE:
        cv2.imwrite(f'{prefix}_cropped.jpg', cropped)
        cv2.imwrite(f'{prefix}_enhanced.jpg', enhanced)
        for name, img in variants:
            cv2.imwrite(f'{prefix}_{name}.jpg', img)
    
    all_texts = []
    
    # Try EasyOCR first (better for industrial text)
    if EASYOCR_AVAILABLE:
        print(f"🔍 Running EasyOCR...")
        start = time.time()
        
        # Try on enhanced image
        easy_texts = extract_with_easyocr(enhanced)
        all_texts.extend(easy_texts)
        
        # Try on preprocessed variants
        for name, img in variants:
            easy_texts = extract_with_easyocr(img)
            all_texts.extend(easy_texts)
        
        print(f"✓ EasyOCR: {len(all_texts)} results in {(time.time()-start)*1000:.0f}ms")
    
    # Also try Tesseract
    print(f"🔍 Running Tesseract...")
    start = time.time()
    
    for name, img in variants:
        tess_texts = extract_with_tesseract(img)
        all_texts.extend(tess_texts)
    
    print(f"✓ Tesseract: {len(all_texts)} results in {(time.time()-start)*1000:.0f}ms")
    
    # Remove duplicates
    all_texts = list(set(all_texts))
    
    return all_texts


# ============================================================================
# PART NUMBER EXTRACTION
# ============================================================================

def extract_part_number(texts):
    """Extract IC part number from OCR results"""
    candidates = []
    
    print(f"   Analyzing {len(texts)} OCR results...")
    
    for text in texts:
        # Normalize
        text_upper = text.upper().replace('\n', ' ').replace('\r', ' ')
        text_upper = ' '.join(text_upper.split())
        
        # Pattern 1: SN74 series (Texas Instruments)
        sn74_patterns = [
            r'SN74[A-Z]{1,4}\d{2,3}[A-Z]?',
            r'SN\s*74\s*[A-Z]{1,4}\s*\d{2,3}\s*[A-Z]?',
        ]
        
        for pattern in sn74_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                clean = match.replace(' ', '')
                if 7 <= len(clean) <= 12:
                    candidates.append(clean)
                    print(f"   → Found SN74: {clean}")
        
        # Pattern 2: PC74 series (Philips/NXP)
        pc74_patterns = [
            r'PC74[A-Z]{1,4}\d{2,4}[A-Z]?',
            r'PC\s*74\s*[A-Z]{1,4}\s*\d{2,4}\s*[A-Z]?',
        ]
        
        for pattern in pc74_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                clean = match.replace(' ', '')
                if 7 <= len(clean) <= 12:
                    candidates.append(clean)
                    print(f"   → Found PC74: {clean}")
        
        # Pattern 3: Generic 74 series
        generic_patterns = [
            r'74[A-Z]{1,4}\d{2,4}[A-Z]?',
        ]
        
        for pattern in generic_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                if 5 <= len(match) <= 10:
                    candidates.append(match)
                    print(f"   → Found 74-series: {match}")
    
    # Vote for most common
    if candidates:
        from collections import Counter
        counter = Counter(candidates)
        winner = counter.most_common(1)[0][0]
        print(f"   ✓ Best match: {winner} (appeared {counter[winner]} times)")
        return winner
    
    return None


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_ic(ref_path, test_path):
    """Main verification"""
    print("="*70)
    print("HYBRID IC VERIFICATION - EasyOCR + Tesseract")
    print("="*70)
    
    # Reference
    print(f"\n📋 REFERENCE IC")
    print("-"*70)
    
    if not os.path.exists(ref_path):
        print(f"❌ Not found: {ref_path}")
        return False
    
    ref_texts = extract_text_from_ic(ref_path, 'ref')
    
    print(f"\n📝 OCR Results ({len(ref_texts)} total):")
    for i, text in enumerate(ref_texts[:10], 1):
        print(f"   {i}. {text[:70]}")
    
    print(f"\n🔍 Extracting part number...")
    ref_part = extract_part_number(ref_texts)
    
    print(f"\n✓ Reference IC: {ref_part or '❌ NOT DETECTED'}")
    
    if not ref_part:
        print(f"\n❌ Cannot read reference IC")
        print(f"💡 Check debug images: ref_*.jpg")
        return False
    
    # Test
    print(f"\n" + "="*70)
    print(f"📋 TEST IC")
    print("-"*70)
    
    if not os.path.exists(test_path):
        print(f"❌ Not found: {test_path}")
        return False
    
    test_texts = extract_text_from_ic(test_path, 'test')
    
    print(f"\n📝 OCR Results ({len(test_texts)} total):")
    for i, text in enumerate(test_texts[:10], 1):
        print(f"   {i}. {text[:70]}")
    
    print(f"\n🔍 Extracting part number...")
    test_part = extract_part_number(test_texts)
    
    print(f"\n✓ Test IC: {test_part or '❌ NOT DETECTED'}")
    
    if not test_part:
        print(f"\n❌ Cannot read test IC")
        print(f"💡 Check debug images: test_*.jpg")
        return False
    
    # Compare
    print(f"\n" + "="*70)
    print(f"🔍 VERIFICATION")
    print("="*70)
    
    match = (test_part == ref_part)
    
    print(f"\n   Reference: {ref_part}")
    print(f"   Test:      {test_part}")
    print(f"   Match:     {'✅ YES' if match else '❌ NO'}")
    
    print(f"\n" + "="*70)
    if match:
        print(f"✅ VERDICT: GENUINE")
        print(f"   Part numbers match!")
    else:
        print(f"❌ VERDICT: COUNTERFEIT/MISMATCH")
        print(f"   Part numbers do NOT match!")
    print("="*70)
    
    return match


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    start = time.time()
    
    result = verify_ic(REFERENCE_IC_PATH, TEST_IC_PATH)
    
    elapsed = (time.time() - start) * 1000
    print(f"\n⚡ Total: {elapsed:.0f}ms")
    
    if DEBUG_MODE:
        print(f"📁 Debug images: ref_*.jpg, test_*.jpg")
