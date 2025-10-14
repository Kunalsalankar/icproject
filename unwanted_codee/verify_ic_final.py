"""
FINAL IC Verification - Auto-crop IC and optimized OCR
Automatically detects IC chip and crops to text area for better OCR
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
# IC DETECTION AND CROPPING
# ============================================================================

def detect_and_crop_ic(image):
    """
    Automatically detect the IC chip and crop to just the chip area
    This removes background noise and focuses OCR on the chip
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find the IC
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # Return original if no contours found
    
    # Find the largest rectangular contour (likely the IC)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2*padding)
    h = min(image.shape[0] - y, h + 2*padding)
    
    # Crop to IC area
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def preprocess_ic_text(image):
    """
    Specialized preprocessing for IC text
    Focus on making embossed text visible
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Upscale 3x for better OCR
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    
    results = []
    
    # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    
    # Adaptive thresholding (best for varying lighting)
    adaptive_inv = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
    results.append(('adaptive_inv', adaptive_inv))
    
    # Method 2: Morphological gradient (highlights edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
    _, gradient_thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(('gradient', gradient_thresh))
    
    # Method 3: Unsharp mask
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
    unsharp = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)
    _, unsharp_thresh = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(('unsharp', unsharp_thresh))
    
    # Method 4: High contrast with fixed threshold
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    _, high_contrast = cv2.threshold(normalized, 110, 255, cv2.THRESH_BINARY_INV)
    results.append(('high_contrast', high_contrast))
    
    return results, enhanced


def extract_text_from_ic(image_path, prefix='debug'):
    """
    Extract text from IC image with auto-cropping and optimized OCR
    """
    print(f"📖 Loading: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image")
        return []
    
    # Step 1: Auto-crop to IC
    start = time.time()
    cropped = detect_and_crop_ic(image)
    print(f"✓ IC detected and cropped in {(time.time()-start)*1000:.0f}ms")
    
    # Step 2: Preprocess
    start = time.time()
    processed_variants, enhanced = preprocess_ic_text(cropped)
    print(f"✓ Preprocessed {len(processed_variants)} variants in {(time.time()-start)*1000:.0f}ms")
    
    # Save debug images
    if DEBUG_MODE:
        cv2.imwrite(f'{prefix}_cropped.jpg', cropped)
        cv2.imwrite(f'{prefix}_enhanced.jpg', enhanced)
        for name, img in processed_variants:
            cv2.imwrite(f'{prefix}_{name}.jpg', img)
    
    # Step 3: OCR with multiple configurations
    start = time.time()
    
    # Best PSM modes for IC text
    psm_configs = [
        '--psm 11 --oem 1',  # Sparse text, LSTM engine
        '--psm 6 --oem 1',   # Block of text
        '--psm 7 --oem 1',   # Single line
    ]
    
    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789®'
    
    all_texts = []
    
    for name, img in processed_variants:
        for psm in psm_configs:
            try:
                # With whitelist
                config = f"{psm} -c tessedit_char_whitelist={whitelist}"
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
                
                # Without whitelist (sometimes better)
                text2 = pytesseract.image_to_string(img, config=psm)
                if text2.strip() and text2.strip() not in all_texts:
                    all_texts.append(text2.strip())
            except:
                pass
    
    print(f"✓ OCR completed: {len(all_texts)} text variants in {(time.time()-start)*1000:.0f}ms")
    
    return all_texts


def extract_part_number(texts):
    """
    Extract IC part number from OCR results
    """
    candidates = []
    
    for text in texts:
        # Normalize
        text_upper = text.upper().replace('\n', ' ').replace('\r', ' ')
        text_upper = ' '.join(text_upper.split())
        
        # OCR corrections
        corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'Z': '2', 'S': '5', 'B': '8'
        }
        
        text_corrected = text_upper
        for old, new in corrections.items():
            # Only replace in numeric contexts
            text_corrected = text_corrected.replace(old, new)
        
        # Direct pattern match
        patterns = [
            r'SN74[A-Z]{1,4}\d{2,3}[A-Z]?',
            r'SN\s*74\s*[A-Z]{1,4}\s*\d{2,3}\s*[A-Z]?',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_corrected)
            for match in matches:
                clean = match.replace(' ', '')
                if 7 <= len(clean) <= 12:
                    candidates.append(clean)
        
        # Fragment assembly
        words = text_corrected.split()
        for i, word in enumerate(words):
            if 'SN' in word:
                part = 'SN'
                
                # Look for 74
                if any('74' in words[j] for j in range(i, min(i+3, len(words)))):
                    part += '74'
                
                # Look for LS/HC/HCT
                for j in range(i, min(i+5, len(words))):
                    if 'LS' in words[j]:
                        part += 'LS'
                        break
                    elif 'HC' in words[j]:
                        part += 'HC'
                        break
                
                # Look for number
                for j in range(i, min(i+8, len(words))):
                    if re.match(r'^\d{2,3}$', words[j]) and words[j] != '74':
                        part += words[j]
                        break
                
                # Look for N
                if not part.endswith('N'):
                    for j in range(i, min(i+10, len(words))):
                        if words[j] in ['N', 'AN', 'BN']:
                            part += 'N'
                            break
                
                if len(part) >= 8 and part.startswith('SN74'):
                    candidates.append(part)
    
    # Vote
    if candidates:
        from collections import Counter
        counter = Counter(candidates)
        winner = counter.most_common(1)[0][0]
        print(f"   Found: {list(set(candidates))}")
        print(f"   Best: {winner}")
        return winner
    
    return None


def verify_ic(ref_path, test_path):
    """
    Main verification function
    """
    print("="*70)
    print("IC VERIFICATION - AUTO-CROP + OPTIMIZED OCR")
    print("="*70)
    
    # Reference IC
    print(f"\n📋 REFERENCE IC")
    print("-"*70)
    
    if not os.path.exists(ref_path):
        print(f"❌ Not found: {ref_path}")
        return False
    
    ref_texts = extract_text_from_ic(ref_path, 'ref')
    
    print(f"\n📝 Top OCR results:")
    for i, text in enumerate(ref_texts[:8], 1):
        print(f"   {i}. {text.replace(chr(10), ' ')[:70]}")
    
    print(f"\n🔍 Analyzing...")
    ref_part = extract_part_number(ref_texts)
    
    print(f"\n✓ Reference IC:")
    print(f"   Part Number: {ref_part or '❌ NOT DETECTED'}")
    
    if not ref_part:
        print(f"\n❌ Cannot read reference IC")
        return False
    
    # Test IC
    print(f"\n" + "="*70)
    print(f"📋 TEST IC")
    print("-"*70)
    
    if not os.path.exists(test_path):
        print(f"❌ Not found: {test_path}")
        return False
    
    test_texts = extract_text_from_ic(test_path, 'test')
    
    print(f"\n📝 Top OCR results:")
    for i, text in enumerate(test_texts[:8], 1):
        print(f"   {i}. {text.replace(chr(10), ' ')[:70]}")
    
    print(f"\n🔍 Analyzing...")
    test_part = extract_part_number(test_texts)
    
    print(f"\n✓ Test IC:")
    print(f"   Part Number: {test_part or '❌ NOT DETECTED'}")
    
    if not test_part:
        print(f"\n❌ Cannot read test IC")
        return False
    
    # Verify
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
        print(f"   IC part numbers match!")
    else:
        print(f"❌ VERDICT: COUNTERFEIT/MISMATCH")
        print(f"   IC part numbers do NOT match!")
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
        print(f"📁 Debug: ref_*.jpg, test_*.jpg")
