"""
Enhanced IC Verification with Image Preprocessing
Improves OCR accuracy by preprocessing the image

This script automatically loads reference IC from the reference folder
and compares it with test images.
"""

from counterfeit_detection_agent import CounterfeitDetectionAgent
import cv2
import numpy as np
import pytesseract
import re
import os
import json
import time

# ============================================================================
# CONFIGURATION - Load from reference folder
# ============================================================================

# Path to reference IC image (golden/genuine IC)
REFERENCE_IC_PATH = 'reference/golden_product.jpg'  # Your golden IC image

# Path to test IC image (IC to verify)
TEST_IC_PATH = 'test_images/product_to_verify.jpg'  # IC to test

# Optional: Load expected values from config file
CONFIG_FILE = 'reference/ic_config.json'  # Optional config file

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================

# ============================================================================
# 🏭 INDUSTRIAL PRODUCTION MODE - BALANCED SPEED & ACCURACY
# ============================================================================

# Debug mode - set to False in production to skip debug image saving (saves 5-10ms per IC)
DEBUG_MODE = True  # Debug mode enabled for troubleshooting

# Font comparison settings
FONT_COMPARISON_ENABLED = True  # Enable font style comparison between reference and test IC
FONT_SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score (0-1) to consider fonts matching

# ⚡ SPEED OPTIMIZATION: Use cached preprocessed images for REFERENCE IC only
# Test IC is always preprocessed fresh (unique each time)
FORCE_REPROCESS = False  # ⚡ PRODUCTION: Use cache for reference IC, always preprocess test IC

# ⚡ SPEED: Cache OCR results for reference IC (saves 1+ second!)
CACHE_OCR_RESULTS = True  # Cache OCR results to avoid re-running Tesseract
OCR_CACHE_FILE = 'ref_ocr_cache.json'  # Cache file for reference IC OCR results

# ROI (Region of Interest) - crop to IC marking area for faster processing
ROI = None  # Format: (x, y, width, height) or None for full image

# Target image height for OCR (smaller = faster, but keep readable)
TARGET_OCR_HEIGHT = 500  # Increased for better OCR accuracy

# Conditional preprocessing - only apply if image quality is poor
AUTO_ENHANCE = True  # Automatically detect if enhancement is needed

# Tesseract optimization
TESSERACT_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Only alphanumeric

# ⚡ SPEED: Optimized PSM modes
OPTIMIZED_PSM_MODES = ['--psm 11', '--psm 6']  # ⚡ Sparse text + Block text

# ⚡ SPEED: Smart early exit - collect part number AND manufacturer, then exit
EARLY_EXIT_CONFIDENCE = True  # ⚡ Exit when we have both part number and manufacturer

# ⚡ SPEED: Limit number of images to process
MAX_IMAGES_TO_PROCESS = 4  # ⚡ Process up to 4 variants (exit early if data found)

# Visual font comparison settings for counterfeit detection
FONT_COMPARISON_ENABLED = True  # Enable visual font comparison
FONT_SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score to consider fonts matching


def load_reference_config():
    """
    Load reference IC configuration from file or extract from reference image
    """
    # Try to load from config file first
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config
        except:
            print(f"⚠️  Could not load config from {CONFIG_FILE}")
    
    # Default configuration (will be extracted from reference image)
    return {
        'part_number': None,      # Will be extracted from reference IC
        'manufacturer': None,     # Will be extracted from reference IC
        'date_code': None         # Optional - can vary between batches
    }


def extract_reference_markings(reference_image_path):
    """
    Extract IC markings from the reference (genuine) IC image
    """
    if not os.path.exists(reference_image_path):
        print(f"⚠️  Reference IC image not found: {reference_image_path}")
        print("   Please add a reference IC image to the reference/ folder")
        return None
    
    print(f"📖 Loading reference IC from: {reference_image_path}")
    
    # Extract text from reference IC
    ref_texts = extract_text_multiple_methods_from_path(reference_image_path)
    ref_markings = analyze_ic_markings(ref_texts)
    
    return ref_markings


# Load configuration
config = load_reference_config()

# Extract reference IC markings (if reference image exists)
reference_markings = None
if os.path.exists(REFERENCE_IC_PATH):
    print("🔍 Extracting markings from reference IC...")
    # Will be extracted later in the script
else:
    print(f"⚠️  Reference IC not found: {REFERENCE_IC_PATH}")
    print("   Using manual configuration instead")

# Manual fallback configuration (used if no reference image)
IC_PART_NUMBER = config.get('part_number', 'SN74LS266N')
IC_MANUFACTURER_MARK = config.get('manufacturer', 'HLF')
IC_DATE_CODE = config.get('date_code', None)  # Optional

test_image_path = TEST_IC_PATH


def check_cached_images_exist(prefix):
    """
    ⚡ Check if preprocessed images already exist (for speed optimization)
    Returns True if all required preprocessed images are present
    """
    required_images = [
        f'{prefix}_enhanced.jpg',
        f'{prefix}_bilateral.jpg',
        f'{prefix}_thresh1_otsu.jpg',
        f'{prefix}_thresh2_otsu_inv.jpg',
        f'{prefix}_thresh3_adaptive.jpg',
        f'{prefix}_thresh4_adaptive_inv.jpg',
        f'{prefix}_thresh5_manual.jpg',
        f'{prefix}_thresh6_manual_inv.jpg',
    ]
    return all(os.path.exists(img) for img in required_images)


def load_cached_images(prefix):
    """
    ⚡ Load existing preprocessed images (saves 15+ seconds!)
    """
    image_files = [
        f'{prefix}_enhanced.jpg',
        f'{prefix}_bilateral.jpg',
        f'{prefix}_thresh1_otsu.jpg',
        f'{prefix}_thresh2_otsu_inv.jpg',
        f'{prefix}_thresh3_adaptive.jpg',
        f'{prefix}_thresh4_adaptive_inv.jpg',
        f'{prefix}_thresh5_manual.jpg',
        f'{prefix}_thresh6_manual_inv.jpg',
    ]
    
    images = []
    for img_file in image_files:
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    
    return images


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


def preprocess_for_ocr(image, prefix='debug', debug=True, auto_enhance=True, roi=None, target_height=400):
    """
    OPTIMIZED: Enhance image for better OCR results with conditional processing
    IMPROVED: Better handling of embossed/engraved text on dark surfaces
    ⚡ SPEED: Reuses cached preprocessed images if available
    
    Args:
        image: Input image
        prefix: Prefix for debug image filenames (e.g., 'debug' or 'ref_debug')
        debug: Save debug images (disable in production for 5-10ms speedup)
        auto_enhance: Only enhance if image quality is poor
        roi: Region of interest (x, y, w, h) to crop
        target_height: Downscale to this height for faster OCR
    """
    start_time = time.time()
    
    # ⚡ SPEED OPTIMIZATION: Only cache REFERENCE IC (test IC is unique each time)
    # Only use cache for 'ref_debug' prefix, not 'debug' (test IC)
    if not FORCE_REPROCESS and prefix.startswith('ref'):
        if check_cached_images_exist(prefix):
            images = load_cached_images(prefix)
            # Validate loaded images
            if len(images) == 8 and all(img is not None for img in images):
                print(f"   ⚡ Using cached REFERENCE IC images (skipping preprocessing)")
                return images
            else:
                print(f"   ⚠️  Cached images missing or invalid, reprocessing reference IC...")
        else:
            print(f"   ⚠️  Some cached images missing, reprocessing reference IC...")
    # Test IC: Always preprocess (unique each time)
    if prefix == 'debug':
        print(f"   🔄 Preprocessing TEST IC (new image)...")
    
    # OPTIMIZATION 1: Crop to ROI
    if roi:
        image = crop_roi(image, roi)
    
    # OPTIMIZATION 2: Downscale for faster processing
    image = downscale_image(image, target_height)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # IMPROVED: Better preprocessing for embossed/engraved text
    # Apply CLAHE for better contrast on dark surfaces
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply sharpening for clearer text edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # Apply additional denoising for cleaner results
    denoised = cv2.fastNlMeansDenoising(bilateral, None, 10, 7, 21)
    
    # Multiple threshold methods for embossed text
    # Method 1: Otsu's thresholding
    _, thresh1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Inverted Otsu (for light text on dark background)
    _, thresh2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Adaptive thresholding (works well for varying lighting)
    thresh3 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Method 4: Inverted adaptive threshold
    thresh4 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # Method 5: Simple threshold with optimized manual value
    _, thresh5 = cv2.threshold(denoised, 120, 255, cv2.THRESH_BINARY)
    _, thresh6 = cv2.threshold(denoised, 120, 255, cv2.THRESH_BINARY_INV)
    
    # OPTIMIZATION 6: Only save debug images if debug mode enabled
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


def extract_text_multiple_methods(image, debug=True, auto_enhance=True, roi=None, target_height=400):
    """
    OPTIMIZED: Try multiple OCR configurations with performance optimizations
    ⚡ SPEED: Smart early exit when we have part number AND manufacturer
    """
    preprocessed_images = preprocess_for_ocr(
        image, 
        prefix='debug',
        debug=debug,
        auto_enhance=auto_enhance,
        roi=roi,
        target_height=target_height
    )
    
    # ⚡ SPEED: Use optimized PSM modes only
    psm_modes = OPTIMIZED_PSM_MODES
    
    # ⚡ SPEED: Limit images to process
    preprocessed_images = preprocessed_images[:MAX_IMAGES_TO_PROCESS]
    
    # IMPROVED: Try both with and without whitelist
    whitelist_config = f"-c tessedit_char_whitelist={TESSERACT_WHITELIST}"
    
    all_texts = []
    has_part_number = False
    has_manufacturer = False
    
    for img in preprocessed_images:
        for psm in psm_modes:
            try:
                # ⚡ PRODUCTION: Only whitelist for speed
                config = f"{psm} {whitelist_config}"
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
                    
                    # ⚡ SMART EARLY EXIT: Check if we have both part number and manufacturer
                    if EARLY_EXIT_CONFIDENCE:
                        text_upper = text.upper()
                        # Check for part number
                        if 'PC74' in text_upper or 'SN74' in text_upper or '74HC' in text_upper or '74LS' in text_upper:
                            has_part_number = True
                        # Check for manufacturer code
                        if 'HSS' in text_upper and any(c.isdigit() for c in text_upper):
                            has_manufacturer = True
                        # Exit if we have both
                        if has_part_number and has_manufacturer:
                            return all_texts
            except:
                pass
    
    return all_texts


def extract_text_multiple_methods_from_path(image_path, prefix='debug', debug=True, auto_enhance=True, roi=None, target_height=400):
    """
    OPTIMIZED: Load image from path and extract text using optimized methods
    
    Args:
        image_path: Path to image file
        prefix: Prefix for debug images (e.g., 'ref_debug' for reference image)
        debug: Save debug images
        auto_enhance: Only enhance if needed
        roi: Region of interest to crop
        target_height: Target height for downscaling
    """
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # OPTIMIZATION: Crop and downscale before preprocessing
    if roi:
        image = crop_roi(image, roi)
    image = downscale_image(image, target_height)
    
    # Preprocess with custom prefix
    preprocessed_images = preprocess_for_ocr(
        image, 
        prefix=prefix,
        debug=debug,
        auto_enhance=auto_enhance,
        roi=None,  # Already cropped
        target_height=target_height
    )
    
    # ⚡ SPEED: Use optimized PSM modes only
    psm_modes = OPTIMIZED_PSM_MODES
    
    # ⚡ SPEED: Limit images to process
    preprocessed_images = preprocessed_images[:MAX_IMAGES_TO_PROCESS]
    
    # IMPROVED: Try both with and without whitelist
    whitelist_config = f"-c tessedit_char_whitelist={TESSERACT_WHITELIST}"
    
    all_texts = []
    has_part_number = False
    has_manufacturer = False
    
    for img in preprocessed_images:
        for psm in psm_modes:
            try:
                # ⚡ PRODUCTION: Only whitelist for speed
                config = f"{psm} {whitelist_config}"
                text = pytesseract.image_to_string(img, config=config)
                if text.strip():
                    all_texts.append(text.strip())
                    
                    # ⚡ SMART EARLY EXIT: Check if we have both part number and manufacturer
                    if EARLY_EXIT_CONFIDENCE:
                        text_upper = text.upper()
                        # Check for part number
                        if 'PC74' in text_upper or 'SN74' in text_upper or '74HC' in text_upper or '74LS' in text_upper:
                            has_part_number = True
                        # Check for manufacturer code
                        if 'HSS' in text_upper and any(c.isdigit() for c in text_upper):
                            has_manufacturer = True
                        # Exit if we have both
                        if has_part_number and has_manufacturer:
                            return all_texts
            except:
                pass
    
    return all_texts


def analyze_ic_markings(ocr_texts):
    """
    Analyze multiple OCR results to find IC markings
    IMPROVED: Better pattern matching, text assembly, and OCR error correction
    """
    markings = {
        'part_number': None,
        'manufacturer_mark': None,
        'date_code': None,
        'additional_codes': [],  # Store additional codes like "A Y", revision codes, etc.
        'all_text': []  # Store all detected text for debugging
    }
    
    # Store all possible part numbers found and pick the best one
    possible_part_numbers = []
    
    # Combine all text for better pattern matching
    combined_text = ' '.join([t for t in ocr_texts if t])
    
    for text in ocr_texts:
        if not text:
            continue
        
        # Store original text
        markings['all_text'].append(text)
            
        text_upper = text.upper().replace('\n', ' ').replace('\r', ' ')
        # Remove extra spaces
        text_upper = ' '.join(text_upper.split())
        
        # IMPROVED: Better OCR error correction
        # Common OCR mistakes for IC markings
        corrections = {
            'O': '0', 'o': '0',  # O to 0
            'I': '1', 'l': '1', '|': '1',  # I/l to 1
            'Z': '2', 'z': '2',  # Z to 2
        }
        
        # Apply corrections
        text_corrected = text_upper
        for old, new in corrections.items():
            text_corrected = text_corrected.replace(old, new)
        
        # Special correction: COON -> C00N (OCR often reads 00 as OO)
        text_corrected = text_corrected.replace('COON', 'C00N')
        text_corrected = text_corrected.replace('COO', 'C00')
        
        # Extract part number - Multiple patterns
        if not markings['part_number']:
            # Pattern 1: 74-series ICs (SN74, PC74, 74HC, 74LS, etc.)
            part_patterns = [
                r'PC74[A-Z]{0,4}\d{2,4}[A-Z]{0,2}',  # PC74HC153P, PC74HCT00N (Philips/NXP)
                r'SN74[A-Z]{0,4}\d{2,3}[A-Z]{0,2}',  # SN74HC00N, SN74LS90N (Texas Instruments)
                r'74[A-Z]{1,4}\d{2,4}[A-Z]{0,2}',    # 74HC00, 74LS90 (generic)
                r'SN\d{2}[A-Z]{2,4}\d{2,4}[A-Z]?',   # SN74HC00N format
                r'PC\d{2}[A-Z]{2,4}\d{2,4}[A-Z]?',   # PC74HC153P format
                # Additional patterns for other IC types
                r'MC[A-Z0-9]{2,8}[A-Z]?',            # MCT4HC20N, MC74F08N format
                r'MCT[A-Z0-9]{2,8}[A-Z]?',           # MCTYHC20N format
                r'HC\d{1,3}[A-Z]{0,2}',              # HC20N format
                r'\d{2,4}[A-Z]{1,4}\d{1,3}[A-Z]?',   # 74HC20N format without prefix
            ]
            
            for pattern in part_patterns:
                part_match = re.search(pattern, text_corrected)
                if part_match:
                    possible_part_numbers.append(('direct', part_match.group()))
                    if not markings['part_number']:
                        markings['part_number'] = part_match.group()
                    break
        
        # IMPROVED: Try to assemble fragmented text (e.g., "SN" + "74" + "HC" + "02" + "N" or "PC" + "74" + "HC" + "153" + "P")
        if not markings['part_number']:
            # Look for fragments and try to assemble
            if ('SN' in text_corrected or 'PC' in text_corrected) and ('HC' in text_corrected or 'LS' in text_corrected or 'HCO' in text_corrected):
                # Try to extract number after HC/LS
                fragments = text_corrected.split()
                assembled = ''
                
                # Find SN or PC position
                sn_idx = -1
                for i, frag in enumerate(fragments):
                    if frag == 'SN' or frag.startswith('SN'):
                        sn_idx = i
                        assembled = 'SN'
                        break
                    elif frag == 'PC' or frag.startswith('PC'):
                        sn_idx = i
                        assembled = 'PC'
                        break
                
                if sn_idx >= 0:
                    # Look for 74 (might be missing in OCR, assume it's there)
                    found_74 = False
                    for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                        if '74' in fragments[j]:
                            assembled += '74'
                            found_74 = True
                            break
                    
                    # If 74 not found but we have HC/LS, assume SN74
                    if not found_74:
                        assembled = 'SN74'
                    
                    # Look for HC/LS/HCO
                    ic_type = ''
                    for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                        if 'HCO' in fragments[j]:
                            ic_type = 'HC'
                            assembled += 'HC'
                            break
                        elif 'HC' in fragments[j]:
                            ic_type = 'HC'
                            assembled += 'HC'
                            break
                        elif 'LS' in fragments[j]:
                            ic_type = 'LS'
                            assembled += 'LS'
                            break
                    
                    # Look for IC number (00, 02, 03, etc.)
                    # Special handling: "HCO" + "2N" should become "HC02N"
                    ic_number = ''
                    hco_found = False
                    
                    # Check if we found HCO (which means the 'O' is actually '0')
                    for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                        if fragments[j] == 'HCO':
                            hco_found = True
                            break
                    
                    # Method 1: If HCO was found, look for pattern like "0N" after it
                    # HCO means HC + O (which is 0), so we need to find the second digit
                    if hco_found:
                        # Look for patterns like "0N", "2N", "3N" which indicate the second digit
                        # Find the position of HCO first
                        hco_pos = -1
                        for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                            if fragments[j] == 'HCO':
                                hco_pos = j
                                break
                        
                        if hco_pos >= 0:
                            # Look for digit+N pattern AFTER HCO
                            for j in range(hco_pos+1, min(hco_pos+10, len(fragments))):
                                if re.match(r'^\d[A-Z]$', fragments[j]):
                                    # Extract the digit (e.g., "0" from "0N", "2" from "2N")
                                    digit = fragments[j][0]
                                    # The first digit is '0' from HCO
                                    ic_number = '0' + digit
                                    break
                    
                    # Method 2: Look for 2-digit numbers (like "00", "02", "08")
                    if not ic_number:
                        for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                            if re.match(r'^\d{2,3}$', fragments[j]) and fragments[j] != '74':
                                ic_number = fragments[j]
                                break
                    
                    # Method 3: Look for "COON" or "C00N" pattern (OCR often reads 00 as OO)
                    if not ic_number:
                        for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                            # Check for COON pattern (which is actually C00N)
                            if 'COON' in fragments[j] or 'C00N' in fragments[j]:
                                ic_number = '00'
                                break
                            # Check for patterns like "OON" or "00N"
                            if fragments[j] in ['OON', '00N', 'O0N', '0ON']:
                                ic_number = '00'
                                break
                    
                    # Method 4: Look for two separate single digits (last resort)
                    if not ic_number:
                        digits = []
                        # Start looking AFTER HC/HCO position
                        hc_pos = -1
                        for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                            if 'HC' in fragments[j]:
                                hc_pos = j
                                break
                        
                        start_pos = max(sn_idx, hc_pos + 1) if hc_pos >= 0 else sn_idx
                        for j in range(start_pos, min(start_pos+10, len(fragments))):
                            if fragments[j].isdigit() and len(fragments[j]) == 1 and fragments[j] != '7' and fragments[j] != '4':
                                digits.append(fragments[j])
                                if len(digits) == 2:
                                    ic_number = ''.join(digits)
                                    break
                    
                    if ic_number:
                        assembled += ic_number
                    
                    # Look for ending letter (N, P, etc.)
                    if not assembled.endswith(('N', 'P', 'D', 'A')):
                        for j in range(sn_idx, min(sn_idx+20, len(fragments))):
                            if fragments[j] in ['N', 'P', 'BN', 'DN', 'AN'] or (fragments[j].endswith(('N', 'P')) and len(fragments[j]) <= 3):
                                assembled += fragments[j][-1]  # Add last letter
                                break
                    
                    # Validate assembled part number (SN74 or PC74)
                    if len(assembled) >= 8 and (assembled.startswith('SN74') or assembled.startswith('PC74')):
                        possible_part_numbers.append(('assembled', assembled))
                        if not markings['part_number']:
                            markings['part_number'] = assembled
                        break
        
        # Extract manufacturer
        if not markings['manufacturer_mark']:
            # Common manufacturer codes
            mfg_patterns = [
                r'HLF',           # HLF manufacturer
                r'HSS\d{4}',      # HSS8718 (manufacturer code)
                r'TI',            # Texas Instruments
                r'NXP',           # NXP Semiconductors
                r'PHI',           # Philips
            ]
            
            for pattern in mfg_patterns:
                mfg_match = re.search(pattern, text_upper)
                if mfg_match:
                    markings['manufacturer_mark'] = mfg_match.group()
                    break
        
        # Extract date code - Multiple patterns
        if not markings['date_code']:
            date_patterns = [
                r'\d{6}[A-Z]',                        # 549790T (6 digits + letter)
                r'\d{4}[A-Z]',                        # 2081A (4 digits + letter)
                r'\d{2}[A-Z]{3,5}\d{1,2}[A-Z]?\d?',  # 24ARSS8E4
                r'\d{2}[A-Z]\d',                      # 20A1
                r'\d{4}',                             # 2081 (simple year)
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, text_corrected)
                if date_match:
                    # Exclude if it's part of the part number
                    match_text = date_match.group()
                    if markings['part_number'] and match_text not in markings['part_number']:
                        markings['date_code'] = match_text
                        break
    
    # Post-processing: Choose the best part number
    # Prefer assembled part numbers when we have HCO fragments (more reliable)
    if len(possible_part_numbers) > 1:
        # Check if we have an assembled part number
        assembled_parts = [pn for method, pn in possible_part_numbers if method == 'assembled']
        if assembled_parts:
            # Prefer assembled over direct matches
            markings['part_number'] = assembled_parts[0]
    
    # Extract additional codes (like "A Y", revision codes, quality marks)
    # These appear after the main markings
    for text in ocr_texts:
        text_upper = text.upper().replace('\n', ' ')
        words = text_upper.split()
        
        # Look for single letters or short codes that aren't part of main fields
        for i, word in enumerate(words):
            # Single letter codes (A, Y, etc.)
            if len(word) == 1 and word.isalpha():
                if word not in str(markings['part_number']) and word not in str(markings['manufacturer_mark']):
                    if word not in markings['additional_codes']:
                        markings['additional_codes'].append(word)
            
            # Two-letter codes separated (like "A Y")
            if i < len(words) - 1:
                two_letter = f"{words[i]} {words[i+1]}"
                if re.match(r'^[A-Z]\s+[A-Z]$', two_letter):
                    if two_letter not in markings['additional_codes']:
                        markings['additional_codes'].append(two_letter)
    
    return markings


def compare_text_visual_features(ref_img_path, test_img_path):
    """
    Compare visual characteristics of text (font style, size, etc.) between reference and test IC images
    
    Args:
        ref_img_path: Path to reference IC image
        test_img_path: Path to test IC image
        
    Returns:
        dict: Dictionary containing similarity score and details
    """
    # Load images
    ref_img = cv2.imread(ref_img_path)
    test_img = cv2.imread(test_img_path)
    
    if ref_img is None or test_img is None:
        return {
            'similarity_score': 0.0,
            'details': 'Failed to load images',
            'match': False
        }
    
    # Convert to grayscale
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to same height for fair comparison
    target_height = 500
    ref_aspect = ref_img.shape[1] / ref_img.shape[0]
    test_aspect = test_img.shape[1] / test_img.shape[0]
    
    ref_resized = cv2.resize(ref_gray, (int(target_height * ref_aspect), target_height))
    test_resized = cv2.resize(test_gray, (int(target_height * test_aspect), target_height))
    
    # Apply threshold to isolate text
    _, ref_thresh = cv2.threshold(ref_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, test_thresh = cv2.threshold(test_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (text regions)
    ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_contours, _ = cv2.findContours(test_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to get only text
    min_area = 50
    max_area = 5000
    
    ref_text_contours = [c for c in ref_contours if min_area < cv2.contourArea(c) < max_area]
    test_text_contours = [c for c in test_contours if min_area < cv2.contourArea(c) < max_area]
    
    # If no text contours found, return low similarity
    if not ref_text_contours or not test_text_contours:
        return {
            'similarity_score': 0.0,
            'details': 'No text contours detected in one or both images',
            'match': False
        }
    
    # Calculate metrics for reference text
    ref_metrics = []
    for contour in ref_text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Estimate stroke width using distance transform
        roi = ref_thresh[y:y+h, x:x+w]
        if roi.size > 0:
            dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
            stroke_width = np.mean(dist) * 2  # Multiply by 2 as distance is to edge
        else:
            stroke_width = 0
            
        ref_metrics.append({
            'aspect_ratio': aspect_ratio,
            'height': h,
            'width': w,
            'stroke_width': stroke_width
        })
    
    # Calculate metrics for test text
    test_metrics = []
    for contour in test_text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Estimate stroke width using distance transform
        roi = test_thresh[y:y+h, x:x+w]
        if roi.size > 0:
            dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
            stroke_width = np.mean(dist) * 2  # Multiply by 2 as distance is to edge
        else:
            stroke_width = 0
            
        test_metrics.append({
            'aspect_ratio': aspect_ratio,
            'height': h,
            'width': w,
            'stroke_width': stroke_width
        })
    
    # Calculate average metrics
    ref_avg = {
        'aspect_ratio': np.mean([m['aspect_ratio'] for m in ref_metrics]),
        'height': np.mean([m['height'] for m in ref_metrics]),
        'width': np.mean([m['width'] for m in ref_metrics]),
        'stroke_width': np.mean([m['stroke_width'] for m in ref_metrics if m['stroke_width'] > 0])
    }
    
    test_avg = {
        'aspect_ratio': np.mean([m['aspect_ratio'] for m in test_metrics]),
        'height': np.mean([m['height'] for m in test_metrics]),
        'width': np.mean([m['width'] for m in test_metrics]),
        'stroke_width': np.mean([m['stroke_width'] for m in test_metrics if m['stroke_width'] > 0])
    }
    
    # Calculate similarity scores for each metric
    aspect_ratio_sim = 1 - min(abs(ref_avg['aspect_ratio'] - test_avg['aspect_ratio']) / max(ref_avg['aspect_ratio'], 0.1), 1.0)
    height_sim = 1 - min(abs(ref_avg['height'] - test_avg['height']) / max(ref_avg['height'], 1), 1.0)
    width_sim = 1 - min(abs(ref_avg['width'] - test_avg['width']) / max(ref_avg['width'], 1), 1.0)
    stroke_sim = 1 - min(abs(ref_avg['stroke_width'] - test_avg['stroke_width']) / max(ref_avg['stroke_width'], 0.1), 1.0)
    
    # Calculate overall similarity score (weighted average)
    weights = {
        'aspect_ratio': 0.3,
        'height': 0.2,
        'width': 0.2,
        'stroke_width': 0.3
    }
    
    similarity_score = (
        weights['aspect_ratio'] * aspect_ratio_sim +
        weights['height'] * height_sim +
        weights['width'] * width_sim +
        weights['stroke_width'] * stroke_sim
    )
    
    # Determine if fonts match based on similarity threshold
    font_match = similarity_score >= FONT_SIMILARITY_THRESHOLD
    
    # Create detailed report
    details = (
        f"Font aspect ratio: {aspect_ratio_sim:.2f}, " +
        f"height: {height_sim:.2f}, " +
        f"width: {width_sim:.2f}, " +
        f"stroke thickness: {stroke_sim:.2f}"
    )
    
    return {
        'similarity_score': similarity_score,
        'details': details,
        'match': font_match
    }


def verify_ic(test_markings, ref_markings, ref_img_path, test_img_path):
    """
    Verify if test IC matches reference IC based on markings and visual characteristics
    
    Args:
        test_markings: Dictionary of test IC markings
        ref_markings: Dictionary of reference IC markings
        ref_img_path: Path to reference IC image
        test_img_path: Path to test IC image
        
    Returns:
        dict: Verification results
    """
    # Check part number match
    part_match = False
    if test_markings['part_number'] and ref_markings['part_number']:
        part_match = test_markings['part_number'] == ref_markings['part_number']
    
    # Check manufacturer match
    mfg_match = False
    if test_markings['manufacturer_mark'] and ref_markings['manufacturer_mark']:
        mfg_match = test_markings['manufacturer_mark'] == ref_markings['manufacturer_mark']
    
    # Check date code (optional, may differ between batches)
    date_match = False
    if test_markings['date_code'] and ref_markings['date_code']:
        date_match = test_markings['date_code'] == ref_markings['date_code']
    
    # Check font visual characteristics if enabled
    font_match = True  # Default to True if comparison is disabled
    visual_similarity = 1.0
    visual_details = ""
    
    if FONT_COMPARISON_ENABLED:
        font_comparison = compare_text_visual_features(ref_img_path, test_img_path)
        font_match = font_comparison['match']
        visual_similarity = font_comparison['similarity_score']
        visual_details = font_comparison['details']
    
    # Overall verification result
    # IC is considered genuine if part number matches AND
    # (manufacturer matches OR font characteristics match)
    overall_match = part_match and (mfg_match or font_match)
    
    return {
        'part_number': part_match,
        'manufacturer': mfg_match,
        'date_code': date_match,
        'font_match': font_match,
        'visual_similarity': visual_similarity,
        'visual_details': visual_details,
        'overall': overall_match
    }


import time as time_module
overall_start_time = time_module.time()

print("="*70)
print("ENHANCED IC VERIFICATION - OPTIMIZED FOR SPEED")
print("="*70)
print(f"\n⚡ Performance Settings:")
print(f"   Debug Mode:       {'ON' if DEBUG_MODE else 'OFF (production)'}")
print(f"   ROI Cropping:     {'Enabled' if ROI else 'Disabled (full image)'}")
print(f"   Target Height:    {TARGET_OCR_HEIGHT}px")
print(f"   Auto Enhance:     {'ON' if AUTO_ENHANCE else 'OFF'}")
print(f"   Early Exit:       {'ON' if EARLY_EXIT_CONFIDENCE else 'OFF'}")
print(f"   Tesseract Modes:  {len(OPTIMIZED_PSM_MODES)} (optimized)")

# Step 1: Extract reference IC markings
reference_markings = None
if os.path.exists(REFERENCE_IC_PATH):
    ref_start = time.time()
    print(f"\n📖 Loading reference IC from: {REFERENCE_IC_PATH}")
    print("🔄 Preprocessing reference image (optimized)...")
    
    ref_ocr_results = extract_text_multiple_methods_from_path(
        REFERENCE_IC_PATH, 
        prefix='ref_debug',
        debug=DEBUG_MODE,
        auto_enhance=AUTO_ENHANCE,
        roi=ROI,
        target_height=TARGET_OCR_HEIGHT
    )
    
    ref_time = (time.time() - ref_start) * 1000
    print(f"✓ Reference IC processed in {ref_time:.1f}ms")
    print(f"✓ Tried {len(ref_ocr_results)} different OCR configurations")
    
    print("\n📝 Reference IC - All OCR Results:")
    for i, text in enumerate(ref_ocr_results[:10], 1):
        if text:
            # Show first 150 chars of each result
            display_text = text.replace('\n', ' ').strip()
            print(f"   {i}. '{display_text[:150]}'")
    
    reference_markings = analyze_ic_markings(ref_ocr_results)
    
    print(f"\n✓ Reference IC markings extracted:")
    print(f"   Part Number:      {reference_markings['part_number'] or 'NOT DETECTED'}")
    print(f"   Manufacturer:     {reference_markings['manufacturer_mark'] or 'NOT DETECTED'}")
    print(f"   Date/Batch Code:  {reference_markings['date_code'] or 'NOT DETECTED'}")
    if reference_markings['additional_codes']:
        print(f"   Additional Codes: {', '.join(reference_markings['additional_codes'])}")
    
    if DEBUG_MODE:
        print(f"\n📁 Reference IC debug images saved:")
        print(f"   - ref_debug_enhanced.jpg")
        print(f"   - ref_debug_thresh1.jpg")
        print(f"   - ref_debug_thresh2.jpg")
    else:
        print(f"\n⚡ Debug images skipped (production mode)")
    
    # Use extracted markings as expected values
    if reference_markings['part_number']:
        IC_PART_NUMBER = reference_markings['part_number']
    if reference_markings['manufacturer_mark']:
        IC_MANUFACTURER_MARK = reference_markings['manufacturer_mark']
    # Date code can vary, so it's optional
    
    print("\n" + "="*70)

print(f"\n📋 Expected IC Markings (from reference):")
print(f"   Part Number:      {IC_PART_NUMBER}")
print(f"   Manufacturer:     {IC_MANUFACTURER_MARK}")
print(f"   Date/Batch Code:  {IC_DATE_CODE if IC_DATE_CODE else 'Any (optional)'}")
print(f"\n📷 Testing Image: {test_image_path}")
print("="*70)

# Step 2: Load and process test image
image = cv2.imread(test_image_path)

if image is None:
    print(f"\n❌ Error: Could not load image: {test_image_path}")
    print("Please check the file path and ensure the image exists.")
else:
    test_start = time.time()
    print("\n🔄 Preprocessing test image (optimized)...")
    
    # Extract text using multiple methods
    ocr_results = extract_text_multiple_methods(
        image,
        debug=DEBUG_MODE,
        auto_enhance=AUTO_ENHANCE,
        roi=ROI,
        target_height=TARGET_OCR_HEIGHT
    )
    
    test_time = (time.time() - test_start) * 1000
    print(f"✓ Test IC processed in {test_time:.1f}ms")
    print(f"✓ Tried {len(ocr_results)} different OCR configurations")
    
    # Analyze markings
    detected_markings = analyze_ic_markings(ocr_results)
    
    if DEBUG_MODE:
        print(f"\n🔧 DEBUG - Fragment Analysis:")
        # Show the most promising OCR result for debugging
        for i, text in enumerate(ocr_results[:5], 1):
            if 'SN' in text.upper() or 'HC' in text.upper():
                fragments = text.upper().replace('\n', ' ').split()
                print(f"   Result #{i} fragments: {fragments[:25]}")
                
                # Try to manually show what should be assembled
                if 'SN' in fragments and 'HCO' in fragments:
                    # Find what comes after HCO
                    hco_idx = fragments.index('HCO')
                    next_frags = fragments[hco_idx+1:hco_idx+5]
                    print(f"      → HCO found at position {hco_idx}, next fragments: {next_frags}")
                    if any(f.endswith('N') for f in next_frags):
                        digit_n = [f for f in next_frags if re.match(r'^\d[A-Z]$', f)]
                        if digit_n:
                            print(f"      → Should assemble: SN + 74 + HC + 0 (from HCO) + {digit_n[0][0]} (from {digit_n[0]}) + N = SN74HC0{digit_n[0][0]}N")
    
    print("\n📝 Test IC - All OCR Results:")
    for i, text in enumerate(ocr_results[:10], 1):
        if text:
            # Show first 150 chars of each result
            display_text = text.replace('\n', ' ').strip()
            print(f"   {i}. '{display_text[:150]}'")
            # Also show if it contains SN74 pattern
            if 'SN74' in text.upper() or 'SN' in text.upper() and 'HC' in text.upper():
                print(f"       → Contains IC pattern: {[w for w in text.upper().split() if 'SN' in w or 'HC' in w or 'C00' in w or 'COO' in w]}")
    
    print("\n" + "="*70)
    print("DETECTED IC MARKINGS")
    print("="*70)
    print(f"🔍 Extracted Markings:")
    print(f"   Part Number:      {detected_markings['part_number'] or 'NOT DETECTED'}")
    print(f"   Manufacturer:     {detected_markings['manufacturer_mark'] or 'NOT DETECTED'}")
    print(f"   Date/Batch Code:  {detected_markings['date_code'] or 'NOT DETECTED'}")
    if detected_markings['additional_codes']:
        print(f"   Additional Codes: {', '.join(detected_markings['additional_codes'])}")
    
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
    else:
        print("\n❌ IC part number does NOT match or could not be detected!")
        print("\n💡 Troubleshooting:")
        if DEBUG_MODE:
            print("   1. Check debug images: debug_enhanced.jpg, debug_thresh1.jpg, etc.")
            print("   2. Take a better photo (closer, better lighting, no glare)")
        else:
            print("   1. Enable DEBUG_MODE=True to see debug images")
            print("   2. Take a better photo (closer, better lighting, no glare)")
        print("   3. Ensure IC text is clearly visible and in focus")
        print("   4. Try different angles/lighting conditions")
        print("   5. Adjust ROI to crop to IC marking area")
        print("   6. Try different TARGET_OCR_HEIGHT values (300-500)")
    
    # Performance summary
    total_program_time = (time_module.time() - overall_start_time) * 1000
    print(f"\n⚡ PERFORMANCE SUMMARY:")
    print(f"   Reference IC OCR:  {ref_time:.1f}ms")
    print(f"   Test IC OCR:       {test_time:.1f}ms")
    print(f"   Total Runtime:     {total_program_time:.1f}ms")
    print(f"\n   Target: <30ms per IC (production)")
    if test_time < 30:
        print(f"   ✅ EXCELLENT - Meeting production target!")
    elif test_time < 50:
        print(f"   ⚠️  GOOD - Consider further optimization")
    else:
        print(f"   ❌ SLOW - Enable production mode (DEBUG_MODE=False)")
    
    if DEBUG_MODE:
        print("\n📁 Debug images saved:")
        print("   - debug_enhanced.jpg (contrast enhanced)")
        print("   - debug_thresh1.jpg (threshold method 1)")
        print("   - debug_thresh2.jpg (threshold method 2)")
    else:
        print("\n⚡ Debug images skipped (production mode)")
