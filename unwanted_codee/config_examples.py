"""
IC Verification - Configuration Examples
Copy the appropriate configuration to your verify_ic.py or verify_ic_enhanced.py
"""

# ============================================================================
# PRODUCTION MODE - Maximum Speed (2,000-3,000 ICs/min)
# ============================================================================
PRODUCTION_CONFIG = {
    'DEBUG_MODE': False,                    # Skip debug images (saves 5-10ms)
    'ROI': (100, 50, 800, 600),            # Crop to IC area - ADJUST COORDINATES
    'TARGET_OCR_HEIGHT': 300,               # Smaller = faster
    'AUTO_ENHANCE': True,                   # Skip enhancement on good images
    'TESSERACT_WHITELIST': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    'OPTIMIZED_PSM_MODES': ['--psm 6', '--psm 11'],
    'EARLY_EXIT_CONFIDENCE': True           # Stop on confident match
}

# ============================================================================
# TESTING MODE - Maximum Accuracy (1,000-1,500 ICs/min)
# ============================================================================
TESTING_CONFIG = {
    'DEBUG_MODE': True,                     # Save debug images
    'ROI': None,                            # Process full image
    'TARGET_OCR_HEIGHT': 500,               # Higher resolution
    'AUTO_ENHANCE': False,                  # Always apply full enhancement
    'TESSERACT_WHITELIST': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    'OPTIMIZED_PSM_MODES': ['--psm 6', '--psm 7', '--psm 11'],
    'EARLY_EXIT_CONFIDENCE': False          # Try all configurations
}

# ============================================================================
# BALANCED MODE - Good Speed & Accuracy (1,500-2,500 ICs/min)
# ============================================================================
BALANCED_CONFIG = {
    'DEBUG_MODE': True,                     # Enable during initial testing
    'ROI': None,                            # Set after identifying IC location
    'TARGET_OCR_HEIGHT': 400,               # Good balance
    'AUTO_ENHANCE': True,                   # Conditional enhancement
    'TESSERACT_WHITELIST': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    'OPTIMIZED_PSM_MODES': ['--psm 6', '--psm 11'],
    'EARLY_EXIT_CONFIDENCE': True           # Early exit enabled
}

# ============================================================================
# HIGH ACCURACY MODE - For difficult ICs (800-1,200 ICs/min)
# ============================================================================
HIGH_ACCURACY_CONFIG = {
    'DEBUG_MODE': True,                     # Save debug images
    'ROI': None,                            # Process full image
    'TARGET_OCR_HEIGHT': 600,               # Very high resolution
    'AUTO_ENHANCE': False,                  # Always enhance
    'TESSERACT_WHITELIST': None,            # No whitelist (all characters)
    'OPTIMIZED_PSM_MODES': ['--psm 6', '--psm 7', '--psm 11', '--psm 13'],
    'EARLY_EXIT_CONFIDENCE': False          # Try all configurations
}

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
1. Choose the appropriate configuration above
2. Copy the values to your verify_ic.py or verify_ic_enhanced.py
3. Adjust ROI coordinates based on your images (use debug images to measure)
4. Test and monitor performance

Example - Apply PRODUCTION_CONFIG:

# In verify_ic.py, replace the settings section with:
DEBUG_MODE = False
ROI = (100, 50, 800, 600)  # ADJUST THESE COORDINATES
TARGET_OCR_HEIGHT = 300
AUTO_ENHANCE = True
TESSERACT_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
OPTIMIZED_PSM_MODES = ['--psm 6', '--psm 11']
EARLY_EXIT_CONFIDENCE = True
"""

# ============================================================================
# ROI CONFIGURATION GUIDE
# ============================================================================
"""
To find ROI coordinates:

1. Run with DEBUG_MODE=True and ROI=None
2. Open debug_enhanced.jpg or debug_thresh.jpg
3. Use an image viewer to measure the IC marking area
4. Note the coordinates: (x, y, width, height)
   - x: Left edge of IC marking
   - y: Top edge of IC marking
   - width: Width of IC marking area
   - height: Height of IC marking area

Example measurements:
- Full HD image (1920x1080): ROI = (400, 200, 1200, 600)
- Cropped image (1024x768): ROI = (200, 150, 600, 400)
- Close-up (640x480): ROI = None (already cropped)

Tips:
- Include some margin around the text (10-20 pixels)
- Ensure all IC markings are within the ROI
- Smaller ROI = faster processing
"""

# ============================================================================
# TARGET_OCR_HEIGHT GUIDE
# ============================================================================
"""
Recommended values based on IC text size:

- 200-300px: Very small IC text (< 2mm), maximum speed
- 300-400px: Small IC text (2-4mm), balanced
- 400-500px: Medium IC text (4-6mm), recommended
- 500-600px: Large IC text (> 6mm), high accuracy
- 600+px: Very large or difficult text, maximum accuracy

Rule of thumb:
- Text should be at least 20-30 pixels tall after downscaling
- If OCR fails, increase by 100px and test again
- If speed is critical, decrease by 50px and verify accuracy
"""

# ============================================================================
# PERFORMANCE TUNING WORKFLOW
# ============================================================================
"""
Step-by-step optimization process:

1. START WITH BALANCED_CONFIG
   - Run on 10-20 test images
   - Check accuracy and timing

2. IF ACCURACY IS GOOD (>95%):
   - Switch to PRODUCTION_CONFIG
   - Set DEBUG_MODE=False
   - Measure ROI and configure
   - Test on 50+ images

3. IF ACCURACY IS LOW (<90%):
   - Switch to HIGH_ACCURACY_CONFIG
   - Increase TARGET_OCR_HEIGHT
   - Disable EARLY_EXIT_CONFIDENCE
   - Check debug images for issues

4. IF SPEED IS TOO SLOW (>50ms):
   - Enable DEBUG_MODE=False
   - Configure ROI to crop tightly
   - Reduce TARGET_OCR_HEIGHT
   - Enable EARLY_EXIT_CONFIDENCE

5. FINE-TUNE:
   - Adjust TARGET_OCR_HEIGHT in steps of 50px
   - Test different ROI sizes
   - Monitor both speed and accuracy

6. PRODUCTION DEPLOYMENT:
   - Set DEBUG_MODE=False
   - Use optimized ROI
   - Enable EARLY_EXIT_CONFIDENCE
   - Monitor and log results
"""

# ============================================================================
# EXPECTED PERFORMANCE BY CONFIGURATION
# ============================================================================
PERFORMANCE_BENCHMARKS = {
    'PRODUCTION_CONFIG': {
        'time_per_ic_ms': '15-25',
        'throughput_per_min': '2,400-4,000',
        'accuracy': '90-95%',
        'use_case': 'High-speed production line'
    },
    'BALANCED_CONFIG': {
        'time_per_ic_ms': '25-35',
        'throughput_per_min': '1,700-2,400',
        'accuracy': '95-98%',
        'use_case': 'Standard production line'
    },
    'TESTING_CONFIG': {
        'time_per_ic_ms': '40-60',
        'throughput_per_min': '1,000-1,500',
        'accuracy': '98-99%',
        'use_case': 'Quality control, testing'
    },
    'HIGH_ACCURACY_CONFIG': {
        'time_per_ic_ms': '60-100',
        'throughput_per_min': '600-1,000',
        'accuracy': '99%+',
        'use_case': 'Difficult ICs, maximum accuracy'
    }
}

"""
Note: Actual performance depends on:
- CPU speed and cores
- Image quality and resolution
- IC text clarity and size
- Tesseract version and configuration
- System load and other processes
"""
