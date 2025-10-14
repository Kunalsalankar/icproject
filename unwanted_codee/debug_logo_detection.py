"""
Debug Logo Detection - Test different approaches to find the Motorola logo
"""

import cv2
import numpy as np
import os

def debug_logo_detection(image_path):
    """Debug version to understand why logo detection is failing"""
    print(f"Debugging logo detection for: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Could not load image")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Save original grayscale
    cv2.imwrite('debug_original_gray.jpg', gray)
    
    # Try different threshold values and save results
    threshold_values = [150, 180, 200, 220, 240]
    
    for i, thresh_val in enumerate(threshold_values):
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'debug_thresh_{thresh_val}.jpg', thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Threshold {thresh_val}: Found {len(contours)} contours")
        
        # Analyze contours
        for j, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 20:  # Very low threshold for debugging
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                print(f"  Contour {j}: area={area:.0f}, circularity={circularity:.3f}, bbox=({x},{y},{w},{h}), aspect={aspect_ratio:.3f}")
                
                # Save individual contours
                if area > 50:
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    cv2.imwrite(f'debug_contour_{thresh_val}_{j}.jpg', mask)
    
    # Try adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite('debug_adaptive_thresh.jpg', adaptive_thresh)
    
    # Try Otsu threshold
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('debug_otsu_thresh.jpg', otsu_thresh)
    
    # Try edge detection
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite('debug_edges.jpg', edges)
    
    # Try morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('debug_morph.jpg', morph)
    
    print("Debug images saved. Check debug_*.jpg files to see what's happening.")

if __name__ == "__main__":
    # Test on both images
    debug_logo_detection('reference/golden_product.jpg')
    print("\n" + "="*50 + "\n")
    debug_logo_detection('test_images/product_to_verify.jpg')


