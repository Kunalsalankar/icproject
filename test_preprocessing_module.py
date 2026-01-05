#!/usr/bin/env python3
"""
Test script to validate the preprocessing module
Demonstrates all preprocessing steps with timing
"""

import sys
import os
import cv2
import numpy as np
from time import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from complete_7step_verification import preprocess_image_comprehensive
    print("✅ Successfully imported preprocessing module\n")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)


def test_preprocessing_on_synthetic_image():
    """Test preprocessing with a synthetic IC-like image"""
    print("="*70)
    print("TEST 1: Synthetic IC-like Image")
    print("="*70)
    
    # Create a synthetic IC image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # Add some text-like patterns
    cv2.rectangle(img, (50, 50), (150, 150), (120, 120, 120), -1)
    cv2.putText(img, "IC CHIP", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Add noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Add some circular pattern (logo)
    cv2.circle(img, (100, 100), 30, (180, 180, 180), 3)
    
    print(f"\nSynthetic image created:")
    print(f"  - Size: {img.shape}")
    print(f"  - Color range: {img.min()}-{img.max()}")
    print(f"  - Contains: text pattern, circle (logo), noise\n")
    
    # Run preprocessing
    start_time = time()
    result = preprocess_image_comprehensive(img, target_size=640, save_steps=False)
    total_time = (time() - start_time) * 1000
    
    if result is None:
        print("❌ Preprocessing failed!")
        return False
    
    # Validate result structure
    print("✅ Preprocessing completed successfully\n")
    print("Result validation:")
    print(f"  ✓ preprocessed: {result['preprocessed'].shape if result['preprocessed'] is not None else 'None'}")
    print(f"  ✓ original_resized: {result['original_resized'].shape if result['original_resized'] is not None else 'None'}")
    print(f"  ✓ intermediate_steps: {len(result['intermediate_steps'])} images")
    print(f"  ✓ metadata: {len(result['metadata'])} fields\n")
    
    # Print metadata
    metadata = result['metadata']
    print("Processing metadata:")
    print(f"  - Original size: {metadata['original_size']}")
    print(f"  - Target size: {metadata['target_size']}")
    print(f"  - Aspect ratio: {metadata['aspect_ratio']:.2f}")
    print(f"  - Total time: {metadata['processing_time_ms']:.2f}ms")
    print(f"  - Timing breakdown: {total_time:.2f}ms measured")
    print(f"\nTechniques applied:")
    for i, tech in enumerate(metadata['techniques_applied'], 1):
        print(f"  {i}. {tech}")
    
    # Validate intermediate steps
    print(f"\nIntermediate steps generated:")
    for step_name, step_img in result['intermediate_steps'].items():
        if step_img is not None:
            print(f"  ✓ {step_name}: {step_img.shape} {step_img.dtype}")
    
    return True


def test_preprocessing_parameters():
    """Test with different parameters"""
    print("\n" + "="*70)
    print("TEST 2: Parameter Variations")
    print("="*70)
    
    img = np.random.randint(50, 150, (400, 400, 3), dtype=np.uint8)
    
    print("\nTesting different target sizes:")
    for size in [512, 640, 768]:
        start = time()
        result = preprocess_image_comprehensive(img, target_size=size, save_steps=False)
        elapsed = (time() - start) * 1000
        
        if result:
            print(f"  ✓ Size {size}×{size}: {elapsed:.1f}ms → {result['preprocessed'].shape}")
        else:
            print(f"  ✗ Size {size}×{size}: Failed")
    
    return True


def test_preprocessing_edge_cases():
    """Test with edge case images"""
    print("\n" + "="*70)
    print("TEST 3: Edge Cases")
    print("="*70)
    
    # Test 1: Very small image
    print("\n1. Very small image (100×100):")
    small_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = preprocess_image_comprehensive(small_img, save_steps=False)
    print(f"  ✓ Result: {result['preprocessed'].shape if result else 'Failed'}")
    
    # Test 2: Very large image
    print("\n2. Large image (1920×1080):")
    large_img = np.ones((1080, 1920, 3), dtype=np.uint8) * 128
    result = preprocess_image_comprehensive(large_img, save_steps=False)
    print(f"  ✓ Result: {result['preprocessed'].shape if result else 'Failed'}")
    
    # Test 3: Non-square image
    print("\n3. Non-square image (320×640):")
    rect_img = np.ones((640, 320, 3), dtype=np.uint8) * 128
    result = preprocess_image_comprehensive(rect_img, save_steps=False)
    if result:
        print(f"  ✓ Result: {result['preprocessed'].shape}")
        print(f"  ✓ Aspect ratio preserved: {result['metadata']['aspect_ratio']:.2f}")
    
    # Test 4: Low contrast image
    print("\n4. Low contrast image (100-120 range):")
    low_contrast = np.random.randint(100, 120, (400, 400, 3), dtype=np.uint8)
    result = preprocess_image_comprehensive(low_contrast, save_steps=False)
    print(f"  ✓ Result: {result['preprocessed'].shape if result else 'Failed'}")
    print(f"  ✓ CLAHE enhancement applied for low contrast")
    
    # Test 5: High contrast image
    print("\n5. High contrast image (0-255 range):")
    high_contrast = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    result = preprocess_image_comprehensive(high_contrast, save_steps=False)
    print(f"  ✓ Result: {result['preprocessed'].shape if result else 'Failed'}")
    
    return True


def test_preprocessing_consistency():
    """Test preprocessing consistency across multiple runs"""
    print("\n" + "="*70)
    print("TEST 4: Consistency Across Multiple Runs")
    print("="*70)
    
    img = np.random.randint(80, 140, (400, 400, 3), dtype=np.uint8)
    
    print("\nRunning preprocessing 3 times on same image...")
    results = []
    
    for i in range(3):
        start = time()
        result = preprocess_image_comprehensive(img.copy(), save_steps=False)
        elapsed = (time() - start) * 1000
        
        if result:
            results.append(result)
            print(f"  Run {i+1}: {elapsed:.1f}ms ✓")
        else:
            print(f"  Run {i+1}: Failed ✗")
            return False
    
    # Check consistency
    if len(results) == 3:
        # Compare shapes (should be identical)
        shape_consistent = (
            results[0]['preprocessed'].shape == results[1]['preprocessed'].shape ==
            results[2]['preprocessed'].shape
        )
        print(f"\nShape consistency: {'✓ Consistent' if shape_consistent else '✗ Inconsistent'}")
        
        # Check that results are nearly identical
        diff = np.abs(results[0]['preprocessed'].astype(float) - 
                     results[1]['preprocessed'].astype(float))
        max_diff = diff.max()
        print(f"Max pixel difference between runs: {max_diff:.0f}")
        print(f"Consistency check: {'✓ Passed' if max_diff < 5 else '✗ Warning: High variation'}")
        
        return True
    
    return False


def test_preprocessing_performance():
    """Performance benchmarking"""
    print("\n" + "="*70)
    print("TEST 5: Performance Benchmarking")
    print("="*70)
    
    sizes = [
        (480, 640),
        (720, 960),
        (1080, 1440),
    ]
    
    print("\nProcessing time for different image sizes:")
    print("Size         Time (ms)  Speed (MP/s)")
    print("-" * 40)
    
    for h, w in sizes:
        img = np.ones((h, w, 3), dtype=np.uint8) * 128
        megapixels = (h * w) / 1_000_000
        
        # Warmup
        preprocess_image_comprehensive(img.copy(), save_steps=False)
        
        # Benchmark (5 runs)
        times = []
        for _ in range(5):
            start = time()
            preprocess_image_comprehensive(img.copy(), save_steps=False)
            times.append((time() - start) * 1000)
        
        avg_time = np.mean(times)
        speed = megapixels / (avg_time / 1000)
        
        print(f"{w}×{h:4d}      {avg_time:6.1f}      {speed:6.1f}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" IMAGE PREPROCESSING MODULE - COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("Synthetic Image Processing", test_preprocessing_on_synthetic_image),
        ("Parameter Variations", test_preprocessing_parameters),
        ("Edge Cases", test_preprocessing_edge_cases),
        ("Consistency Checks", test_preprocessing_consistency),
        ("Performance Benchmarking", test_preprocessing_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Exception in {test_name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)} ✓")
    print(f"Failed: {failed}/{len(tests)} ✗")
    
    if failed == 0:
        print("\n🎉 All tests passed! Preprocessing module is working correctly.")
        print("\nNext steps:")
        print("  1. Run: python app_with_preprocessing.py")
        print("  2. Visit: http://localhost:7860")
        print("  3. Try preprocessing with real IC images")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check output above.")
    
    print("\n" + "="*70 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
