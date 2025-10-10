"""
Test script for Counterfeit Detection Agent
Tests the agent with synthetic/demo data when real images are not available
"""

import cv2
import numpy as np
from counterfeit_detection_agent import CounterfeitDetectionAgent
import os


def create_synthetic_logo(size=(200, 200)):
    """Create a synthetic logo for testing"""
    logo = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # Draw a circle
    cv2.circle(logo, (100, 100), 60, (0, 0, 255), -1)
    
    # Draw text
    cv2.putText(logo, "LOGO", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    return logo


def create_synthetic_product_image(genuine=True):
    """Create a synthetic product image for testing"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    # Add logo
    logo = create_synthetic_logo((150, 150))
    if not genuine:
        # Slightly modify logo for counterfeit
        logo = cv2.GaussianBlur(logo, (5, 5), 0)
    img[50:200, 50:200] = cv2.resize(logo, (150, 150))
    
    # Add serial text
    serial = "SN123456789ABC" if genuine else "SN987654321XYZ"
    cv2.putText(img, serial, (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add QR code placeholder (black square)
    qr_size = 120
    qr_color = (0, 0, 0) if genuine else (50, 50, 50)
    cv2.rectangle(img, (600, 50), (600 + qr_size, 50 + qr_size), qr_color, -1)
    cv2.putText(img, "QR", (630, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add some geometric shapes (for edge detection)
    cv2.rectangle(img, (50, 300), (750, 550), (100, 100, 100), 3)
    cv2.line(img, (50, 300), (750, 300), (100, 100, 100), 2)
    
    # Add some defects for counterfeit
    if not genuine:
        # Add random noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Add some scratches
        cv2.line(img, (300, 200), (500, 400), (150, 150, 150), 2)
    
    return img


def setup_test_environment():
    """Create test directories and synthetic images"""
    print("Setting up test environment...")
    
    # Create directories
    os.makedirs('reference', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)
    
    # Create synthetic logo
    logo = create_synthetic_logo()
    cv2.imwrite('reference/genuine_logo.png', logo)
    print("✓ Created reference/genuine_logo.png")
    
    # Create golden product image
    golden = create_synthetic_product_image(genuine=True)
    cv2.imwrite('reference/golden_product.png', golden)
    print("✓ Created reference/golden_product.png")
    
    # Create test images
    genuine_product = create_synthetic_product_image(genuine=True)
    cv2.imwrite('test_images/genuine_product.jpg', genuine_product)
    print("✓ Created test_images/genuine_product.jpg")
    
    counterfeit_product = create_synthetic_product_image(genuine=False)
    cv2.imwrite('test_images/counterfeit_product.jpg', counterfeit_product)
    print("✓ Created test_images/counterfeit_product.jpg")
    
    print("\n✅ Test environment setup complete!\n")


def test_genuine_product():
    """Test detection on a genuine product"""
    print("="*70)
    print("TEST 1: GENUINE PRODUCT DETECTION")
    print("="*70)
    
    agent = CounterfeitDetectionAgent()
    
    reference_data = {
        'logo_path': 'reference/genuine_logo.png',
        'expected_text': 'SN123456789ABC',
        'expected_qr_data': None,  # Skip QR verification for synthetic test
        'golden_image_path': 'reference/golden_product.png',
        'color_reference': {
            'bgr': [200, 200, 200],
        }
    }
    
    try:
        report = agent.process_image('test_images/genuine_product.jpg', reference_data)
        agent.save_report(report, 'test_genuine_report.json')
        
        print("\n" + "="*70)
        print(f"RESULT: {report['verdict']}")
        print(f"Expected: GENUINE")
        print(f"Match: {'✅ PASS' if report['verdict'] == 'GENUINE' else '❌ FAIL'}")
        print("="*70)
        
        return report['verdict'] == 'GENUINE'
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_counterfeit_product():
    """Test detection on a counterfeit product"""
    print("\n" + "="*70)
    print("TEST 2: COUNTERFEIT PRODUCT DETECTION")
    print("="*70)
    
    agent = CounterfeitDetectionAgent()
    
    reference_data = {
        'logo_path': 'reference/genuine_logo.png',
        'expected_text': 'SN123456789ABC',
        'expected_qr_data': None,
        'golden_image_path': 'reference/golden_product.png',
        'color_reference': {
            'bgr': [200, 200, 200],
        }
    }
    
    try:
        report = agent.process_image('test_images/counterfeit_product.jpg', reference_data)
        agent.save_report(report, 'test_counterfeit_report.json')
        
        print("\n" + "="*70)
        print(f"RESULT: {report['verdict']}")
        print(f"Expected: COUNTERFEIT")
        print(f"Match: {'✅ PASS' if report['verdict'] == 'COUNTERFEIT' else '❌ FAIL'}")
        print("="*70)
        
        return report['verdict'] == 'COUNTERFEIT'
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_modules():
    """Test individual detection modules"""
    print("\n" + "="*70)
    print("TEST 3: INDIVIDUAL MODULE TESTING")
    print("="*70)
    
    agent = CounterfeitDetectionAgent()
    test_image = cv2.imread('test_images/genuine_product.jpg')
    
    if test_image is None:
        print("❌ Could not load test image")
        return False
    
    print("\n1. Testing Logo Detection...")
    logo_result = agent.detect_logo(test_image, 'reference/genuine_logo.png')
    print(f"   Status: {logo_result.status}, Confidence: {logo_result.confidence:.3f}")
    
    print("\n2. Testing Text Detection + OCR...")
    text_result = agent.detect_and_read_text(test_image, 'SN123456789ABC')
    print(f"   Status: {text_result.status}, Confidence: {text_result.confidence:.3f}")
    
    print("\n3. Testing QR/DMC Detection...")
    qr_result = agent.detect_qr_code(test_image, None)
    print(f"   Status: {qr_result.status}, Confidence: {qr_result.confidence:.3f}")
    
    print("\n4. Testing Defect Detection...")
    defect_result = agent.detect_defects(test_image, 'reference/golden_product.png')
    print(f"   Status: {defect_result.status}, Confidence: {defect_result.confidence:.3f}")
    
    print("\n5. Testing Alignment Check...")
    angle_result = agent.check_alignment(test_image)
    print(f"   Status: {angle_result.status}, Confidence: {angle_result.confidence:.3f}")
    
    print("\n6. Testing Color Verification...")
    color_result = agent.verify_color(test_image, {'bgr': [200, 200, 200]})
    print(f"   Status: {color_result.status}, Confidence: {color_result.confidence:.3f}")
    
    print("\n✅ Individual module testing complete")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("COUNTERFEIT DETECTION AGENT - TEST SUITE")
    print("="*70 + "\n")
    
    # Setup test environment
    setup_test_environment()
    
    # Run tests
    results = []
    
    results.append(("Genuine Product Detection", test_genuine_product()))
    results.append(("Counterfeit Product Detection", test_counterfeit_product()))
    results.append(("Individual Module Testing", test_individual_modules()))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*70)
    print(f"Tests Passed: {passed}/{total}")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()
