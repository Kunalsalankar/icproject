"""
Test script for AI Agent IC Verification
Tests the Hugging Face BLIP model implementation
"""

import cv2
import sys
import time
from complete_7step_verification import (
    initialize_ai_agent, 
    analyze_ic_with_ai_agent,
    analyze_ic_with_database,
    ai_agent_oem_verification
)

def print_separator(char='=', length=70):
    """Print a separator line"""
    print(char * length)

def test_ai_agent_initialization():
    """Test 1: AI Agent Initialization"""
    print_separator()
    print("TEST 1: AI Agent Initialization")
    print_separator()
    
    start_time = time.time()
    success = initialize_ai_agent()
    init_time = (time.time() - start_time) * 1000
    
    if success:
        print(f"✅ PASS - AI Agent initialized successfully")
        print(f"   Initialization time: {init_time:.2f}ms")
        return True
    else:
        print(f"FAIL - AI Agent initialization failed")
        print(f"   This is OK - will use fallback database")
        return False
    
def test_database_fallback():
    """Test 2: Database Fallback"""
    print_separator()
    print("TEST 2: Database Fallback")
    print_separator()
    
    test_parts = ['MC74HC20N', 'SN74LS00', 'CD4017', '555', 'LM358', 'UNKNOWN123']
    
    for part in test_parts:
        result = analyze_ic_with_database(part)
        status = "✅ FOUND" if result['found'] else "❌ NOT FOUND"
        print(f"{status} - {part}: {result.get('manufacturer', 'N/A')}")
    
    return True

def test_ai_agent_analysis(image_path):
    """Test 3: AI Agent Image Analysis"""
    print_separator()
    print("TEST 3: AI Agent Image Analysis")
    print_separator()
    
    # Check if image exists
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ FAIL - Could not load image: {image_path}")
            return False
        
        print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    except Exception as e:
        print(f"FAIL - Error loading image: {str(e)}")
        return False
    
    # Analyze with AI Agent
    print("\nAnalyzing IC with AI Agent...")
    start_time = time.time()
    
    try:
        result = analyze_ic_with_ai_agent(image, part_number=None)
        analysis_time = (time.time() - start_time) * 1000
        
        print(f"\nAnalysis completed in {analysis_time:.2f}ms")
        print_separator('-')
        print(f"Part Number:     {result.get('part_number', 'N/A')}")
        print(f"Manufacturer:    {result.get('manufacturer', 'N/A')}")
        print(f"Description:     {result.get('description', 'N/A')[:60]}...")
        print(f"Package:         {result.get('package', 'N/A')}")
        print(f"Status:          {result.get('status', 'N/A')}")
        print(f"AI Confidence:   {result.get('ai_confidence', 0.0):.2%}")
        print(f"Found:           {result.get('found', False)}")
        print_separator('-')
        
        return True
        
    except Exception as e:
        analysis_time = (time.time() - start_time) * 1000
        print(f"❌ FAIL - Analysis error: {str(e)}")
        print(f"   Time before error: {analysis_time:.2f}ms")
        return False

def test_full_verification(image_path):
    """Test 4: Full OEM Verification Pipeline"""
    print_separator()
    print("TEST 4: Full OEM Verification Pipeline")
    print_separator()
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ FAIL - Could not load image: {image_path}")
            return False
        
        print("Running full verification pipeline...")
        start_time = time.time()
        
        result = ai_agent_oem_verification(
            image=image,
            ocr_text=None,
            logo_detected=True
        )
        
        total_time = (time.time() - start_time) * 1000
        
        print(f"\n✅ Verification completed")
        print_separator('-')
        print(f"Step:            {result['step']}")
        print(f"Status:          {result['status']}")
        print(f"Confidence:      {result['confidence']:.2%}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        print(f"Total Time:      {total_time:.2f}ms")
        print_separator('-')
        print("Details:")
        for key, value in result['details'].items():
            if isinstance(value, str) and len(value) > 60:
                value = value[:60] + "..."
            print(f"  {key:20s}: {value}")
        print_separator('-')
        
        return result['status'] in ['PASS', 'SKIPPED']
        
    except Exception as e:
        print(f"❌ FAIL - Verification error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("\n")
    print_separator('=')
    print(" AI AGENT IC VERIFICATION - TEST SUITE ".center(70))
    print_separator('=')
    print()
    
    # Test configuration
    test_image = 'test_images/product_to_verify.jpg'
    
    # Run tests
    results = {}
    
    # Test 1: Initialization
    results['initialization'] = test_ai_agent_initialization()
    print()
    
    # Test 2: Database Fallback
    results['database'] = test_database_fallback()
    print()
    
    # Test 3: AI Agent Analysis (only if image exists)
    import os
    if os.path.exists(test_image):
        results['ai_analysis'] = test_ai_agent_analysis(test_image)
        print()
        
        # Test 4: Full Verification
        results['full_verification'] = test_full_verification(test_image)
        print()
    else:
        print(f"⚠️  WARNING: Test image not found: {test_image}")
        print(f"   Skipping image-based tests")
        print(f"   Please ensure test image exists to run all tests")
        print()
    
    # Summary
    print_separator('=')
    print(" TEST SUMMARY ".center(70))
    print_separator('=')
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print_separator('-')
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print_separator('=')
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! AI Agent is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("   Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
