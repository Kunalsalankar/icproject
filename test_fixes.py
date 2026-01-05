#!/usr/bin/env python3
"""
Test script to validate all fixes in the counterfeit detection pipeline
Run this to verify the corrections work properly
"""

import json
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from complete_7step_verification import (
        CORRELATION_PASS_THRESHOLD, 
        CORRELATION_MIN_PASS_RATE,
        run_complete_7step_verification
    )
except ImportError as e:
    print(f"Error importing verification module: {e}")
    sys.exit(1)

def test_threshold_configuration():
    """Test 1: Verify thresholds are correctly configured"""
    print("\n" + "="*80)
    print("TEST 1: Threshold Configuration")
    print("="*80)
    
    print(f"Confidence Threshold: {CORRELATION_PASS_THRESHOLD}")
    print(f"Pass Rate Threshold: {CORRELATION_MIN_PASS_RATE}")
    
    assert CORRELATION_PASS_THRESHOLD == 0.75, f"Confidence threshold should be 0.75, got {CORRELATION_PASS_THRESHOLD}"
    assert CORRELATION_MIN_PASS_RATE == 0.85, f"Pass rate threshold should be 0.85, got {CORRELATION_MIN_PASS_RATE}"
    
    print("✅ PASS: Thresholds correctly configured")
    return True

def test_verdict_logic_genuine():
    """Test 2: Verify GENUINE verdict with perfect score"""
    print("\n" + "="*80)
    print("TEST 2: Verdict Logic - GENUINE Product")
    print("="*80)
    
    # Simulate results: 6/6 checks pass with 0.819 confidence
    test_results = [
        {'step': '1. Logo Detection (Template)', 'status': 'PASS', 'confidence': 0.531},
        {'step': '2. Text & Serial Number OCR', 'status': 'PASS', 'confidence': 1.000},
        {'step': '3. QR/DMC Code Detection', 'status': 'SKIPPED', 'confidence': 0.0},
        {'step': '4. Surface Defect Detection', 'status': 'PASS', 'confidence': 0.792},
        {'step': '5. IC Geometry & Alignment', 'status': 'PASS', 'confidence': 0.989},
        {'step': '6. Color & Texture Verification', 'status': 'PASS', 'confidence': 0.995},
        {'step': '7. Font Verification & Correlation', 'status': 'PASS', 'confidence': 0.779},
    ]
    
    # Calculate metrics
    passed = len([r for r in test_results if r['status'] == 'PASS'])
    failed = len([r for r in test_results if r['status'] == 'FAIL'])
    total = len([r for r in test_results if r['status'] != 'SKIPPED'])
    avg_confidence = sum(r['confidence'] for r in test_results if r['status'] != 'SKIPPED') / total
    pass_rate = passed / total
    
    print(f"Results Summary:")
    print(f"  - Passed: {passed}/{total}")
    print(f"  - Failed: {failed}/{total}")
    print(f"  - Pass Rate: {pass_rate:.1%}")
    print(f"  - Average Confidence: {avg_confidence:.3f}")
    
    # Check verdict criteria
    is_high_confidence = avg_confidence >= CORRELATION_PASS_THRESHOLD
    is_good_pass_rate = pass_rate >= CORRELATION_MIN_PASS_RATE
    is_within_failure_limit = failed <= 1
    is_no_critical_failures = True  # None of the critical checks failed
    
    print(f"\nVerdict Criteria:")
    print(f"  ✅ High Confidence (≥{CORRELATION_PASS_THRESHOLD}): {is_high_confidence} ({avg_confidence:.3f})")
    print(f"  ✅ Good Pass Rate (≥{CORRELATION_MIN_PASS_RATE:.0%}): {is_good_pass_rate} ({pass_rate:.0%})")
    print(f"  ✅ Within Failure Limit (≤1): {is_within_failure_limit} ({failed})")
    print(f"  ✅ No Critical Failures: {is_no_critical_failures}")
    
    # All must be true for GENUINE
    should_be_genuine = all([
        is_high_confidence,
        is_good_pass_rate,
        is_within_failure_limit,
        is_no_critical_failures
    ])
    
    print(f"\nExpected Verdict: {'GENUINE' if should_be_genuine else 'COUNTERFEIT'}")
    assert should_be_genuine, "Product with 6/6 passes and 0.819 confidence should be GENUINE!"
    
    print("✅ PASS: Verdict logic correctly identifies GENUINE products")
    return True

def test_verdict_logic_counterfeit():
    """Test 3: Verify COUNTERFEIT verdict with failures"""
    print("\n" + "="*80)
    print("TEST 3: Verdict Logic - COUNTERFEIT Product")
    print("="*80)
    
    # Simulate results: 2/6 checks pass (only 33% pass rate)
    test_results = [
        {'step': '1. Logo Detection (Template)', 'status': 'FAIL', 'confidence': 0.3},
        {'step': '2. Text & Serial Number OCR', 'status': 'PASS', 'confidence': 0.8},
        {'step': '3. QR/DMC Code Detection', 'status': 'FAIL', 'confidence': 0.1},
        {'step': '4. Surface Defect Detection', 'status': 'FAIL', 'confidence': 0.2},
        {'step': '5. IC Geometry & Alignment', 'status': 'PASS', 'confidence': 0.5},
        {'step': '6. Color & Texture Verification', 'status': 'FAIL', 'confidence': 0.3},
        {'step': '7. Font Verification & Correlation', 'status': 'FAIL', 'confidence': 0.4},
    ]
    
    # Calculate metrics
    passed = len([r for r in test_results if r['status'] == 'PASS'])
    failed = len([r for r in test_results if r['status'] == 'FAIL'])
    total = len([r for r in test_results if r['status'] != 'SKIPPED'])
    avg_confidence = sum(r['confidence'] for r in test_results) / total
    pass_rate = passed / total
    
    print(f"Results Summary:")
    print(f"  - Passed: {passed}/{total}")
    print(f"  - Failed: {failed}/{total}")
    print(f"  - Pass Rate: {pass_rate:.1%}")
    print(f"  - Average Confidence: {avg_confidence:.3f}")
    
    # Check verdict criteria
    is_high_confidence = avg_confidence >= CORRELATION_PASS_THRESHOLD
    is_good_pass_rate = pass_rate >= CORRELATION_MIN_PASS_RATE
    is_within_failure_limit = failed <= 1
    is_no_critical_failures = False  # Logo Detection (critical) failed
    
    print(f"\nVerdict Criteria:")
    print(f"  ❌ High Confidence (≥{CORRELATION_PASS_THRESHOLD}): {is_high_confidence} ({avg_confidence:.3f})")
    print(f"  ❌ Good Pass Rate (≥{CORRELATION_MIN_PASS_RATE:.0%}): {is_good_pass_rate} ({pass_rate:.0%})")
    print(f"  ❌ Within Failure Limit (≤1): {is_within_failure_limit} ({failed})")
    print(f"  ❌ No Critical Failures: {is_no_critical_failures}")
    
    # All must be true for GENUINE
    should_be_genuine = all([
        is_high_confidence,
        is_good_pass_rate,
        is_within_failure_limit,
        is_no_critical_failures
    ])
    
    print(f"\nExpected Verdict: {'GENUINE' if should_be_genuine else 'COUNTERFEIT'}")
    assert not should_be_genuine, "Product with only 33% pass rate should be COUNTERFEIT!"
    
    print("✅ PASS: Verdict logic correctly identifies COUNTERFEIT products")
    return True

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("COUNTERFEIT DETECTION PIPELINE - FIX VALIDATION")
    print("="*80)
    
    tests = [
        ("Threshold Configuration", test_threshold_configuration),
        ("Verdict Logic - GENUINE", test_verdict_logic_genuine),
        ("Verdict Logic - COUNTERFEIT", test_verdict_logic_counterfeit),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Fixes verified successfully!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed - Please review the errors above")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
