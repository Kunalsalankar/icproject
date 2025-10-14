"""
Test script for same IC with adjusted thresholds
This shows how to configure the agent for real-world scenarios
"""

from counterfeit_detection_agent import CounterfeitDetectionAgent

# Initialize agent with LENIENT thresholds for real-world use
agent = CounterfeitDetectionAgent()

# Adjust thresholds for same product, different photos
agent.logo_match_threshold = 0.3      # Lower from 0.7 (more lenient)
agent.ssim_threshold = 0.5            # Lower from 0.85 (more lenient)
agent.color_tolerance = 50            # Higher from 15 (more lenient)
agent.angle_tolerance = 15.0          # Higher from 5.0 (more lenient)

print("="*70)
print("TESTING SAME IC WITH LENIENT THRESHOLDS")
print("="*70)

# Use one image as reference, another as test
reference_data = {
    'logo_path': None,  # Skip logo check
    'expected_text': None,  # Skip exact text match
    'expected_qr_data': None,  # Skip QR check
    'golden_image_path': 'test_images/ic1.jpg',  # Use first image as reference
    'color_reference': None  # Skip color check
}

# Test with second image
report = agent.process_image('test_images/ic2.jpg', reference_data)

print("\n" + "="*70)
print(f"RESULT: {report['verdict']}")
print(f"Confidence: {report['pipeline_results'][-1]['confidence']:.2f}")
print("="*70)

# Save report
agent.save_report(report, 'same_ic_test_report.json')
print("\nReport saved to: same_ic_test_report.json")
