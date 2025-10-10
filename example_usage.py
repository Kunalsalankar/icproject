"""
Example usage of the Counterfeit Detection Agent
This script demonstrates how to use the agent with sample data
"""

from counterfeit_detection_agent import CounterfeitDetectionAgent
import os


def run_detection_example():
    """
    Example: Running counterfeit detection on a product image
    """
    print("="*70)
    print("COUNTERFEIT DETECTION AGENT - EXAMPLE USAGE")
    print("="*70)
    
    # Initialize the agent
    agent = CounterfeitDetectionAgent()
    
    # Optional: Customize thresholds
    agent.logo_match_threshold = 0.7
    agent.ssim_threshold = 0.85
    agent.color_tolerance = 15
    agent.angle_tolerance = 5.0
    
    # Prepare reference data
    # NOTE: Update these paths to your actual reference images
    reference_data = {
        'logo_path': 'reference/genuine_logo.png',
        'expected_text': 'SN123456789ABC',
        'expected_qr_data': 'GENUINE-PRODUCT-CODE-12345',
        'golden_image_path': 'reference/golden_product.png',
        'color_reference': {
            'bgr': [180, 180, 180],  # Average BGR color values
        }
    }
    
    # Path to the product image you want to verify
    test_image_path = 'test_images/product_to_verify.jpg'
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"\n⚠️  Test image not found: {test_image_path}")
        print("Please add a test image to verify.")
        print("\nCreating sample directory structure...")
        os.makedirs('reference', exist_ok=True)
        os.makedirs('test_images', exist_ok=True)
        print("✓ Directories created: reference/ and test_images/")
        print("\nNext steps:")
        print("1. Add reference images to reference/ folder")
        print("2. Add test images to test_images/ folder")
        print("3. Run this script again")
        return
    
    try:
        # Process the image through the detection pipeline
        report = agent.process_image(test_image_path, reference_data)
        
        # Save the report to a JSON file
        output_report_path = 'detection_report.json'
        agent.save_report(report, output_report_path)
        
        # Display summary
        print("\n" + "="*70)
        print("DETECTION SUMMARY")
        print("="*70)
        print(f"📋 Verdict: {report['verdict']}")
        print(f"📊 Total Checks: {report['summary']['total_checks']}")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"⏭️  Skipped: {report['summary']['skipped']}")
        print(f"⚠️  Errors: {report['summary']['errors']}")
        print("="*70)
        
        # Display individual step results
        print("\nDETAILED RESULTS:")
        print("-"*70)
        for result in report['pipeline_results']:
            status_icon = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⏭️"
            print(f"{status_icon} {result['step']}: {result['status']} (Confidence: {result['confidence']:.2f})")
        print("-"*70)
        
        print(f"\n📄 Full report saved to: {output_report_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure all reference images exist.")
    except Exception as e:
        print(f"\n❌ Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()


def run_batch_detection():
    """
    Example: Running detection on multiple images
    """
    print("\n" + "="*70)
    print("BATCH DETECTION MODE")
    print("="*70)
    
    agent = CounterfeitDetectionAgent()
    
    reference_data = {
        'logo_path': 'reference/genuine_logo.png',
        'expected_text': 'SN123456789ABC',
        'expected_qr_data': 'GENUINE-PRODUCT-CODE-12345',
        'golden_image_path': 'reference/golden_product.png',
        'color_reference': {'bgr': [180, 180, 180]}
    }
    
    # List of test images
    test_images = [
        'test_images/product1.jpg',
        'test_images/product2.jpg',
        'test_images/product3.jpg',
    ]
    
    results_summary = []
    
    for idx, image_path in enumerate(test_images, 1):
        if not os.path.exists(image_path):
            print(f"\n⏭️  Skipping {image_path} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing Image {idx}/{len(test_images)}: {image_path}")
        print(f"{'='*70}")
        
        try:
            report = agent.process_image(image_path, reference_data)
            
            # Save individual report
            report_path = f'detection_report_{idx}.json'
            agent.save_report(report, report_path)
            
            results_summary.append({
                'image': image_path,
                'verdict': report['verdict'],
                'confidence': report['pipeline_results'][-1]['confidence']
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            results_summary.append({
                'image': image_path,
                'verdict': 'ERROR',
                'confidence': 0.0
            })
    
    # Print batch summary
    print("\n" + "="*70)
    print("BATCH DETECTION SUMMARY")
    print("="*70)
    for result in results_summary:
        verdict_icon = "✅" if result['verdict'] == "GENUINE" else "❌"
        print(f"{verdict_icon} {result['image']}: {result['verdict']} (Confidence: {result['confidence']:.2f})")
    print("="*70)


def create_sample_reference_data():
    """
    Helper function to create sample reference data structure
    """
    import json
    
    sample_config = {
        "reference_images": {
            "logo_path": "reference/genuine_logo.png",
            "golden_image_path": "reference/golden_product.png"
        },
        "expected_values": {
            "serial_text": "SN123456789ABC",
            "qr_data": "GENUINE-PRODUCT-CODE-12345"
        },
        "color_reference": {
            "bgr": [180, 180, 180],
            "lab": [128, 128, 128]
        },
        "thresholds": {
            "logo_match": 0.7,
            "ssim": 0.85,
            "color_tolerance": 15,
            "angle_tolerance": 5.0
        }
    }
    
    config_path = 'config.json'
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"✓ Sample configuration saved to: {config_path}")


if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('reference', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)
    
    # Run single detection example
    run_detection_example()
    
    # Uncomment to run batch detection
    # run_batch_detection()
    
    # Uncomment to create sample config
    # create_sample_reference_data()
