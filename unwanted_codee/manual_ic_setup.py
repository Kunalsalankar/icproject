"""
Manual IC Setup Tool
Use this when OCR fails to read IC markings
"""

import json
import os

print("="*70)
print("MANUAL IC REFERENCE SETUP")
print("="*70)
print("\nThis tool helps you manually configure IC reference data")
print("when OCR cannot read the IC markings from images.\n")

# Step 1: Get reference IC information
print("Step 1: Look at your REFERENCE IC (golden_product.jpg)")
print("-"*70)
print("Please enter the markings you see on the REFERENCE IC:\n")

ref_part_number = input("Part Number (e.g., SN74LS266N): ").strip().upper()
ref_manufacturer = input("Manufacturer Mark (e.g., HLF) [press Enter to skip]: ").strip().upper()
ref_date_code = input("Date/Batch Code (e.g., 20A1) [press Enter to skip]: ").strip().upper()

# Step 2: Confirm
print("\n" + "="*70)
print("REFERENCE IC MARKINGS ENTERED:")
print("="*70)
print(f"Part Number:      {ref_part_number or 'NOT PROVIDED'}")
print(f"Manufacturer:     {ref_manufacturer or 'NOT PROVIDED'}")
print(f"Date/Batch Code:  {ref_date_code or 'NOT PROVIDED'}")
print("="*70)

confirm = input("\nIs this correct? (yes/no): ").strip().lower()

if confirm not in ['yes', 'y']:
    print("\n❌ Setup cancelled. Please run again.")
    exit()

# Step 3: Save to config
config_data = {
    "part_number": ref_part_number if ref_part_number else None,
    "manufacturer": ref_manufacturer if ref_manufacturer else None,
    "date_code": None,  # Always set to None to accept any date code
    "use_manual_config": True,
    "description": "Manual IC Configuration - Entered by user",
    "notes": [
        "This configuration was created manually because OCR failed",
        "part_number: CRITICAL - Must match exactly",
        "manufacturer: Optional but recommended",
        "date_code: Set to null to accept any date code (recommended)"
    ]
}

# Save to reference folder
config_path = 'reference/ic_config.json'
os.makedirs('reference', exist_ok=True)

with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)

print(f"\n✅ Configuration saved to: {config_path}")

# Step 4: Create a simple verification script
print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Your reference IC configuration is saved")
print("2. Now run: python verify_ic_simple.py")
print("3. The script will compare test ICs with your reference")
print("="*70)

# Create a simple verification script
simple_script = '''"""
Simple IC Verification - Uses Manual Configuration
"""

import cv2
import pytesseract
import json
import os
import re

# Load manual configuration
with open('reference/ic_config.json', 'r') as f:
    config = json.load(f)

EXPECTED_PART_NUMBER = config['part_number']
EXPECTED_MANUFACTURER = config.get('manufacturer')

print("="*70)
print("SIMPLE IC VERIFICATION")
print("="*70)
print(f"\\nExpected Part Number: {EXPECTED_PART_NUMBER}")
print(f"Expected Manufacturer: {EXPECTED_MANUFACTURER or 'Any'}")
print("="*70)

# Get test image
test_image_path = input("\\nEnter path to test IC image (or press Enter for default): ").strip()
if not test_image_path:
    test_image_path = 'test_images/product_to_verify.jpg'

if not os.path.exists(test_image_path):
    print(f"\\n❌ Image not found: {test_image_path}")
    exit()

print(f"\\nTesting image: {test_image_path}")
print("\\nOption 1: Let OCR try to read it")
print("Option 2: Manually enter what you see on the test IC")
choice = input("\\nChoose option (1 or 2): ").strip()

if choice == '1':
    # Try OCR
    print("\\n🔄 Running OCR...")
    image = cv2.imread(test_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    
    print(f"\\nOCR Result: {text}")
    
    # Try to extract part number
    part_pattern = r'SN\\d{2}[A-Z]{2,4}\\d{2,4}[A-Z]?'
    match = re.search(part_pattern, text.upper())
    
    if match:
        detected_part = match.group()
        print(f"\\n✓ Detected Part Number: {detected_part}")
    else:
        print("\\n❌ Could not detect part number from OCR")
        detected_part = input("\\nManually enter part number from test IC: ").strip().upper()
else:
    # Manual entry
    print("\\nLook at the test IC image and enter what you see:")
    detected_part = input("Part Number: ").strip().upper()

# Verify
print("\\n" + "="*70)
print("VERIFICATION RESULT")
print("="*70)
print(f"Expected:  {EXPECTED_PART_NUMBER}")
print(f"Detected:  {detected_part}")

if detected_part == EXPECTED_PART_NUMBER:
    print("\\n✅ VERDICT: GENUINE")
    print("   Part number matches the reference IC!")
else:
    print("\\n❌ VERDICT: COUNTERFEIT/UNVERIFIED")
    print("   Part number does NOT match!")

print("="*70)
'''

with open('verify_ic_simple.py', 'w') as f:
    f.write(simple_script)

print("\n✅ Created: verify_ic_simple.py")
print("\nYou can now run: python verify_ic_simple.py")
