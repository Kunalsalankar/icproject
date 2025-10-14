"""
Quick test to verify COON → C00N correction works
"""

# Test the correction logic
test_strings = [
    "SN74HCOON",
    "SN74HC00N",
    "W24RS SN74HCOONJ S",
    "SN74HCOON\? Ji",
    "COON",
    "HC00N",
]

print("Testing COON → C00N Correction:")
print("="*50)

for test in test_strings:
    corrected = test.replace('COON', 'C00N').replace('COO', 'C00')
    if corrected != test:
        print(f"✓ '{test}' → '{corrected}'")
    else:
        print(f"  '{test}' (no change)")

print("\n" + "="*50)
print("Expected IC Part Number: SN74HC00N")
print("="*50)
