# Reference Folder

This folder contains reference images and configuration for IC verification.

## Files Needed:

### 1. `golden_product.jpg` (Required)
- **Purpose:** Golden/reference image of a genuine IC chip
- **What to include:** Clear photo of a genuine IC showing all markings
- **Quality requirements:**
  - High resolution (at least 1920×1080)
  - Good lighting (no shadows or glare)
  - Sharp focus on IC text
  - Straight-on angle (perpendicular to IC)

### 2. `ic_config.json` (Optional)
- **Purpose:** Manual configuration if reference image is not available
- **Format:**
```json
{
  "part_number": "SN74LS266N",
  "manufacturer": "HLF",
  "date_code": null
}
```

## How It Works:

### Option 1: Automatic (Recommended)
1. Add `golden_product.jpg` to this folder
2. Run `python verify_ic_enhanced.py`
3. Script automatically extracts markings from golden IC
4. Compares test IC with golden IC

### Option 2: Manual Configuration
1. Edit `ic_config.json` with expected values
2. Run `python verify_ic_enhanced.py`
3. Script uses manual configuration for verification

## Setup Instructions:

### Step 1: Add Golden IC Image
```
reference/
└── golden_product.jpg  ← Add your genuine IC image here
```

### Step 2: (Optional) Configure Expected Values
Edit `ic_config.json`:
```json
{
  "part_number": "YOUR_IC_PART_NUMBER",
  "manufacturer": "YOUR_MANUFACTURER",
  "date_code": null
}
```

**Note:** Set `date_code` to `null` to accept any date code (recommended, as date codes vary between batches).

### Step 3: Run Verification
```bash
python verify_ic_enhanced.py
```

## Example Golden IC Image:

Your `golden_product.jpg` should look like:
- Clear, high-resolution photo
- IC markings clearly visible
- Good lighting, no reflections
- Text in focus

## Troubleshooting:

### Reference IC not detected
1. Check image quality (resolution, focus, lighting)
2. Ensure IC text is clearly visible
3. Try different angles/lighting
4. Use manual configuration as fallback

### OCR fails to read reference IC
1. Take a better photo (closer, better lighting)
2. Use `debug_enhanced.jpg` to see what OCR sees
3. Manually configure `ic_config.json`

## File Structure:

```
reference/
├── README.md              # This file
├── golden_product.jpg     # Your genuine IC image (add this) ⭐
├── ic_config.json         # Optional manual configuration
└── genuine_logo.png       # (Optional) For logo detection
```

## Notes:

- **Part number** is CRITICAL - must match exactly
- **Manufacturer** is optional but recommended
- **Date code** should be set to `null` (varies between batches)
- Reference IC image is automatically processed with the same enhancements as test images
