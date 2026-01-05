# 🔬 IC Counterfeit Detection System - Quick Start Guide

A comprehensive guide to run the IC Chip Counterfeit Detection application on your computer.

---

## **Step 1: Download & Prepare**

1. **Get the project folder** from your friend
   - The folder should be named something like `AI_PROect` or similar
   - Keep it in an easy location (Desktop or Documents)

2. **Make sure you have Python installed**
   - Download Python 3.9 or higher from: https://www.python.org/downloads/
   - During installation, **CHECK the box** that says "Add Python to PATH"

3. **Open Command Prompt or PowerShell**
   - Press `Windows Key + R`
   - Type `cmd` and press Enter
   - OR use PowerShell (recommended)

---

## **Step 2: Navigate to the Project Folder**

In the Command Prompt/PowerShell window, type:

```
cd "C:\path\to\your\AI_PROect"
```

Replace `C:\path\to\your\AI_PROect` with the actual location of your project folder.

**Example:**
```
cd "C:\Users\YourName\Desktop\AI_PROect"
```

Then press Enter.

---

## **Step 3: Install Required Packages**

Type this command and press Enter:

```
pip install -r requirements.txt
```

This will download and install all necessary libraries. It may take a few minutes. Wait until you see `Successfully installed...` message.

---

## **Step 4: Start the Application**

Type this command and press Enter:

```
python app_with_preprocessing.py
```

You should see output like:
```
* Running on local URL:  http://0.0.0.0:7860
* To create a public link, set `share=True` in `launch()`.
```

This means the app is running! ✅

---

## **Step 5: Open the Web Interface**

1. **Open your web browser** (Chrome, Firefox, Edge, etc.)
2. **Go to:** `http://localhost:7860`

You should see the **IC Chip Counterfeit Detection System** interface!

---

## **Step 6: Use the Application**

### **Tab 1: ✅ IC Verification (Main Feature)**

This is where you test IC chips:

1. **Upload Test IC Image:**
   - Click "Upload IC Image" button
   - Select a photo of an IC chip you want to verify

2. **Upload Reference Image (Optional):**
   - Click "Upload Reference IC Image" 
   - This is a known genuine IC for comparison
   - You can skip this if you only have one image

3. **Click "🔍 Verify IC Chip" Button:**
   - The system will analyze the image
   - It runs 7 different checks:
     - Logo detection
     - Text/Serial number extraction
     - QR code detection
     - Surface defects
     - Geometry verification
     - Color & texture analysis
     - Font verification

4. **See the Results:**
   - **Green = GENUINE** ✅
   - **Red = COUNTERFEIT** ⚠️
   - Full details show up below

### **Tab 2: 🖼️ Image Preprocessing**

See how images are processed internally:

1. **Upload an IC Image**
2. **Click "🔄 Run Preprocessing"** to see:
   - Original image
   - Grayscale conversion
   - Noise removal
   - Contrast enhancement
   - Edge detection
   - All processing details

3. **Click "📊 View Saved Layer Visualizations"** to see saved analysis images from previous verifications

### **Tab 3: 📚 Documentation**

Read detailed explanation of all 7 verification steps

---

## **Troubleshooting**

### **"Port 7860 already in use"**
- Another instance is running
- Close it or use a different port
- Or restart your computer

### **"ModuleNotFoundError: No module named 'gradio'"**
- The dependencies weren't installed properly
- Run: `pip install -r requirements.txt` again

### **"Cannot connect to localhost:7860"**
- Make sure the app is still running (check the terminal)
- Try using `http://127.0.0.1:7860` instead
- Wait a few seconds and refresh the page (Ctrl+R)

### **Slow processing**
- The app can take a few seconds to analyze images
- Be patient while it processes all 7 checks
- Close other heavy applications to free up memory

---

## **Tips & Tricks**

✅ **Best Results:**
- Use clear, well-lit photos of IC chips
- Straight-on angle works best
- At least 2 MP image resolution recommended

✅ **Reference Images:**
- Having a genuine IC reference image gives more accurate results
- Can test multiple chips against the same reference

✅ **Batch Testing:**
- Test multiple chips one by one
- The app will show comparison visualizations

---

## **Need Help?**

1. **App won't start?** → Make sure Python is installed and in PATH
2. **Dependencies failing?** → Try: `pip install --upgrade pip`
3. **Website won't load?** → Wait 10 seconds and refresh browser
4. **Stuck on analyzing?** → The image might be too large, try a smaller file

---

## **What the System Does**

The IC Counterfeit Detection System uses **Artificial Intelligence & Computer Vision** to detect counterfeit integrated circuits by analyzing:

1. 🏷️ **Logo authenticity** - Is the manufacturer logo genuine?
2. 📝 **Serial numbers** - Are text/markings correct?
3. 🔲 **QR/Data Matrix codes** - Do they decode properly?
4. 🔍 **Surface defects** - Are there scratches or damage?
5. 📐 **Geometry** - Is the chip size and shape correct?
6. 🎨 **Color & texture** - Do colors match authentic chips?
7. ✍️ **Font verification** - Is text printing authentic?

If **6 out of 7 tests pass**, the IC is likely **GENUINE** ✅

---

**Enjoy testing! Let me know if you have questions!** 🚀
