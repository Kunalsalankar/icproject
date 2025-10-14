# 🚀 Deployment Guide: Hugging Face Spaces

## Complete Guide to Deploy Your IC Verification System

---

## 📋 Prerequisites

1. **Hugging Face Account** (Free)
   - Sign up at: https://huggingface.co/join
   
2. **Git Installed**
   - Download: https://git-scm.com/downloads

3. **Files Ready**
   - ✅ `app.py` (Gradio interface)
   - ✅ `complete_7step_verification.py` (main code)
   - ✅ `requirements.txt` (dependencies)
   - ✅ `README_HUGGINGFACE.md` (documentation)

---

## 🎯 Method 1: Deploy via Hugging Face Web Interface (Easiest)

### **Step 1: Create New Space**
2. Click **"Create new Space"**
   - **Space name**: `ic-counterfeit-detection`
   - **License**: MIT
   - **SDK**: Select **Gradio**
   - **Hardware**: CPU (free) or GPU (paid)
   - **Visibility**: Public

4. Click **"Create Space"**

### **Step 2: Upload Files**

You'll see a file upload interface. Upload these files:

1. **app.py** (your Gradio interface)
2. **complete_7step_verification.py** (main verification code)
3. **requirements.txt** (dependencies)
4. **README.md** (rename README_HUGGINGFACE.md to README.md)

**Optional:**
5. Upload `reference/golden_product.jpg` (reference image)
6. Upload `test_images/product_to_verify.jpg` (example image)

### **Step 3: Wait for Build**

- Hugging Face will automatically:
  - Install dependencies from `requirements.txt`
  - Build the Docker container
  - Start the Gradio app
  - Generate public URL

- **Build time**: 5-10 minutes (first time)
- **Status**: Watch the "Building" indicator

### **Step 4: Get Your URL**

Once built, you'll get a public URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/ic-counterfeit-detection
```

**Share this URL** with anyone to use your IC verification system!

---

## 🎯 Method 2: Deploy via Git (Advanced)

### **Step 1: Install Git LFS**

```bash
git lfs install
```

### **Step 2: Clone Your Space**

```bash
# Replace YOUR_USERNAME with your Hugging Face username
git clone https://huggingface.co/spaces/YOUR_USERNAME/ic-counterfeit-detection
cd ic-counterfeit-detection
```

### **Step 3: Copy Files**

Copy these files to the cloned directory:
```bash
# From your project directory
cp app.py ic-counterfeit-detection/
cp complete_7step_verification.py ic-counterfeit-detection/
cp requirements.txt ic-counterfeit-detection/
cp README_HUGGINGFACE.md ic-counterfeit-detection/README.md
```

### **Step 4: Create Space Metadata**

Create a file named `README.md` with this header:
```yaml
---
title: IC Counterfeit Detection
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---
```

### **Step 5: Push to Hugging Face**

```bash
git add .
git commit -m "Initial deployment"
git push
```

### **Step 6: Access Your URL**

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/ic-counterfeit-detection`

---

## 🧪 Test Locally First

Before deploying, test locally:

### **Install Gradio**
```bash
pip install gradio
```

### **Run App**
```bash
python app.py
```

### **Test Interface**
- Open: http://localhost:7860
- Upload test image
- Click "Verify IC"
- Check results

If it works locally, it will work on Hugging Face!

---

## 📱 Using Your Deployed App

### **Web Interface**

1. Visit your Space URL
2. Upload IC image
3. Click "Verify IC"
4. Wait 10-15 seconds
5. View results

### **API Access**

Use the Gradio API client:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/ic-counterfeit-detection")

# Upload image and get results
result = client.predict(
    test_image="path/to/ic_image.jpg",
    reference_image=None,
    api_name="/predict"
)

print(result)
```

### **Direct API Call**

```python
import requests

url = "https://YOUR_USERNAME-ic-counterfeit-detection.hf.space/api/predict"

files = {
    'test_image': open('ic_image.jpg', 'rb')
}

response = requests.post(url, files=files)
print(response.json())
```

---

## ⚙️ Configuration Options

### **Change Hardware**

In Space settings:
- **CPU Basic** (Free): 2 vCPU, 16GB RAM
- **CPU Upgrade** ($0.60/hour): 8 vCPU, 32GB RAM
- **GPU T4** ($0.60/hour): 1x NVIDIA T4, 16GB VRAM
- **GPU A10G** ($3.15/hour): 1x NVIDIA A10G, 24GB VRAM

**Recommendation**: Start with free CPU, upgrade to GPU if needed.

### **Enable GPU in app.py**

The app automatically detects GPU. No changes needed!

### **Adjust Model**

For faster inference, edit `complete_7step_verification.py`:
```python
# Line 122: Use smaller model
AI_MODEL_NAME = "Salesforce/blip-image-captioning-base"
```

---

## 🔧 Troubleshooting

### **Issue: Build Failed**

**Solution**: Check `requirements.txt` for incompatible versions
```bash
# Simplify requirements
transformers
torch
gradio
opencv-python
numpy
scikit-image
Pillow
pytesseract
```

### **Issue: Out of Memory**

**Solution 1**: Use smaller model
```python
AI_MODEL_NAME = "Salesforce/blip-image-captioning-base"
```

**Solution 2**: Upgrade to GPU hardware

### **Issue: Tesseract Not Found**

**Solution**: Add to `packages.txt` (Hugging Face will install):
```
tesseract-ocr
```

Create file `packages.txt` in your Space:
```
tesseract-ocr
libtesseract-dev
```

### **Issue: Slow Processing**

**Solutions**:
1. Upgrade to GPU hardware
2. Use smaller BLIP model
3. Reduce image resolution in preprocessing

---

## 📊 Monitoring & Analytics

### **View Logs**

In your Space:
1. Click "Logs" tab
2. See real-time processing logs
3. Debug errors

### **Check Usage**

- View visitor count
- Monitor API calls
- Track processing time

### **Analytics Dashboard**

Hugging Face provides:
- Daily active users
- Total requests
- Error rates
- Processing time stats

---

## 🌐 Share Your Space

### **Embed in Website**

```html
<iframe
  src="https://YOUR_USERNAME-ic-counterfeit-detection.hf.space"
  width="100%"
  height="800px"
></iframe>
```

### **Share on Social Media**

Hugging Face auto-generates preview cards:
- Twitter/X
- LinkedIn
- Facebook

### **Add to Portfolio**

Your Space URL is a live demo of your AI project!

---

## 💰 Cost Estimation

### **Free Tier**
- CPU Basic: **FREE**
- Storage: 50GB free
- Bandwidth: Unlimited
- **Perfect for demos and testing**

### **Paid Tiers**
- CPU Upgrade: $0.60/hour (~$432/month if always on)
- GPU T4: $0.60/hour (~$432/month)
- GPU A10G: $3.15/hour (~$2,268/month)

**Recommendation**: 
- Use **FREE CPU** for demos
- Use **GPU** only when actively testing
- Enable "Sleep after inactivity" to save costs

---

## 🎯 Quick Start Checklist

- [ ] Create Hugging Face account
- [ ] Create new Space (Gradio SDK)
- [ ] Upload `app.py`
- [ ] Upload `complete_7step_verification.py`
- [ ] Upload `requirements.txt`
- [ ] Upload `README.md`
- [ ] Wait for build (5-10 min)
- [ ] Test with sample image
- [ ] Share your URL! 🎉

---

## 📞 Support

### **Hugging Face Docs**
- Spaces: https://huggingface.co/docs/hub/spaces
- Gradio: https://www.gradio.app/docs/

### **Community**
- Forum: https://discuss.huggingface.co/
- Discord: https://discord.gg/hugging-face

---

## 🎉 Success!

Once deployed, your IC verification system will be:
- ✅ Publicly accessible via URL
- ✅ Shareable with anyone
- ✅ Embeddable in websites
- ✅ API-accessible for integration
- ✅ Automatically scaled by Hugging Face

**Your URL will look like:**
```
https://huggingface.co/spaces/YOUR_USERNAME/ic-counterfeit-detection
```

**Share it with the world!** 🚀
