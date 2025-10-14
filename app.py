"""
Gradio Web Interface for IC Counterfeit Detection
Deploy to Hugging Face Spaces for public URL
"""

import gradio as gr
import cv2
import numpy as np
import json
from PIL import Image
import os

# Import verification functions
from complete_7step_verification import (
    ai_agent_oem_verification,
    detect_logo_template,
    detect_and_read_text,
    detect_defects,
    detect_edges_canny,
    check_ic_geometry,
    check_angle_detection,
    verify_color,  # Fixed: was verify_color_surface
    verify_texture,
    verify_font_characteristics,
    initialize_ai_agent
)

# Initialize AI Agent on startup
print("🚀 Initializing AI Agent...")
initialize_ai_agent()
print("✅ AI Agent ready!")

def verify_ic(test_image, reference_image=None):
    """
    Main verification function for Gradio interface
    """
    try:
        # Convert PIL Image to OpenCV format
        test_img_cv = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)
        
        if reference_image is not None:
            ref_img_cv = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
        else:
            # Use default reference image
            ref_path = 'reference/golden_product.jpg'
            if os.path.exists(ref_path):
                ref_img_cv = cv2.imread(ref_path)
            else:
                ref_img_cv = None
        
        results = []
        
        # Step 1: Logo Detection
        logo_result = detect_logo_template(test_img_cv)
        results.append(logo_result)
        
        # Step 2: AI Agent OEM Verification
        ai_result = ai_agent_oem_verification(test_img_cv)
        results.append(ai_result)
        
        # Step 3: OCR
        ocr_result = detect_and_read_text(test_img_cv)
        results.append(ocr_result)
        
        # Additional checks if reference image provided
        if ref_img_cv is not None:
            # Surface Defect Detection
            defect_result = detect_defects(test_img_cv, None)
            results.append(defect_result)
            
            # Edge Detection
            edge_result = detect_edges_canny(test_img_cv)
            results.append(edge_result)
            
            # Geometry Check
            geom_result = check_ic_geometry(test_img_cv, None)
            results.append(geom_result)
        
        # Calculate overall verdict
        passed = sum(1 for r in results if r.get('status') == 'PASS')
        total = len(results)
        confidence = sum(r.get('confidence', 0) for r in results) / total if total > 0 else 0
        
        verdict = "✅ GENUINE" if confidence > 0.7 else "❌ COUNTERFEIT"
        
        # Format results for display with better styling
        verdict_color = "#16a34a" if "GENUINE" in verdict else "#dc2626"
        
        output_text = f"""
<div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 24px; border-radius: 12px; border: 2px solid #93c5fd; margin-bottom: 20px;">
    <h1 style="color: #1e3a8a; margin: 0 0 16px 0;">🔍 IC Verification Results</h1>
    <div style="background: white; padding: 20px; border-radius: 10px; border-left: 5px solid {verdict_color};">
        <h2 style="color: {verdict_color}; margin: 0 0 12px 0;">{verdict}</h2>
        <p style="margin: 4px 0; color: #475569;"><strong>Overall Confidence:</strong> <span style="color: #1e40af; font-size: 1.2em; font-weight: 600;">{confidence:.1%}</span></p>
        <p style="margin: 4px 0; color: #475569;"><strong>Tests Passed:</strong> <span style="color: #1e40af; font-weight: 600;">{passed}/{total}</span></p>
    </div>
</div>
"""
        
        # Add AI Agent results with better formatting
        for result in results:
            if 'AI Agent' in result.get('step', ''):
                details = result.get('details', {})
                ai_conf = details.get('ai_confidence', 0)
                ai_conf_color = "#16a34a" if ai_conf > 0.5 else "#f59e0b"
                
                output_text += f"""
<div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h2 style="color: #2563eb; margin: 0 0 16px 0; border-bottom: 2px solid #dbeafe; padding-bottom: 8px;">🤖 AI Agent Analysis</h2>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
        <div style="background: #f8fafc; padding: 12px; border-radius: 8px;">
            <p style="margin: 0; color: #64748b; font-size: 0.85em;">Part Number</p>
            <p style="margin: 4px 0 0 0; color: #1e293b; font-weight: 600; font-size: 1.1em;">{details.get('part_number', 'Unknown')}</p>
        </div>
        <div style="background: #f8fafc; padding: 12px; border-radius: 8px;">
            <p style="margin: 0; color: #64748b; font-size: 0.85em;">Manufacturer</p>
            <p style="margin: 4px 0 0 0; color: #1e293b; font-weight: 600; font-size: 1.1em;">{details.get('oem_manufacturer', 'Unknown')}</p>
        </div>
        <div style="background: #f8fafc; padding: 12px; border-radius: 8px;">
            <p style="margin: 0; color: #64748b; font-size: 0.85em;">Package</p>
            <p style="margin: 4px 0 0 0; color: #1e293b; font-weight: 600;">{details.get('oem_package', 'Unknown')}</p>
        </div>
        <div style="background: #f8fafc; padding: 12px; border-radius: 8px;">
            <p style="margin: 0; color: #64748b; font-size: 0.85em;">Status</p>
            <p style="margin: 4px 0 0 0; color: #1e293b; font-weight: 600;">{details.get('oem_status', 'Unknown')}</p>
        </div>
    </div>
    <div style="background: #eff6ff; padding: 12px; border-radius: 8px; margin-top: 12px; border-left: 4px solid {ai_conf_color};">
        <p style="margin: 0; color: #64748b; font-size: 0.85em;">AI Confidence</p>
        <p style="margin: 4px 0 0 0; color: {ai_conf_color}; font-weight: 700; font-size: 1.3em;">{ai_conf:.1%}</p>
        <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;"><strong>Method:</strong> {details.get('method', 'N/A')}</p>
    </div>
    <div style="background: #f8fafc; padding: 12px; border-radius: 8px; margin-top: 12px;">
        <p style="margin: 0; color: #64748b; font-size: 0.85em;">Description</p>
        <p style="margin: 4px 0 0 0; color: #475569; line-height: 1.5;">{details.get('oem_description', 'Not found in database')}</p>
    </div>
</div>
"""
        
        # Add other test results with better formatting
        output_text += """
<div style="background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h2 style="color: #2563eb; margin: 0 0 16px 0; border-bottom: 2px solid #dbeafe; padding-bottom: 8px;">📊 Detailed Test Results</h2>
"""
        
        for i, result in enumerate(results, 1):
            step = result.get('step', f'Test {i}')
            status = result.get('status', 'UNKNOWN')
            conf = result.get('confidence', 0)
            
            if 'AI Agent' in step:
                continue  # Skip AI Agent as we already showed it
            
            status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            status_color = "#16a34a" if status == "PASS" else "#dc2626" if status == "FAIL" else "#f59e0b"
            
            output_text += f"""
    <div style="background: #f8fafc; padding: 14px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {status_color};">
        <h3 style="margin: 0 0 8px 0; color: #1e293b; font-size: 1.05em;">{status_emoji} {step}</h3>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
            <span style="color: #64748b;"><strong>Status:</strong> <span style="color: {status_color}; font-weight: 600;">{status}</span></span>
            <span style="color: #64748b;"><strong>Confidence:</strong> <span style="color: #2563eb; font-weight: 600;">{conf:.1%}</span></span>
        </div>
"""
            
            # Add specific details
            details = result.get('details', {})
            if details:
                output_text += '<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e2e8f0;">'
                for key, value in list(details.items())[:3]:  # Show first 3 details
                    if isinstance(value, (int, float, str, bool)) and key not in ['part_number', 'oem_manufacturer', 'oem_description', 'oem_package', 'oem_status', 'ai_confidence', 'method']:
                        output_text += f'<span style="color: #64748b; font-size: 0.9em; margin-right: 15px;">• <strong>{key}:</strong> {value}</span>'
                output_text += '</div>'
            
            output_text += '    </div>\n'
        
        output_text += "</div>"
        
        # Create JSON output
        json_output = json.dumps({
            'verdict': verdict,
            'confidence': float(confidence),
            'tests_passed': f"{passed}/{total}",
            'results': results
        }, indent=2, default=str)
        
        return output_text, json_output
        
    except Exception as e:
        error_msg = f"❌ Error during verification: {str(e)}"
        return error_msg, json.dumps({'error': str(e)}, indent=2)


# Create Gradio Interface with Professional Light Theme
custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
).set(
    # Professional color scheme
    body_background_fill="#f8fafc",  # Softer background
    body_background_fill_dark="#f8fafc",
    block_background_fill="white",
    block_background_fill_dark="white",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    button_primary_background_fill="#2563eb",  # Professional blue
    button_primary_background_fill_hover="#1d4ed8",  # Darker blue on hover
    button_primary_text_color="white",
    block_title_text_color="#0f172a",  # Darker text
    block_title_text_color_dark="#0f172a",
    block_label_text_color="#475569",  # Medium gray
    block_label_text_color_dark="#475569",
    body_text_color="#1e293b",  # Dark slate
    body_text_color_dark="#1e293b",
    block_title_text_weight="600",
    block_label_text_weight="500",
    panel_background_fill="white",
    panel_background_fill_dark="white",
    border_color_primary="#e2e8f0",  # Light border
    block_title_background_fill="#eff6ff",  # Light blue header
    block_title_background_fill_dark="#eff6ff",  # Override dark mode
)

with gr.Blocks(title="IC Counterfeit Detection System", theme=custom_theme, css="""
    /* Professional Light Theme with Modern Design */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    body, .gradio-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
        padding: 20px !important;
    }
    
    /* Override dark mode */
    .dark, [data-theme="dark"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%) !important;
        color: #0f172a !important;
    }
    
    /* Smooth animations */
    * {
        transition: all 0.3s ease !important;
    }
    
    /* Upload areas - Professional with hover effects */
    .image-container, [data-testid="image"] {
        background: white !important;
        border: 3px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
    }
    
    .image-container:hover, [data-testid="image"]:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Remove ALL dark headers from upload boxes */
    .image-container > div:first-child,
    [data-testid="image"] > div:first-child,
    .image-frame,
    .upload-container > div:first-child,
    div[class*="header"],
    div[data-testid*="header"] {
        background: #eff6ff !important;
        background-color: #eff6ff !important;
        border-bottom: 1px solid #dbeafe !important;
    }
    
    /* Force light background on all image upload components */
    .svelte-1b19cri,
    .image-container,
    [data-testid="image"],
    .upload-container {
        background: #f8fafc !important;
    }
    
    /* Upload box header buttons - light style */
    .image-container button,
    [data-testid="image"] button,
    button[class*="icon"] {
        background: #f1f5f9 !important;
        color: #64748b !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .image-container button:hover,
    [data-testid="image"] button:hover {
        background: #dbeafe !important;
        color: #2563eb !important;
    }
    
    /* Remove ANY dark/black backgrounds - AGGRESSIVE */
    div[style*="background: rgb(0, 0, 0)"],
    div[style*="background-color: rgb(0, 0, 0)"],
    div[style*="background: black"],
    div[style*="background-color: black"],
    div[style*="rgb(15, 23, 42)"],
    div[style*="rgb(30, 41, 59)"],
    div[class*="dark"] {
        background: #eff6ff !important;
        background-color: #eff6ff !important;
    }
    
    /* Target Gradio's specific dark header class */
    .image-container header,
    [data-testid="image"] header,
    div[class*="image"] header,
    .block header,
    header[class*="svelte"] {
        background: #eff6ff !important;
        background-color: #eff6ff !important;
        border-bottom: 1px solid #dbeafe !important;
    }
    
    /* HIDE the dark header bar completely */
    .image-container > div:first-child > div:first-child,
    [data-testid="image"] > div:first-child > div:first-child {
        display: none !important;
    }
    
    /* Alternative: Make it transparent and very small */
    .image-container > div[style*="background"],
    [data-testid="image"] > div[style*="background"] {
        background: transparent !important;
        min-height: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    
    /* Override inline styles */
    * {
        --block-title-background-fill: #eff6ff !important;
        --block-background-fill: #f8fafc !important;
    }
    
    /* Blocks and panels - Enhanced */
    .block, .panel {
        background: white !important;
        color: #0f172a !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        border: 1px solid #e2e8f0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .block:hover, .panel:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
    }
    
    /* Labels and text */
    label, .label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    span, p {
        color: #475569 !important;
    }
    
    h1, h2, h3, h4 {
        color: #0f172a !important;
    }
    
    /* Primary button - Enhanced with animation */
    button[variant="primary"], .primary-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05em !important;
        padding: 16px 32px !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    button[variant="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-3px) scale(1.02) !important;
    }
    
    button[variant="primary"]:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* Code blocks */
    .code-container, pre, code {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Accordion */
    .accordion {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Input fields */
    input, textarea {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
        border-radius: 8px !important;
    }
    
    input:focus, textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
""") as demo:
    
    # Header - Light blue gradient design
    gr.HTML("""
    <div style="text-align: center; padding: 50px 40px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #93c5fd 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(147, 197, 253, 0.4); position: relative; overflow: hidden; border: 2px solid #60a5fa;">
        <div style="position: relative; z-index: 1;">
            <div style="display: inline-block; background: rgba(59,130,246,0.2); padding: 12px 24px; border-radius: 50px; margin-bottom: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59,130,246,0.3);">
                <span style="color: #1e40af; font-size: 0.9em; font-weight: 700; letter-spacing: 1px;">ADVANCED AI VERIFICATION</span>
            </div>
            <h1 style="color: #1e3a8a; margin: 0; font-size: 3em; font-weight: 800; letter-spacing: -1px; text-shadow: 0 2px 8px rgba(30, 58, 138, 0.1);">
                🔬 IC Counterfeit Detection System
            </h1>
            <p style="color: #1e40af; margin-top: 16px; font-size: 1.2em; font-weight: 600;">
                Powered by Hugging Face AI Agent + Computer Vision
            </p>
            <p style="color: #2563eb; margin-top: 8px; font-size: 1em; font-weight: 500;">
                Advanced 11-layer verification using BLIP Vision-Language Model
            </p>
            <div style="margin-top: 24px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: rgba(59,130,246,0.15); padding: 10px 18px; border-radius: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59,130,246,0.3);">
                    <span style="color: #1e40af; font-size: 0.9em; font-weight: 600;">✓ 11 Detection Layers</span>
                </div>
                <div style="background: rgba(59,130,246,0.15); padding: 10px 18px; border-radius: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59,130,246,0.3);">
                    <span style="color: #1e40af; font-size: 0.9em; font-weight: 600;">✓ AI-Powered Analysis</span>
                </div>
                <div style="background: rgba(59,130,246,0.15); padding: 10px 18px; border-radius: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59,130,246,0.3);">
                    <span style="color: #1e40af; font-size: 0.9em; font-weight: 600;">✓ Real-time Results</span>
                </div>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, white 0%, #f8fafc 100%); padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; margin-bottom: 20px; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: linear-gradient(135deg, #3b82f6 0%, transparent 100%); opacity: 0.1; border-radius: 0 0 0 100%;"></div>
                <div style="position: relative; z-index: 1;">
                    <h3 style="color: #2563eb; margin: 0 0 8px 0; font-weight: 700; font-size: 1.3em; display: flex; align-items: center; gap: 10px;">
                        <span style="background: linear-gradient(135deg, #3b82f6, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.2em;">📸</span>
                        Upload IC Images
                    </h3>
                    <p style="color: #64748b; margin: 0; font-size: 0.95em; line-height: 1.5;">Upload clear, high-resolution images of the IC chip for comprehensive verification</p>
                </div>
            </div>
            """)
            
            gr.Markdown("**Test IC Image** (Required)")
            test_image = gr.Image(
                type="pil",
                label="",
                show_label=False,
                height=320,
                elem_classes="upload-box"
            )
            
            gr.Markdown("**Reference IC Image** (Optional)")
            reference_image = gr.Image(
                type="pil",
                label="",
                show_label=False,
                height=320,
                elem_classes="upload-box"
            )
            
            verify_btn = gr.Button(
                "🚀 Start Verification", 
                variant="primary", 
                size="lg",
                elem_id="verify-button"
            )
            
            gr.HTML("""
            <div style="background: #eff6ff; padding: 18px; border-radius: 10px; border-left: 4px solid #2563eb; margin-top: 20px; border: 1px solid #dbeafe;">
                <h4 style="margin-top: 0; color: #1e40af; font-weight: 600;">📋 Instructions</h4>
                <ol style="margin: 0; padding-left: 20px; color: #475569; line-height: 1.8;">
                    <li>Upload the IC chip image you want to verify</li>
                    <li>Optionally upload a reference (genuine) IC image</li>
                    <li>Click "Start Verification" to begin analysis</li>
                </ol>
            </div>
            """)
    
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, white 0%, #f8fafc 100%); padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; margin-bottom: 20px; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, transparent 100%); opacity: 0.1; border-radius: 0 0 0 100%;"></div>
                <div style="position: relative; z-index: 1;">
                    <h3 style="color: #2563eb; margin: 0 0 8px 0; font-weight: 700; font-size: 1.3em; display: flex; align-items: center; gap: 10px;">
                        <span style="background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.2em;">📊</span>
                        Verification Results
                    </h3>
                    <p style="color: #64748b; margin: 0; font-size: 0.95em; line-height: 1.5;">Comprehensive analysis with AI-powered confidence scores and detailed test results</p>
                </div>
            </div>
            """)
            
            output_text = gr.HTML(
                value="""
<div style="background: #eff6ff; padding: 24px; border-radius: 12px; text-align: center; border: 1px solid #dbeafe;">
    <p style="color: #1e40af; font-size: 1.1em; margin: 0; font-weight: 500;">
        ⬆️ Upload an IC image and click <strong>"Start Verification"</strong> to begin analysis
    </p>
   
</div>
                """
            )
            
            with gr.Accordion("📄 JSON Results (for API integration)", open=False):
                json_output = gr.Code(
                    value="{}",
                    language="json",
                    lines=10
                )
    
    # Examples Section
    gr.HTML("""
    <div style="margin-top: 30px; padding: 22px; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        <h3 style="color: #2563eb; margin-top: 0; font-weight: 600;">📋 Try Example Images</h3>
        <p style="color: #64748b; margin-bottom: 0;">Click on an example below to load it automatically</p>
    </div>
    """)
    
    gr.Examples(
        examples=[
            ["test_images/product_to_verify.jpg", None],
        ],
        inputs=[test_image, reference_image],
    )
    
    # Connect button to function
    verify_btn.click(
        fn=verify_ic,
        inputs=[test_image, reference_image],
        outputs=[output_text, json_output]
    )
    
    # Features Section
    gr.HTML("""
    <div style="margin-top: 30px; padding: 32px; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        <h3 style="color: #2563eb; text-align: center; margin-top: 0; font-weight: 600;">🛠️ Technology Stack</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; margin-top: 24px;">
            <div style="padding: 18px; background: #eff6ff; border-radius: 10px; border-left: 4px solid #2563eb; border: 1px solid #dbeafe;">
                <h4 style="margin: 0; color: #1e40af; font-weight: 600;">🤖 AI Model</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">Salesforce BLIP Vision-Language Model</p>
            </div>
            <div style="padding: 18px; background: #f0fdf4; border-radius: 10px; border-left: 4px solid #16a34a; border: 1px solid #dcfce7;">
                <h4 style="margin: 0; color: #15803d; font-weight: 600;">🔧 Framework</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">Hugging Face Transformers + PyTorch</p>
            </div>
            <div style="padding: 18px; background: #fef3c7; border-radius: 10px; border-left: 4px solid #f59e0b; border: 1px solid #fde68a;">
                <h4 style="margin: 0; color: #d97706; font-weight: 600;">👁️ Computer Vision</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">OpenCV + scikit-image</p>
            </div>
            <div style="padding: 18px; background: #fce7f3; border-radius: 10px; border-left: 4px solid #ec4899; border: 1px solid #fbcfe8;">
                <h4 style="margin: 0; color: #db2777; font-weight: 600;">📝 OCR</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">Tesseract OCR Engine</p>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 20px; padding: 32px; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h3 style="color: #2563eb; text-align: center; margin-top: 0; font-weight: 600;">📖 11-Layer Verification System</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 24px;">
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; transition: all 0.2s;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Logo Detection</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">AI Part Number Extraction</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">OCR Text Reading</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">QR/DMC Detection</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Surface Defect Analysis</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Edge Detection</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Geometry Analysis</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Angle Detection</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Color Verification</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Texture Analysis</span>
            </div>
            <div style="padding: 14px; background: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <strong style="color: #2563eb;">✓</strong> <span style="color: #475569;">Font Verification</span>
            </div>
        </div>
       
    </div>
    
    <div style="margin-top: 20px; padding: 24px; background: white; border-radius: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        <p style="color: #64748b; margin: 0; font-size: 0.9em;">
            Made with ❤️ using Hugging Face Spaces | 
            <a href="https://huggingface.co/Salesforce/blip-image-captioning-large" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">BLIP Model</a> | 
            <a href="https://github.com" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">GitHub</a>
        </p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates public URL
        server_name="0.0.0.0",
        server_port=7860
    )
