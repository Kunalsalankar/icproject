
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
# Import the module first, then get the functions to avoid circular import issues
import sys
import os

# Add current directory to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import using importlib to avoid circular import issues
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "complete_7step_verification",
        os.path.join(current_dir, "complete_7step_verification.py")
    )
    verification_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verification_module)
    
    # Get the functions we need
    run_complete_7step_verification = verification_module.run_complete_7step_verification
    initialize_ai_agent = verification_module.initialize_ai_agent
    
    # Initialize AI Agent on startup
    print("Initializing AI Agent...")
    initialize_ai_agent()
    print("AI Agent ready!")
    
except Exception as e:
    # Fallback to regular import if the above fails
    try:
        from complete_7step_verification import (
            run_complete_7step_verification,
            initialize_ai_agent
        )
        print("Initializing AI Agent...")
        initialize_ai_agent()
        print("AI Agent ready!")
    except ImportError as import_error:
        print(f"Error: Could not import verification module: {import_error}")
        print(f"Original error: {e}")
        print("This might be due to:")
        print("  1. Missing dependencies (install with: pip install -r requirements.txt)")
        print("  2. File not found (ensure complete_7step_verification.py is in the same directory)")
        print("  3. Circular import issue")
        
        # Define dummy functions to prevent crashes
        def run_complete_7step_verification(*args, **kwargs):
            return None
        
        def initialize_ai_agent():
            pass

def verify_ic(test_image, reference_image=None):
    """
    Main verification function for Gradio interface
    """
    try:
        # Save test image temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_test:
            test_image.save(tmp_test.name)
            test_image_path = tmp_test.name
        
        # Handle reference image
        if reference_image is not None:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_ref:
                reference_image.save(tmp_ref.name)
                reference_image_path = tmp_ref.name
        else:
            # Use default reference image
            ref_path = 'reference/golden_product.jpg'
            if os.path.exists(ref_path):
                reference_image_path = ref_path
            else:
                # Create a dummy reference path that will be handled by the verification function
                reference_image_path = None
        
        # Run complete 7-step verification
        if run_complete_7step_verification is None:
            return "Error: Verification module not loaded", json.dumps({'error': 'Module not available'}, indent=2)
        
        report = run_complete_7step_verification(
            test_image_path=test_image_path,
            reference_image_path=reference_image_path,
            logo_template_path=None,
            expected_text=None,
            expected_qr_data=None,
            color_reference=None
        )
        
        if report is None:
            return "Error: Could not process images", json.dumps({'error': 'Processing failed'}, indent=2)
        
        # Extract results
        results = report.get('pipeline_results', [])
        verdict = report.get('verdict', 'UNKNOWN')
        confidence = report.get('overall_confidence', 0.0)
        
        # Count passed tests (excluding final verdict)
        passed = sum(1 for r in results if r.get('status') == 'PASS' and r.get('step') != 'Final Verdict')
        total = len([r for r in results if r.get('step') != 'Final Verdict'])
        
        # Format results for display with better styling
        verdict_color = "#16a34a" if "GENUINE" in verdict else "#dc2626"
        confidence_percent = confidence * 100
        
        # Create animated progress bar
        progress_color = "#16a34a" if confidence >= 0.75 else "#f59e0b" if confidence >= 0.5 else "#dc2626"
        
        output_text = f"""
<div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 32px; border-radius: 16px; border: 2px solid #93c5fd; margin-bottom: 24px; box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15); position: relative; overflow: hidden;">
    <div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%); border-radius: 50%;"></div>
    <div style="position: relative; z-index: 1;">
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, {verdict_color} 0%, {verdict_color}dd 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <span style="color: white; font-size: 2em; font-weight: 800;">{'✓' if 'GENUINE' in verdict else '✗'}</span>
            </div>
            <div style="flex: 1;">
                <h1 style="color: #0f172a; margin: 0 0 8px 0; font-weight: 800; font-size: 2.2em; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">IC Verification Results</h1>
                <p style="color: #64748b; margin: 0; font-size: 1em; font-weight: 500;">7-Step Comprehensive Analysis Complete</p>
            </div>
        </div>
        
        <div style="background: white; padding: 28px; border-radius: 12px; border-left: 6px solid {verdict_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap; gap: 16px;">
                <div>
                    <h2 style="color: {verdict_color}; margin: 0 0 8px 0; font-weight: 800; font-size: 2.5em; letter-spacing: -1px;">{verdict}</h2>
                    <p style="color: #64748b; margin: 0; font-size: 0.95em; font-weight: 500;">Final Verdict</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0 0 4px 0; color: #475569; font-size: 0.9em; font-weight: 600;">Overall Confidence</p>
                    <p style="margin: 0; color: #1e40af; font-size: 2.2em; font-weight: 800;">{confidence_percent:.1f}%</p>
                </div>
            </div>
            
            <!-- Animated Progress Bar -->
            <div style="background: #e2e8f0; height: 12px; border-radius: 10px; overflow: hidden; margin-bottom: 16px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(90deg, {progress_color} 0%, {progress_color}dd 100%); height: 100%; width: {confidence_percent}%; border-radius: 10px; transition: width 1s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.2); animation: slideIn 1s ease-out;"></div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-top: 20px;">
                <div style="background: #f8fafc; padding: 16px; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0;">
                    <p style="margin: 0 0 4px 0; color: #64748b; font-size: 0.85em; font-weight: 600;">Tests Passed</p>
                    <p style="margin: 0; color: #16a34a; font-size: 1.8em; font-weight: 800;">{passed}/{total}</p>
                </div>
                <div style="background: #f8fafc; padding: 16px; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0;">
                    <p style="margin: 0 0 4px 0; color: #64748b; font-size: 0.85em; font-weight: 600;">Success Rate</p>
                    <p style="margin: 0; color: #2563eb; font-size: 1.8em; font-weight: 800;">{(passed/total*100) if total > 0 else 0:.0f}%</p>
                </div>
                <div style="background: #f8fafc; padding: 16px; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0;">
                    <p style="margin: 0 0 4px 0; color: #64748b; font-size: 0.85em; font-weight: 600;">Total Steps</p>
                    <p style="margin: 0; color: #1e40af; font-size: 1.8em; font-weight: 800;">7</p>
                </div>
            </div>
        </div>
    </div>
</div>
<style>
@keyframes slideIn {{
    from {{ width: 0%; }}
    to {{ width: {confidence_percent}%; }}
}}
</style>
"""
        
        # Add detailed test results with modern cards
        output_text += """
<div style="background: white; padding: 28px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 24px;">
    <h2 style="color: #1e40af; margin: 0 0 24px 0; border-bottom: 3px solid #dbeafe; padding-bottom: 12px; font-weight: 800; font-size: 1.8em; display: flex; align-items: center; gap: 12px;">
        <span style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2em; box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);">📊</span>
        Detailed Test Results
    </h2>
    <div style="display: grid; gap: 16px;">
"""
        
        step_icons = {
            "1. Logo Detection": "🏷️",
            "2. Text & Serial Number OCR": "📝",
            "3. QR/DMC Code Detection": "🔲",
            "4. Surface Defect Detection": "🔍",
            "5. IC Geometry & Alignment": "📐",
            "6. Color & Texture Verification": "🎨",
            "7. Font Verification & Final Correlation": "✍️"
        }
        
        # Add serial number counter
        serial_number = 0
        
        for result in results:
            step = result.get('step', 'Unknown Step')
            status = result.get('status', 'UNKNOWN')
            conf = result.get('confidence', 0.0)
            
            # Skip final verdict in detailed results
            if step == 'Final Verdict':
                continue
            
            # Increment serial number for each test result
            serial_number += 1
            
            status_color = "#16a34a" if status == "PASS" else "#dc2626" if status == "FAIL" else "#f59e0b"
            status_bg = "#dcfce7" if status == "PASS" else "#fee2e2" if status == "FAIL" else "#fef3c7"
            icon = step_icons.get(step, "✓")
            
            # Get processing time
            proc_time = result.get('processing_time_ms', 0)
            
            output_text += f"""
        <div style="background: linear-gradient(135deg, #ffffff 0%, {status_bg}15 100%); padding: 20px; border-radius: 12px; border-left: 5px solid {status_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.06); transition: all 0.3s ease; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: radial-gradient(circle, {status_color}10 0%, transparent 70%); border-radius: 0 0 0 100%;"></div>
            <div style="position: relative; z-index: 1;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div style="position: relative;">
                        <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, {status_color} 0%, {status_color}dd 100%); display: flex; align-items: center; justify-content: center; font-size: 1.5em; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                            {icon}
                        </div>
                        <div style="position: absolute; top: -8px; right: -8px; width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 0.85em; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4); border: 2px solid white;">
                            {serial_number}
                        </div>
                    </div>
                    <div style="flex: 1;">
                        <h3 style="margin: 0 0 4px 0; color: #0f172a; font-size: 1.2em; font-weight: 700;">{step}</h3>
                        <p style="margin: 0; color: #64748b; font-size: 0.9em; font-weight: 500;">Processing time: {proc_time:.0f}ms</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="background: {status_bg}; padding: 8px 16px; border-radius: 8px; display: inline-block; margin-bottom: 8px;">
                            <span style="color: {status_color}; font-weight: 700; font-size: 0.95em;">{status}</span>
                        </div>
                        <p style="margin: 0; color: #1e40af; font-size: 1.3em; font-weight: 800;">{conf*100:.1f}%</p>
                    </div>
                </div>
"""
            
            # Add sub-details for combined steps and other tests
            details = result.get('details', {})
            has_details = False
            details_html = '<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">'
            
            # Logo Detection Details - Show actual detected result
            if 'test_logo_found' in details:
                has_details = True
                logo_found = details.get('test_logo_found', False)
                # Show what was actually detected based on the test result
                if logo_found:
                    # Logo was found - show the detection confidence
                    logo_conf = details.get('test_logo_confidence', 0) * 100
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Logo:</strong> <span style="color: #0f172a; font-weight: 600;">Detected ({logo_conf:.0f}%)</span></div>'
                else:
                    # Logo was NOT found by the test
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Logo:</strong> <span style="color: #0f172a; font-weight: 600;">Not Detected</span></div>'
            
            # OCR Details - Show actual extracted text
            if 'ocr_text' in details:
                has_details = True
                ocr_text = details.get('ocr_text', 'N/A')
                if isinstance(ocr_text, str) and len(ocr_text) > 0:
                    ocr_text_display = ocr_text[:35]  # Truncate for display
                    if len(ocr_text) > 35:
                        ocr_text_display += "..."
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Text:</strong> <span style="color: #0f172a; font-weight: 600;">{ocr_text_display}</span></div>'
                else:
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Text:</strong> <span style="color: #0f172a; font-weight: 600;">No text found</span></div>'
            
            # QR Code Details - Show actual decoded data
            if 'qr_data' in details:
                has_details = True
                qr_data = details.get('qr_data', 'N/A')
                if isinstance(qr_data, str):
                    if qr_data != 'No QR code detected':
                        qr_data_display = qr_data[:35]
                        if len(qr_data) > 35:
                            qr_data_display += "..."
                        details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">QR Data:</strong> <span style="color: #0f172a; font-weight: 600;">{qr_data_display}</span></div>'
                    else:
                        details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">QR Code:</strong> <span style="color: #0f172a; font-weight: 600;">Not Found</span></div>'
            
            # Defect Detection Details - Show SSIM score
            if 'ssim_score' in details:
                has_details = True
                ssim = details.get('ssim_score', 0)
                defect_ratio = details.get('defect_ratio', 0)
                ssim_status = "Excellent" if ssim > 0.9 else "Good" if ssim > 0.8 else "Fair" if ssim > 0.7 else "Poor"
                details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">SSIM:</strong> <span style="color: #0f172a; font-weight: 600;">{ssim:.3f} ({ssim_status})</span></div>'
                if defect_ratio > 0:
                    defect_pct = defect_ratio * 100
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Defects:</strong> <span style="color: #0f172a; font-weight: 600;">{defect_pct:.1f}%</span></div>'
            
            # Edge/Geometry/Angle Details (Combined Step 5) - Show actual measurements
            if 'edge_detection' in details:
                has_details = True
                edge = details['edge_detection']
                if isinstance(edge, dict):
                    edge_status = edge.get("status", "N/A")
                    edge_conf = edge.get("confidence", 0)
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Edge:</strong> <span style="color: #0f172a; font-weight: 600;">{edge_status} ({edge_conf*100:.0f}%)</span></div>'
            
            if 'geometry_check' in details:
                has_details = True
                geom = details['geometry_check']
                if isinstance(geom, dict):
                    geom_status = geom.get("status", "N/A")
                    geom_conf = geom.get("confidence", 0)
                    # Try to get size and aspect ratio info
                    size_dev = geom.get("details", {}).get("size_deviation", 0) if isinstance(geom.get("details"), dict) else 0
                    if size_dev > 0:
                        details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Geometry:</strong> <span style="color: #0f172a; font-weight: 600;">{geom_status} (±{size_dev*100:.1f}%)</span></div>'
                    else:
                        details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Geometry:</strong> <span style="color: #0f172a; font-weight: 600;">{geom_status} ({geom_conf*100:.0f}%)</span></div>'
            
            if 'angle_detection' in details:
                has_details = True
                angle = details['angle_detection']
                if isinstance(angle, dict):
                    angle_status = angle.get("status", "N/A")
                    angle_value = angle.get("details", {}).get("detected_angle", 0) if isinstance(angle.get("details"), dict) else 0
                    if angle_value != 0:
                        details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Angle:</strong> <span style="color: #0f172a; font-weight: 600;">{angle_status} ({angle_value:.1f}°)</span></div>'
                    else:
                        details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Angle:</strong> <span style="color: #0f172a; font-weight: 600;">{angle_status}</span></div>'
            
            # Color & Texture Details (Combined Step 6)
            if 'color_verification' in details:
                has_details = True
                color = details['color_verification']
                if isinstance(color, dict):
                    color_status = color.get("status", "N/A")
                    color_details = color.get('details', {})
                    color_value = color_details.get("color_distance", 0) if isinstance(color_details, dict) else 0
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Color:</strong> <span style="color: #0f172a; font-weight: 600;">{color_status} (ΔE: {color_value:.2f})</span></div>'
            
            if 'texture_verification' in details:
                has_details = True
                texture = details['texture_verification']
                if isinstance(texture, dict):
                    texture_status = texture.get("status", "N/A")
                    details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Texture:</strong> <span style="color: #0f172a; font-weight: 600;">{texture_status}</span></div>'
            
            # Font Details (Step 7)
            if 'font_verification' in details:
                has_details = True
                font = details['font_verification']
                if isinstance(font, dict):
                    font_details = font.get('details', {})
                    if isinstance(font_details, dict):
                        font_sim = font_details.get('font_similarity_score', 0)
                        if isinstance(font_sim, (int, float)):
                            font_sim_pct = font_sim * 100
                            details_html += f'<div style="background: #f8fafc; padding: 10px; border-radius: 8px;"><strong style="color: #475569; font-size: 0.85em;">Font Sim:</strong> <span style="color: #0f172a; font-weight: 600;">{font_sim_pct:.0f}%</span></div>'
            
            details_html += '</div>'
            
            if has_details:
                output_text += details_html
            
            output_text += '            </div>\n        </div>\n'
        
        output_text += """
    </div>
</div>
"""
        
        # Create JSON output
        json_output = json.dumps(report, indent=2, default=str)
        
        # Clean up temp files
        try:
            os.unlink(test_image_path)
            if reference_image is not None:
                os.unlink(reference_image_path)
        except:
            pass
        
        return output_text, json_output
        
    except Exception as e:
        error_msg = f"Error during verification: {str(e)}"
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

css_style = """
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
"""

with gr.Blocks(title="IC Counterfeit Detection System") as demo:
    
    # Header - Light blue gradient design
    gr.HTML("""
    <div style="text-align: center; padding: 50px 40px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #93c5fd 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(147, 197, 253, 0.4); position: relative; overflow: hidden; border: 2px solid #60a5fa;">
        <div style="position: relative; z-index: 1;">
            <div style="display: inline-block; background: rgba(59,130,246,0.2); padding: 12px 24px; border-radius: 50px; margin-bottom: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59,130,246,0.3);">
                <span style="color: #1e40af; font-size: 0.9em; font-weight: 700; letter-spacing: 1px;">ADVANCED AI VERIFICATION</span>
            </div>
            <h1 style="color: #1e3a8a; margin: 0; font-size: 3em; font-weight: 800; letter-spacing: -1px; text-shadow: 0 2px 8px rgba(30, 58, 138, 0.1);">
                IC Counterfeit Detection System
            </h1>
            <p style="color: #1e40af; margin-top: 16px; font-size: 1.2em; font-weight: 600;">
                Powered by Hugging Face AI Agent + Computer Vision
            </p>
            <p style="color: #2563eb; margin-top: 8px; font-size: 1em; font-weight: 500;">
                Advanced 7-step comprehensive verification system
            </p>
            <div style="margin-top: 24px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: rgba(59,130,246,0.15); padding: 10px 18px; border-radius: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59,130,246,0.3);">
                    <span style="color: #1e40af; font-size: 0.9em; font-weight: 600;">✓ 7-Step Verification</span>
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
                "Start Verification", 
                variant="primary", 
                size="lg",
                elem_id="verify-button"
            )
            
            gr.HTML("""
            <div style="background: #eff6ff; padding: 18px; border-radius: 10px; border-left: 4px solid #2563eb; margin-top: 20px; border: 1px solid #dbeafe;">
                <h4 style="margin-top: 0; color: #1e40af; font-weight: 600;">Instructions</h4>
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
        Upload an IC image and click <strong>"Start Verification"</strong> to begin analysis
    </p>
   
</div>
                """
            )
            
            with gr.Accordion("JSON Results (for API integration)", open=False):
                json_output = gr.Code(
                    value="{}",
                    language="json",
                    lines=10
                )
    
    # Examples Section
    gr.HTML("""
    <div style="margin-top: 30px; padding: 22px; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        <h3 style="color: #2563eb; margin-top: 0; font-weight: 600;">Try Example Images</h3>
        <p style="color: #64748b; margin-bottom: 0;">Click on an example below to load it automatically</p>
    </div>
    """)
    
   
    # Connect button to function
    verify_btn.click(
        fn=verify_ic,
        inputs=[test_image, reference_image],
        outputs=[output_text, json_output]
    )
    
    # Features Section
    gr.HTML("""
    <div style="margin-top: 30px; padding: 32px; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        <h3 style="color: #2563eb; text-align: center; margin-top: 0; font-weight: 600;">Technology Stack</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; margin-top: 24px;">
            <div style="padding: 18px; background: #eff6ff; border-radius: 10px; border-left: 4px solid #2563eb; border: 1px solid #dbeafe;">
                <h4 style="margin: 0; color: #1e40af; font-weight: 600;">AI Model</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">Salesforce BLIP Vision-Language Model</p>
            </div>
            <div style="padding: 18px; background: #f0fdf4; border-radius: 10px; border-left: 4px solid #16a34a; border: 1px solid #dcfce7;">
                <h4 style="margin: 0; color: #15803d; font-weight: 600;">Framework</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">Hugging Face Transformers + PyTorch</p>
            </div>
            <div style="padding: 18px; background: #fef3c7; border-radius: 10px; border-left: 4px solid #f59e0b; border: 1px solid #fde68a;">
                <h4 style="margin: 0; color: #d97706; font-weight: 600;">Computer Vision</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">OpenCV + scikit-image</p>
            </div>
            <div style="padding: 18px; background: #fce7f3; border-radius: 10px; border-left: 4px solid #ec4899; border: 1px solid #fbcfe8;">
                <h4 style="margin: 0; color: #db2777; font-weight: 600;">OCR</h4>
                <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.9em;">Tesseract OCR Engine</p>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 20px; padding: 32px; background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%); border-radius: 16px; border: 2px solid #dbeafe; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
        <h3 style="color: #1e40af; text-align: center; margin-top: 0; font-weight: 800; font-size: 1.8em; margin-bottom: 8px;">7-Step Verification System</h3>
        <p style="text-align: center; color: #64748b; margin-bottom: 24px; font-weight: 500;">Comprehensive counterfeit detection with combined analysis layers</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 24px;">
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">🏷️</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 1: Logo Detection</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">HoughCircles-based fast circular logo detection</p>
            </div>
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #10b981 0%, #059669 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">📝</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 2: Text & OCR</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">OCR confidence ≥ 80% for text and serial numbers</p>
            </div>
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #f59e0b 0%, #d97706 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">🔲</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 3: QR/DMC Code</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">≥ 80% success rate for code detection</p>
            </div>
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #ec4899 0%, #db2777 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">🔍</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 4: Surface Defects</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">SSIM + Intensity difference analysis</p>
            </div>
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #8b5cf6 0%, #7c3aed 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">📐</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 5: Geometry & Alignment</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">Edge detection + Size/Aspect + Angle (Combined)</p>
            </div>
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #06b6d4 0%, #0891b2 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">🎨</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 6: Color & Texture</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">Color ΔE < 3-5 (LAB/HSV) + Texture analysis</p>
            </div>
            <div style="padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e2e8f0; transition: all 0.3s ease; position: relative; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: linear-gradient(180deg, #f97316 0%, #ea580c 100%);"></div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 1.8em;">✍️</span>
                    <strong style="color: #1e40af; font-size: 1.1em;">Step 7: Font & Correlation</strong>
                </div>
                <p style="color: #64748b; margin: 0; font-size: 0.9em; line-height: 1.5;">Font verification + Final correlation analysis</p>
            </div>
        </div>
    </div>
    
    <div style="margin-top: 20px; padding: 24px; background: white; border-radius: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        <p style="color: #64748b; margin: 0; font-size: 0.9em;">
            Made with Hugging Face Spaces | 
            <a href="https://huggingface.co/Salesforce/blip-image-captioning-large" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">BLIP Model</a> | 
            <a href="https://github.com" target="_blank" style="color: #2563eb; text-decoration: none; font-weight: 500;">GitHub</a>
        </p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        css=css_style,
        share=False,  # Disabled for Hugging Face Spaces
        server_name="0.0.0.0",
        server_port=7860
    )