"""
Gradio Web Interface for IC Counterfeit Detection WITH IMAGE PREPROCESSING VISUALIZATION
Deploy to Hugging Face Spaces for public URL
Features:
- 7-Step verification pipeline
- Real-time image preprocessing visualization
- Side-by-side comparison of original vs preprocessed images
- Intermediate processing steps display (grayscale, edges, contrast, etc.)
"""

import gradio as gr
import cv2
import numpy as np
import json
from PIL import Image
import os
import tempfile

# Import verification functions
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
    preprocess_image_comprehensive = verification_module.preprocess_image_comprehensive
    
    # Initialize AI Agent on startup
    print("Initializing AI Agent...")
    initialize_ai_agent()
    print("AI Agent ready!")
    print("Preprocessing module loaded successfully!")
    
except Exception as e:
    # Fallback to regular import if the above fails
    try:
        from complete_7step_verification import (
            run_complete_7step_verification,
            initialize_ai_agent,
            preprocess_image_comprehensive
        )
        print("Initializing AI Agent...")
        initialize_ai_agent()
        print("AI Agent ready!")
        print("Preprocessing module loaded successfully!")
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
        
        def preprocess_image_comprehensive(*args, **kwargs):
            return None

# ============================================================================
# LOAD LAYER VISUALIZATIONS FROM TEST FOLDER
# ============================================================================

def load_layer_visualizations():
    """
    Load all saved layer visualization images from test_layer_visualizations folder
    Returns dictionary of images organized by layer type
    """
    viz_folder = "test_layer_visualizations"
    layer_images = {}
    
    if not os.path.exists(viz_folder):
        return {}
    
    # Get all jpg files and sort them
    files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.jpg')])
    
    for filename in files:
        filepath = os.path.join(viz_folder, filename)
        try:
            img = Image.open(filepath)
            layer_images[filename] = img
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return layer_images

# ============================================================================
# PREPROCESSING VISUALIZATION FUNCTION
# ============================================================================

def visualize_preprocessing(test_image):
    """
    Comprehensive image preprocessing visualization
    Shows original and all intermediate processing steps
    """
    if test_image is None:
        return None, "Please upload an image first", None, None, None, None, None, None
    
    try:
        # Convert PIL Image to OpenCV format if needed
        if isinstance(test_image, Image.Image):
            test_image_cv = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)
        else:
            test_image_cv = test_image
        
        # Run comprehensive preprocessing
        print("\n" + "="*70)
        print("PREPROCESSING VISUALIZATION - Starting...")
        print("="*70)
        
        preprocessing_result = preprocess_image_comprehensive(
            test_image_cv,
            target_size=640,
            save_steps=True
        )
        
        if preprocessing_result is None:
            return None, "Preprocessing failed", None, None, None, None, None, None
        
        # Extract results
        preprocessed = preprocessing_result['preprocessed']
        original_resized = preprocessing_result['original_resized']
        intermediate = preprocessing_result['intermediate_steps']
        metadata = preprocessing_result['metadata']
        
        # Convert BGR to RGB for display
        original_display = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB)
        preprocessed_display = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        grayscale_display = cv2.cvtColor(intermediate['grayscale'], cv2.COLOR_GRAY2RGB)
        denoised_display = cv2.cvtColor(intermediate['denoised'], cv2.COLOR_GRAY2RGB)
        contrast_display = cv2.cvtColor(intermediate['contrast_enhanced'], cv2.COLOR_GRAY2RGB)
        threshold_display = cv2.cvtColor(intermediate['adaptive_threshold'], cv2.COLOR_GRAY2RGB)
        edges_display = cv2.cvtColor(intermediate['edges'], cv2.COLOR_GRAY2RGB)
        
        # Create metadata text
        metadata_text = f"""
        IMAGE PREPROCESSING METADATA
        
Original Size: {metadata['original_size'][0]}x{metadata['original_size'][1]} px
Target Size: {metadata['target_size'][0]}x{metadata['target_size'][1]} px
Aspect Ratio: {metadata['aspect_ratio']:.2f}
Processing Time: {metadata['processing_time_ms']:.2f} ms

TECHNIQUES APPLIED:
"""
        for i, technique in enumerate(metadata['techniques_applied'], 1):
            metadata_text += f"{i}. {technique}\n"
        
        metadata_text += """
PREPROCESSING PIPELINE:
1. Resize → Fixed 640x640 size with aspect ratio preserved
2. Grayscale → Convert color to single channel
3. Denoise → Bilateral filtering (d=9, σ_color=75, σ_space=75)
4. CLAHE → Contrast Limited Adaptive Histogram Equalization
5. Threshold → Adaptive Gaussian thresholding for OCR
6. Edges → Canny edge detection (100-200 threshold)

USES:
- Preprocessed image: Logo detection, OCR, text recognition
- Original resized: Defect detection, color analysis, texture
- Intermediate steps: Feature extraction, geometry analysis
"""
        
        print(metadata_text)
        
        # Return all images and metadata
        return (
            original_display,
            "✅ Preprocessing Complete",
            preprocessed_display,
            grayscale_display,
            denoised_display,
            contrast_display,
            threshold_display,
            edges_display,
            metadata_text
        )
        
    except Exception as e:
        error_msg = f"Error during preprocessing: {str(e)}"
        print(error_msg)
        return None, f"❌ {error_msg}", None, None, None, None, None, None, f"Error: {str(e)}"


# ============================================================================
# MAIN VERIFICATION FUNCTION
# ============================================================================

def verify_ic(test_image, reference_image=None):
    """
    Main verification function for Gradio interface
    Includes preprocessing before verification
    """
    try:
        # Save test image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_test:
            if isinstance(test_image, Image.Image):
                test_image.save(tmp_test.name)
            else:
                cv2.imwrite(tmp_test.name, test_image)
            test_image_path = tmp_test.name
        
        # Handle reference image
        if reference_image is not None:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_ref:
                if isinstance(reference_image, Image.Image):
                    reference_image.save(tmp_ref.name)
                else:
                    cv2.imwrite(tmp_ref.name, reference_image)
                reference_image_path = tmp_ref.name
        else:
            # Use default reference image
            ref_path = 'reference/golden_product.jpg'
            if os.path.exists(ref_path):
                reference_image_path = ref_path
            else:
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
        
        # Add detailed test results section
        output_text += "<div style='background: white; padding: 28px; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>\n"
        output_text += "<h2 style='color: #1e40af; margin: 0 0 24px 0; border-bottom: 3px solid #dbeafe; padding-bottom: 12px; font-weight: 800; font-size: 1.8em;'>📊 Detailed Test Results</h2>\n"
        output_text += "<div style='display: grid; gap: 16px;'>\n"
        
        step_icons = {
            "1. Logo Detection": "🏷️",
            "2. Text & Serial Number OCR": "📝",
            "3. QR/DMC Code Detection": "🔲",
            "4. Surface Defect Detection": "🔍",
            "5. IC Geometry & Alignment": "📐",
            "6. Color & Texture Verification": "🎨",
            "7. Font Verification & Final Correlation": "✍️"
        }
        
        serial_number = 0
        for result in results:
            step = result.get('step', 'Unknown Step')
            status = result.get('status', 'UNKNOWN')
            conf = result.get('confidence', 0.0)
            
            if step == 'Final Verdict':
                continue
            
            serial_number += 1
            status_color = "#16a34a" if status == "PASS" else "#dc2626" if status == "FAIL" else "#f59e0b"
            status_bg = "#dcfce7" if status == "PASS" else "#fee2e2" if status == "FAIL" else "#fef3c7"
            icon = step_icons.get(step, "✓")
            proc_time = result.get('processing_time_ms', 0)
            
            output_text += f"""
        <div style="background: linear-gradient(135deg, #ffffff 0%, {status_bg}15 100%); padding: 20px; border-radius: 12px; border-left: 5px solid {status_color}; box-shadow: 0 2px 8px rgba(0,0,0,0.06); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: radial-gradient(circle, {status_color}10 0%, transparent 70%); border-radius: 0 0 0 100%;"></div>
            <div style="position: relative; z-index: 1;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div style="position: relative;">
                        <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, {status_color} 0%, {status_color}dd 100%); display: flex; align-items: center; justify-content: center; font-size: 1.5em; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                            {icon}
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
            </div>
        </div>
"""
        
        output_text += "</div>\n</div>\n"
        
        # Create JSON output
        json_output = json.dumps(report, indent=2, default=str)
        
        # Clean up temp files
        try:
            os.unlink(test_image_path)
            if reference_image is not None and isinstance(reference_image, Image.Image):
                os.unlink(reference_image_path)
        except:
            pass
        
        return output_text, json_output
        
    except Exception as e:
        error_msg = f"Error during verification: {str(e)}"
        return error_msg, json.dumps({'error': str(e)}, indent=2)


# ============================================================================
# GRADIO INTERFACE CONFIGURATION
# ============================================================================

custom_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter")
).set(
    body_background_fill="#f8fafc",
    panel_border_color="#e2e8f0",
    panel_background_fill="#ffffff",
    button_primary_background_fill="#3b82f6",
    button_primary_border_color="#3b82f6",
    button_primary_text_color="#ffffff"
)

# Custom CSS
css_style = """
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
}

.header-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 48px 32px;
    border-radius: 16px;
    margin-bottom: 32px;
    text-align: center;
    box-shadow: 0 12px 32px rgba(0,0,0,0.15);
}

.header-section h1 {
    margin: 0 0 16px 0;
    font-size: 3em;
    font-weight: 800;
    letter-spacing: -1px;
}

.header-section p {
    margin: 0;
    font-size: 1.1em;
    opacity: 0.9;
}

.tab-content {
    background: white;
    padding: 32px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.preprocessing-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 24px;
    margin: 24px 0;
}

.preprocessing-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: all 0.3s ease;
}

.preprocessing-card:hover {
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    border-color: #3b82f6;
    transform: translateY(-2px);
}

.preprocessing-card h3 {
    margin: 0 0 12px 0;
    color: #0f172a;
    font-size: 1.1em;
    font-weight: 700;
}

.preprocessing-card p {
    margin: 0;
    color: #64748b;
    font-size: 0.9em;
    line-height: 1.5;
}
"""

# Build Gradio interface with tabs
with gr.Blocks(theme=custom_theme, css=css_style, title="IC Counterfeit Detection") as demo:
    # Header
    gr.HTML("""
    <div class="header-section">
        <h1>🔬 IC Chip Counterfeit Detection System</h1>
        <p>Comprehensive 7-Step Verification with AI-Powered Image Preprocessing & Analysis</p>
    </div>
    """)
    
    with gr.Tabs():
        # ============================================================================
        # TAB 1: IC VERIFICATION (MAIN)
        # ============================================================================
        with gr.TabItem("✅ IC Verification", id="verification_tab"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 24px; border-radius: 12px; border: 2px solid #86efac; margin-bottom: 24px;">
                <h2 style="color: #0f172a; margin-top: 0;">7-Step IC Verification Pipeline</h2>
                <p style="color: #475569; margin-bottom: 0;">
                    Upload your test IC image and optionally a reference/golden IC image for comprehensive verification.
                    The system runs 7 automated optical inspection (AOI) tests to detect counterfeit products.
                </p>
            </div>
            """)
            
            with gr.Row():
                test_image = gr.Image(
                    label="Test IC Image",
                    type="pil",
                    image_mode="RGB"
                )
                reference_image = gr.Image(
                    label="Reference IC Image (Optional)",
                    type="pil",
                    image_mode="RGB"
                )
            
            verify_button = gr.Button(
                "🔍 Verify IC Chip",
                variant="primary",
                size="lg"
            )
            
            # Results
            gr.HTML("<hr style='margin: 24px 0;'>")
            
            results_output = gr.HTML(label="Verification Results")
            
            json_output = gr.Textbox(
                label="Detailed Report (JSON)",
                interactive=False,
                lines=20,
                max_lines=50
            )
            
            # Connect verification
            verify_button.click(
                fn=verify_ic,
                inputs=[test_image, reference_image],
                outputs=[results_output, json_output]
            )
        
        # ============================================================================
        # TAB 2: PREPROCESSING VISUALIZATION
        # ============================================================================
        with gr.TabItem("🖼️ Image Preprocessing", id="preprocessing_tab"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 24px; border-radius: 12px; border: 2px solid #93c5fd; margin-bottom: 24px;">
                <h2 style="color: #0f172a; margin-top: 0;">Image Preprocessing Pipeline</h2>
                <p style="color: #475569; margin-bottom: 0;">
                    This visualization shows how input images are preprocessed for optimal feature detection and OCR accuracy.
                    The pipeline includes resizing, noise removal, contrast enhancement, and edge detection.
                </p>
            </div>
            """)
            
            # Input
            test_image_input = gr.Image(
                label="Upload IC Chip Image",
                type="pil",
                image_mode="RGB"
            )
            
            preprocess_button = gr.Button(
                "🔄 Run Preprocessing",
                variant="primary",
                size="lg"
            )
            
            view_layers_button = gr.Button(
                "📊 View Saved Layer Visualizations",
                variant="secondary",
                size="lg"
            )
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1
            )
            
            # Outputs in two columns
            gr.HTML("<hr style='margin: 24px 0;'><h3 style='margin-top: 0;'>Original vs Preprocessed</h3>")
            
            with gr.Row():
                original_output = gr.Image(label="Original Image (Resized)", type="numpy")
                preprocessed_output = gr.Image(label="Preprocessed Image (CLAHE Enhanced)", type="numpy")
            
            gr.HTML("<hr style='margin: 24px 0;'><h3 style='margin-top: 0;'>Preprocessing Steps</h3>")
            
            with gr.Row():
                grayscale_output = gr.Image(label="Step 1: Grayscale", type="numpy")
                denoised_output = gr.Image(label="Step 2: Denoised (Bilateral Filter)", type="numpy")
            
            with gr.Row():
                contrast_output = gr.Image(label="Step 3: Contrast Enhanced (CLAHE)", type="numpy")
                threshold_output = gr.Image(label="Step 4: Adaptive Threshold", type="numpy")
            
            edges_output = gr.Image(label="Step 5: Canny Edge Detection", type="numpy")
            
            # Metadata
            gr.HTML("<hr style='margin: 24px 0;'><h3 style='margin-top: 0;'>Processing Details</h3>")
            metadata_output = gr.Textbox(
                label="Preprocessing Metadata",
                lines=15,
                interactive=False
            )
            
            # Load saved visualizations
            gr.HTML("<hr style='margin: 24px 0;'><h3 style='margin-top: 0;'>Saved Layer Visualizations</h3>")
            gr.HTML("""
            <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; border-left: 4px solid #22c55e; margin-bottom: 16px;">
                <p style="margin: 0; color: #166534; font-size: 0.9em;">
                    <strong>Layer Visualizations:</strong> The images below show saved visualization layers from the IC verification process, 
                    including logo detection, OCR, QR codes, defect detection, geometry analysis, and more.
                </p>
            </div>
            """)
            
            # Gallery of all saved layer visualizations
            layer_gallery = gr.Gallery(
                label="All Saved Layer Visualizations",
                show_label=True,
                columns=2,
                rows=6,
                object_fit="scale-down"
            )
            
            # Connect preprocessing visualization
            preprocess_button.click(
                fn=visualize_preprocessing,
                inputs=[test_image_input],
                outputs=[
                    original_output,
                    status_output,
                    preprocessed_output,
                    grayscale_output,
                    denoised_output,
                    contrast_output,
                    threshold_output,
                    edges_output,
                    metadata_output
                ]
            )
            
            # Load and display saved visualizations on tab load
            def load_visualizations():
                """Load and display all saved layer visualizations"""
                layer_images = load_layer_visualizations()
                images_list = [(img, filename) for filename, img in layer_images.items()]
                return images_list
            
            # Display saved visualizations
            def display_saved_layers():
                images_list = load_visualizations()
                return images_list
            
            # Load visualizations on tab interaction
            preprocess_button.click(
                fn=display_saved_layers,
                inputs=[],
                outputs=[layer_gallery]
            )
            
            # Also allow viewing saved layers without running preprocessing
            view_layers_button.click(
                fn=display_saved_layers,
                inputs=[],
                outputs=[layer_gallery]
            )
        
        # ============================================================================
        # TAB 3: DOCUMENTATION
        # ============================================================================
        with gr.TabItem("📚 Documentation", id="docs_tab"):
            gr.HTML("""
            <div style="max-width: 900px; margin: 0 auto;">
                <h2 style="color: #1e40af; margin-bottom: 24px; border-bottom: 3px solid #dbeafe; padding-bottom: 12px;">
                    7-Step IC Verification Pipeline Documentation
                </h2>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">🏷️ Step 1: Logo Detection (HoughCircles)</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Threshold:</strong> Confidence ≥ 0.3 for valid logos<br>
                        <strong>Method:</strong> Circular Hough Transform for fast logo detection<br>
                        <strong>Performance:</strong> < 50ms
                    </p>
                    <p style="color: #64748b; margin: 0;">Detects manufacturer logos on IC chips using fast circle detection. 
                    Compares circular shape and position with reference product.</p>
                </div>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #10b981; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">📝 Step 2: Text & Serial Number OCR</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Threshold:</strong> OCR Confidence ≥ 80%<br>
                        <strong>Method:</strong> Tesseract OCR with fast settings<br>
                        <strong>Performance:</strong> 100-300ms
                    </p>
                    <p style="color: #64748b; margin: 0;">Extracts text and serial numbers using Tesseract OCR. 
                    Compares extracted text with expected values.</p>
                </div>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #f59e0b; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">🔲 Step 3: QR/DMC Code Detection</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Threshold:</strong> ≥ 80% success rate<br>
                        <strong>Method:</strong> OpenCV QRCodeDetector + pyzbar fallback<br>
                        <strong>Performance:</strong> 50-150ms
                    </p>
                    <p style="color: #64748b; margin: 0;">Detects and decodes QR codes and Data Matrix codes. 
                    Verifies code authenticity and data content.</p>
                </div>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #ec4899; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">🔍 Step 4: Surface Defect Detection</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Threshold:</strong> SSIM ≥ 0.8, Intensity difference ≤ 20<br>
                        <strong>Method:</strong> Structural Similarity (SSIM) + Intensity analysis<br>
                        <strong>Performance:</strong> < 100ms
                    </p>
                    <p style="color: #64748b; margin: 0;">Compares test IC surface with reference to detect scratches, 
                    burn marks, discoloration, or other defects.</p>
                </div>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #8b5cf6; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">📐 Step 5: IC Geometry & Alignment</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Thresholds:</strong> Size ±5%, Aspect ±5%, Angle ±2°<br>
                        <strong>Method:</strong> Edge detection + Contour analysis<br>
                        <strong>Performance:</strong> 50-100ms (Combined: Edge + Geometry + Angle)
                    </p>
                    <p style="color: #64748b; margin: 0;">Analyzes IC chip outline, size, aspect ratio, and rotation angle. 
                    Ensures physical dimensions match genuine product.</p>
                </div>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #06b6d4; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">🎨 Step 6: Color & Texture Verification</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Thresholds:</strong> Color ΔE < 3-5 (LAB), Texture distance < 0.15<br>
                        <strong>Method:</strong> LAB color space + Texture feature matching<br>
                        <strong>Performance:</strong> 80-150ms (Combined: Color + Texture)
                    </p>
                    <p style="color: #64748b; margin: 0;">Compares color consistency and surface texture patterns. 
                    Detects color fading or texture abnormalities.</p>
                </div>
                
                <div style="background: white; padding: 24px; border-radius: 12px; border-left: 4px solid #f97316; margin-bottom: 20px;">
                    <h3 style="margin-top: 0; color: #0f172a;">✍️ Step 7: Font Verification & Correlation</h3>
                    <p style="margin: 0 0 12px 0; color: #475569;">
                        <strong>Thresholds:</strong> Font similarity ≥ 75%, Final correlation ≥ 75%<br>
                        <strong>Method:</strong> Stroke analysis + Template matching<br>
                        <strong>Performance:</strong> 150-300ms
                    </p>
                    <p style="color: #64748b; margin: 0;">Verifies font consistency of text printing and generates final 
                    correlation score based on all 7 tests.</p>
                </div>
                
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 24px; border-radius: 12px; border-left: 4px solid #f59e0b; margin-top: 32px;">
                    <h3 style="margin-top: 0; color: #0f172a;">⚡ Performance Summary</h3>
                    <p style="color: #64748b; margin: 0;">
                        <strong>Total Processing Time:</strong> < 1.5 seconds<br>
                        <strong>Total Tests:</strong> 7 comprehensive AOI tests<br>
                        <strong>Verdict Logic:</strong> GENUINE if confidence ≥ 75% AND pass-rate ≥ 85% AND failures ≤ 1
                    </p>
                </div>
            </div>
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        theme=custom_theme,
        css=css_style,
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
