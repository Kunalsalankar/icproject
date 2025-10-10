"""
Counterfeit Detection AI Agent
Complete pipeline for product authentication and counterfeit detection
"""

import cv2
import numpy as np
import pytesseract
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import os

# Try to import pyzbar, but make it optional
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except (ImportError, FileNotFoundError) as e:
    PYZBAR_AVAILABLE = False
    print(f"⚠️  Warning: pyzbar not available ({str(e)})")
    print("   QR/Barcode detection will use OpenCV's QRCodeDetector instead")


@dataclass
class DetectionResult:
    """Data class to store detection results"""
    step: str
    status: str
    confidence: float
    details: Dict
    timestamp: str


class CounterfeitDetectionAgent:
    """
    AI Agent for comprehensive counterfeit detection
    Implements 7-step pipeline for product authentication
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the counterfeit detection agent
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.results = []
        self.config = self._load_config(config_path)
        self.final_verdict = None
        
        # Thresholds
        self.logo_match_threshold = 0.7
        self.ssim_threshold = 0.85
        self.color_tolerance = 15
        self.angle_tolerance = 5.0
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def process_image(self, image_path: str, reference_data: Dict) -> Dict:
        """
        Main processing pipeline for counterfeit detection
        
        Args:
            image_path: Path to the product image to verify
            reference_data: Dictionary containing reference/golden data
                - logo_path: Path to genuine logo template
                - expected_text: Expected serial text
                - expected_qr_data: Expected QR/DMC data
                - golden_image_path: Path to golden IC image
                - color_reference: Reference color values
                
        Returns:
            Complete detection report with verdict
        """
        print(f"\n{'='*60}")
        print(f"COUNTERFEIT DETECTION AGENT - STARTING ANALYSIS")
        print(f"{'='*60}\n")
        
        # Load test image
        test_image = cv2.imread(image_path)
        if test_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 1: Logo Detection
        logo_result = self.detect_logo(test_image, reference_data.get('logo_path'))
        self.results.append(logo_result)
        
        # Step 2: Text Area + OCR
        text_result = self.detect_and_read_text(test_image, reference_data.get('expected_text'))
        self.results.append(text_result)
        
        # Step 3: QR/DMC Detection
        qr_result = self.detect_qr_code(test_image, reference_data.get('expected_qr_data'))
        self.results.append(qr_result)
        
        # Step 4: Defect Detection
        defect_result = self.detect_defects(test_image, reference_data.get('golden_image_path'))
        self.results.append(defect_result)
        
        # Step 5: Angle/Edge Alignment
        angle_result = self.check_alignment(test_image)
        self.results.append(angle_result)
        
        # Step 6: Color Calibration
        color_result = self.verify_color(test_image, reference_data.get('color_reference'))
        self.results.append(color_result)
        
        # Step 7: Counterfeit Check (Final Verdict)
        final_result = self.generate_verdict()
        self.results.append(final_result)
        
        # Generate report
        report = self.generate_report()
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}\n")
        
        return report
    
    def detect_logo(self, image: np.ndarray, logo_template_path: Optional[str]) -> DetectionResult:
        """
        Step 1: Logo Detection using template matching and feature matching
        """
        print("Step 1: Logo Detection - Starting...")
        
        if logo_template_path is None or not os.path.exists(logo_template_path):
            return DetectionResult(
                step="1. Logo Detection",
                status="SKIPPED",
                confidence=0.0,
                details={"message": "No logo template provided"},
                timestamp=datetime.now().isoformat()
            )
        
        template = cv2.imread(logo_template_path)
        if template is None:
            return DetectionResult(
                step="1. Logo Detection",
                status="ERROR",
                confidence=0.0,
                details={"message": "Could not load logo template"},
                timestamp=datetime.now().isoformat()
            )
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Template Matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Method 2: ORB Feature Matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray_template, None)
        kp2, des2 = orb.detectAndCompute(gray_image, None)
        
        feature_match_score = 0.0
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate match score
            good_matches = [m for m in matches if m.distance < 50]
            feature_match_score = len(good_matches) / max(len(kp1), 1)
        
        # Combine scores
        combined_score = (max_val + min(feature_match_score, 1.0)) / 2
        
        status = "PASS" if combined_score >= self.logo_match_threshold else "FAIL"
        
        print(f"  ✓ Template Match Score: {max_val:.3f}")
        print(f"  ✓ Feature Match Score: {feature_match_score:.3f}")
        print(f"  ✓ Combined Score: {combined_score:.3f}")
        print(f"  ✓ Status: {status}\n")
        
        return DetectionResult(
            step="1. Logo Detection",
            status=status,
            confidence=combined_score,
            details={
                "template_match_score": float(max_val),
                "feature_match_score": float(feature_match_score),
                "location": max_loc,
                "method": "matchTemplate + ORB + BFMatcher"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def detect_and_read_text(self, image: np.ndarray, expected_text: Optional[str]) -> DetectionResult:
        """
        Step 2: Text Area Detection + OCR using Tesseract
        """
        print("Step 2: Text Area + OCR - Starting...")
        
        # Preprocess for OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours for text areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # OCR on the image
        try:
            ocr_text = pytesseract.image_to_string(image, config='--psm 6')
            ocr_text_cleaned = ocr_text.strip()
        except Exception as e:
            return DetectionResult(
                step="2. Text Area + OCR",
                status="ERROR",
                confidence=0.0,
                details={"message": f"OCR failed: {str(e)}"},
                timestamp=datetime.now().isoformat()
            )
        
        # Verify text if expected text is provided
        confidence = 0.0
        status = "PASS"
        
        if expected_text:
            # Calculate similarity
            expected_clean = expected_text.strip().lower()
            ocr_clean = ocr_text_cleaned.lower()
            
            # Simple character-based similarity
            if expected_clean in ocr_clean or ocr_clean in expected_clean:
                confidence = 1.0
            else:
                # Calculate Levenshtein-like similarity
                matches = sum(1 for a, b in zip(expected_clean, ocr_clean) if a == b)
                confidence = matches / max(len(expected_clean), len(ocr_clean), 1)
            
            status = "PASS" if confidence >= 0.7 else "FAIL"
        else:
            confidence = 1.0 if len(ocr_text_cleaned) > 0 else 0.0
            status = "PASS" if len(ocr_text_cleaned) > 0 else "FAIL"
        
        print(f"  ✓ Text Areas Found: {len(contours)}")
        print(f"  ✓ OCR Text: '{ocr_text_cleaned[:50]}...'")
        print(f"  ✓ Confidence: {confidence:.3f}")
        print(f"  ✓ Status: {status}\n")
        
        return DetectionResult(
            step="2. Text Area + OCR",
            status=status,
            confidence=confidence,
            details={
                "text_areas_found": len(contours),
                "ocr_text": ocr_text_cleaned,
                "expected_text": expected_text,
                "method": "threshold + findContours + pytesseract"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def detect_qr_code(self, image: np.ndarray, expected_data: Optional[str]) -> DetectionResult:
        """
        Step 3: QR/DMC Detection and Decoding
        """
        print("Step 3: QR/DMC Detection - Starting...")
        
        detected_codes = []
        
        # Try pyzbar first (supports multiple barcode types)
        if PYZBAR_AVAILABLE:
            try:
                decoded_objects = pyzbar.decode(image)
                for obj in decoded_objects:
                    code_data = obj.data.decode('utf-8')
                    detected_codes.append({
                        "type": obj.type,
                        "data": code_data,
                        "rect": obj.rect
                    })
            except Exception as e:
                print(f"  ⚠️  pyzbar decode failed: {str(e)}")
        
        # Fallback to OpenCV QRCodeDetector
        if not detected_codes:
            qr_detector = cv2.QRCodeDetector()
            data, bbox, _ = qr_detector.detectAndDecode(image)
            
            if data:
                detected_codes.append({
                    "type": "QRCODE",
                    "data": data,
                    "rect": bbox
                })
        
        if not detected_codes:
            return DetectionResult(
                step="3. QR/DMC Detection",
                status="SKIPPED",
                confidence=1.0,
                details={"message": "No QR/DMC code detected (skipping check)"},
                timestamp=datetime.now().isoformat()
            )
        
        # Verify against expected data
        confidence = 0.0
        status = "FAIL"
        
        if expected_data:
            for code in detected_codes:
                if code["data"] == expected_data:
                    confidence = 1.0
                    status = "PASS"
                    break
                elif expected_data in code["data"] or code["data"] in expected_data:
                    confidence = 0.8
                    status = "PASS"
                    break
        else:
            # If no expected data, just check if we found codes
            confidence = 1.0
            status = "PASS"
        
        print(f"  ✓ Codes Detected: {len(detected_codes)}")
        for code in detected_codes:
            print(f"    - Type: {code['type']}, Data: {code['data'][:50]}...")
        print(f"  ✓ Confidence: {confidence:.3f}")
        print(f"  ✓ Status: {status}\n")
        
        method = "pyzbar" if PYZBAR_AVAILABLE else "OpenCV QRCodeDetector"
        
        return DetectionResult(
            step="3. QR/DMC Detection",
            status=status,
            confidence=confidence,
            details={
                "codes_detected": len(detected_codes),
                "detected_codes": detected_codes,
                "expected_data": expected_data,
                "method": method
            },
            timestamp=datetime.now().isoformat()
        )
    
    def detect_defects(self, image: np.ndarray, golden_image_path: Optional[str]) -> DetectionResult:
        """
        Step 4: Defect Detection by comparing with golden IC
        """
        print("Step 4: Defect Detection - Starting...")
        
        if golden_image_path is None or not os.path.exists(golden_image_path):
            return DetectionResult(
                step="4. Defect Detection",
                status="SKIPPED",
                confidence=1.0,
                details={"message": "No golden image provided"},
                timestamp=datetime.now().isoformat()
            )
        
        golden_image = cv2.imread(golden_image_path)
        if golden_image is None:
            return DetectionResult(
                step="4. Defect Detection",
                status="ERROR",
                confidence=0.0,
                details={"message": "Could not load golden image"},
                timestamp=datetime.now().isoformat()
            )
        
        # Resize images to same size for comparison
        h, w = golden_image.shape[:2]
        test_resized = cv2.resize(image, (w, h))
        
        # Convert to grayscale
        gray_golden = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM (Structural Similarity Index)
        ssim_score, diff_image = ssim(gray_golden, gray_test, full=True)
        diff_image = (diff_image * 255).astype("uint8")
        
        # Calculate absolute difference
        abs_diff = cv2.absdiff(gray_golden, gray_test)
        _, thresh_diff = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Count defect pixels
        defect_pixels = np.sum(thresh_diff > 0)
        total_pixels = thresh_diff.size
        defect_ratio = defect_pixels / total_pixels
        
        # Combined confidence
        confidence = ssim_score * (1 - defect_ratio)
        status = "PASS" if confidence >= self.ssim_threshold else "FAIL"
        
        print(f"  ✓ SSIM Score: {ssim_score:.3f}")
        print(f"  ✓ Defect Ratio: {defect_ratio:.3f}")
        print(f"  ✓ Combined Confidence: {confidence:.3f}")
        print(f"  ✓ Status: {status}\n")
        
        return DetectionResult(
            step="4. Defect Detection",
            status=status,
            confidence=confidence,
            details={
                "ssim_score": float(ssim_score),
                "defect_ratio": float(defect_ratio),
                "defect_pixels": int(defect_pixels),
                "method": "absdiff + SSIM + threshold"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def check_alignment(self, image: np.ndarray) -> DetectionResult:
        """
        Step 5: Angle/Edge Alignment Check
        """
        print("Step 5: Angle/Edge Alignment - Starting...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return DetectionResult(
                step="5. Angle/Edge Alignment",
                status="FAIL",
                confidence=0.0,
                details={"message": "No lines detected"},
                timestamp=datetime.now().isoformat()
            )
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            angles.append(angle)
        
        # Check alignment (lines should be close to 0, 90, 180, 270 degrees)
        aligned_angles = []
        for angle in angles:
            # Normalize to 0-90 range
            norm_angle = angle % 90
            if norm_angle > 45:
                norm_angle = 90 - norm_angle
            aligned_angles.append(norm_angle)
        
        # Calculate average deviation from perfect alignment
        avg_deviation = np.mean(aligned_angles)
        
        # Use goodFeaturesToTrack for corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        confidence = 1.0 - (avg_deviation / 45.0)  # Normalize to 0-1
        status = "PASS" if avg_deviation <= self.angle_tolerance else "FAIL"
        
        print(f"  ✓ Lines Detected: {len(lines)}")
        print(f"  ✓ Corners Detected: {len(corners) if corners is not None else 0}")
        print(f"  ✓ Average Angle Deviation: {avg_deviation:.2f}°")
        print(f"  ✓ Confidence: {confidence:.3f}")
        print(f"  ✓ Status: {status}\n")
        
        return DetectionResult(
            step="5. Angle/Edge Alignment",
            status=status,
            confidence=confidence,
            details={
                "lines_detected": len(lines),
                "corners_detected": len(corners) if corners is not None else 0,
                "avg_angle_deviation": float(avg_deviation),
                "method": "Canny + HoughLinesP + goodFeaturesToTrack"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def verify_color(self, image: np.ndarray, color_reference: Optional[Dict]) -> DetectionResult:
        """
        Step 6: Color Calibration and Verification
        """
        print("Step 6: Color Calibration - Starting...")
        
        # Convert to LAB color space for better color comparison
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color histogram
        hist_l = cv2.calcHist([lab_image], [0], None, [256], [0, 256])
        hist_a = cv2.calcHist([lab_image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([lab_image], [2], None, [256], [0, 256])
        
        # Calculate average color values
        avg_color_bgr = cv2.mean(image)[:3]
        avg_color_lab = cv2.mean(lab_image)[:3]
        
        confidence = 1.0
        status = "PASS"
        color_diff = 0.0
        
        if color_reference:
            # Compare with reference colors
            ref_bgr = color_reference.get('bgr', [0, 0, 0])
            
            # Calculate color difference
            color_diff = np.sqrt(sum((a - b)**2 for a, b in zip(avg_color_bgr, ref_bgr)))
            
            # Normalize confidence
            max_diff = np.sqrt(3 * 255**2)
            confidence = 1.0 - (color_diff / max_diff)
            
            # Compare histograms if reference histogram is provided
            if 'histogram' in color_reference:
                ref_hist = np.array(color_reference['histogram'])
                test_hist = hist_l.flatten()
                
                # Normalize histograms
                ref_hist = ref_hist / (np.sum(ref_hist) + 1e-7)
                test_hist = test_hist / (np.sum(test_hist) + 1e-7)
                
                # Calculate histogram correlation
                hist_corr = cv2.compareHist(
                    ref_hist.astype(np.float32), 
                    test_hist.astype(np.float32), 
                    cv2.HISTCMP_CORREL
                )
                confidence = (confidence + hist_corr) / 2
            
            status = "PASS" if color_diff <= self.color_tolerance else "FAIL"
        
        print(f"  ✓ Average Color (BGR): {[int(c) for c in avg_color_bgr]}")
        print(f"  ✓ Average Color (LAB): {[int(c) for c in avg_color_lab]}")
        print(f"  ✓ Color Difference: {color_diff:.2f}")
        print(f"  ✓ Confidence: {confidence:.3f}")
        print(f"  ✓ Status: {status}\n")
        
        return DetectionResult(
            step="6. Color Calibration",
            status=status,
            confidence=confidence,
            details={
                "avg_color_bgr": [int(c) for c in avg_color_bgr],
                "avg_color_lab": [int(c) for c in avg_color_lab],
                "color_difference": float(color_diff),
                "method": "LAB hist + compareHist"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def generate_verdict(self) -> DetectionResult:
        """
        Step 7: Generate final counterfeit verdict based on all checks
        """
        print("Step 7: Counterfeit Check - Generating Final Verdict...")
        
        # Calculate overall score
        total_confidence = 0.0
        passed_checks = 0
        failed_checks = 0
        total_checks = 0
        
        for result in self.results:
            if result.status not in ["SKIPPED", "ERROR"]:
                total_confidence += result.confidence
                total_checks += 1
                if result.status == "PASS":
                    passed_checks += 1
                else:
                    failed_checks += 1
        
        avg_confidence = total_confidence / max(total_checks, 1)
        
        # Determine final verdict
        # Product is genuine if:
        # 1. Average confidence >= 0.75
        # 2. No more than 1 failed check
        # 3. Critical checks (logo, QR, defect) must pass
        
        critical_checks = ["1. Logo Detection", "3. QR/DMC Detection", "4. Defect Detection"]
        critical_failed = any(
            r.status == "FAIL" and r.step in critical_checks 
            for r in self.results
        )
        
        if avg_confidence >= 0.75 and failed_checks <= 1 and not critical_failed:
            verdict = "GENUINE"
            status = "PASS"
        else:
            verdict = "COUNTERFEIT"
            status = "FAIL"
        
        self.final_verdict = verdict
        
        print(f"\n{'='*60}")
        print(f"  FINAL VERDICT: {verdict}")
        print(f"  Overall Confidence: {avg_confidence:.3f}")
        print(f"  Checks Passed: {passed_checks}/{total_checks}")
        print(f"  Checks Failed: {failed_checks}/{total_checks}")
        print(f"{'='*60}\n")
        
        return DetectionResult(
            step="7. Counterfeit Check",
            status=status,
            confidence=avg_confidence,
            details={
                "verdict": verdict,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "total_checks": total_checks,
                "critical_failed": critical_failed,
                "method": "SSIM + absdiff"
            },
            timestamp=datetime.now().isoformat()
        )
    
    def generate_report(self) -> Dict:
        """Generate comprehensive detection report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "verdict": self.final_verdict,
            "pipeline_results": [asdict(r) for r in self.results],
            "summary": {
                "total_checks": len([r for r in self.results if r.status not in ["SKIPPED", "ERROR"]]),
                "passed": len([r for r in self.results if r.status == "PASS"]),
                "failed": len([r for r in self.results if r.status == "FAIL"]),
                "skipped": len([r for r in self.results if r.status == "SKIPPED"]),
                "errors": len([r for r in self.results if r.status == "ERROR"])
            }
        }
        return report
    
    def save_report(self, report: Dict, output_path: str):
        """Save detection report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_path}")


def main():
    """
    Example usage of the Counterfeit Detection Agent
    """
    # Initialize agent
    agent = CounterfeitDetectionAgent()
    
    # Prepare reference data
    reference_data = {
        'logo_path': 'reference/genuine_logo.png',  # Path to genuine logo
        'expected_text': 'SN123456789',  # Expected serial number
        'expected_qr_data': 'GENUINE-PRODUCT-CODE-12345',  # Expected QR data
        'golden_image_path': 'reference/golden_product.png',  # Golden reference image
        'color_reference': {
            'bgr': [180, 180, 180],  # Expected average BGR color
        }
    }
    
    # Process test image
    test_image_path = 'test_images/product_to_verify.jpg'
    
    try:
        report = agent.process_image(test_image_path, reference_data)
        
        # Save report
        agent.save_report(report, 'detection_report.json')
        
        # Print summary
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        print(f"Verdict: {report['verdict']}")
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Skipped: {report['summary']['skipped']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
