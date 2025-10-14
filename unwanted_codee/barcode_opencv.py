"""
Barcode and QR Code Reader using only OpenCV
Works without pyzbar - uses OpenCV's built-in barcode detector
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class BarcodeReaderOpenCV:
    """
    Barcode reader using OpenCV's QRCodeDetector and BarcodeDetector.
    No external dependencies beyond OpenCV.
    """
    
    def __init__(self, output_dir: str = "barcode_results"):
        """
        Initialize the BarcodeReader.
        
        Args:
            output_dir: Directory to save results and annotated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scan_history = []
        
        # Initialize detectors
        self.qr_detector = cv2.QRCodeDetector()
        
        # Try to initialize barcode detector (available in OpenCV 4.8+)
        try:
            self.barcode_detector = cv2.barcode.BarcodeDetector()
            self.has_barcode_detector = True
        except AttributeError:
            print("Note: Advanced barcode detection not available. Only QR codes will be detected.")
            print("Upgrade to OpenCV 4.8+ for full barcode support.")
            self.has_barcode_detector = False
    
    def read_barcode_from_image(self, image_path: str, save_result: bool = True) -> List[Dict]:
        """
        Read barcodes from a static image file.
        
        Args:
            image_path: Path to the image file
            save_result: Whether to save annotated image and results
            
        Returns:
            List of dictionaries containing barcode information
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return []
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect and decode barcodes
        results = self._detect_and_decode(image)
        
        # Draw bounding boxes and labels
        annotated_image = self._draw_barcodes(image.copy(), results)
        
        # Save results if requested
        if save_result and results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"barcode_result_{timestamp}.jpg"
            cv2.imwrite(str(output_path), annotated_image)
            
            # Save JSON report
            self._save_json_report(results, timestamp)
            print(f"\nResults saved to {output_path}")
        
        # Display results
        self._display_results(results)
        
        # Show image
        self._show_image(annotated_image, "Barcode Detection Result")
        
        return results
    
    def read_barcode_from_webcam(self, camera_index: int = 0, save_scans: bool = True):
        """
        Read barcodes from webcam in real-time.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
            save_scans: Whether to save detected barcodes
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Barcode Scanner Active - Press 'q' to quit, 's' to save current frame")
        scanned_codes = set()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Detect and decode barcodes
            results = self._detect_and_decode(frame)
            
            # Draw bounding boxes and labels
            annotated_frame = self._draw_barcodes(frame.copy(), results)
            
            # Display barcode info on frame
            for i, barcode_info in enumerate(results):
                text = f"{barcode_info['type']}: {barcode_info['data']}"
                cv2.putText(annotated_frame, text, (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add to scanned codes and save if new
                code_id = f"{barcode_info['type']}_{barcode_info['data']}"
                if code_id not in scanned_codes:
                    scanned_codes.add(code_id)
                    if save_scans:
                        self.scan_history.append({
                            **barcode_info,
                            'timestamp': datetime.now().isoformat()
                        })
                    print(f"\n[NEW SCAN] {text}")
            
            # Display frame
            cv2.imshow('Barcode Scanner', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"webcam_scan_{timestamp}.jpg"
                cv2.imwrite(str(save_path), annotated_frame)
                print(f"Frame saved to {save_path}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save scan history
        if save_scans and self.scan_history:
            self._save_scan_history()
    
    def _detect_and_decode(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and decode barcodes in an image using OpenCV.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing barcode information
        """
        results = []
        
        # Detect QR codes
        qr_results = self._detect_qr_codes(image)
        results.extend(qr_results)
        
        # Detect other barcodes if detector is available
        if self.has_barcode_detector:
            barcode_results = self._detect_barcodes(image)
            results.extend(barcode_results)
        
        return results
    
    def _detect_qr_codes(self, image: np.ndarray) -> List[Dict]:
        """Detect QR codes using OpenCV's QRCodeDetector."""
        results = []
        
        # Try multi-detection first (OpenCV 4.5+)
        try:
            retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(image)
            
            if retval:
                for i, data in enumerate(decoded_info):
                    if data:  # Only add if data was successfully decoded
                        pts = points[i].astype(int)
                        x, y, w, h = cv2.boundingRect(pts)
                        
                        results.append({
                            'data': data,
                            'type': 'QRCODE',
                            'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                            'polygon': [(int(pt[0]), int(pt[1])) for pt in pts]
                        })
        except:
            # Fallback to single detection
            data, points, _ = self.qr_detector.detectAndDecode(image)
            
            if data and points is not None:
                pts = points.astype(int)
                x, y, w, h = cv2.boundingRect(pts)
                
                results.append({
                    'data': data,
                    'type': 'QRCODE',
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'polygon': [(int(pt[0]), int(pt[1])) for pt in pts[0]]
                })
        
        return results
    
    def _detect_barcodes(self, image: np.ndarray) -> List[Dict]:
        """Detect 1D/2D barcodes using OpenCV's BarcodeDetector."""
        results = []
        
        try:
            retval, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecode(image)
            
            if retval:
                for i, data in enumerate(decoded_info):
                    if data:  # Only add if data was successfully decoded
                        barcode_type = decoded_type[i] if i < len(decoded_type) else "UNKNOWN"
                        pts = points[i].astype(int) if points is not None and i < len(points) else None
                        
                        if pts is not None:
                            x, y, w, h = cv2.boundingRect(pts)
                            
                            results.append({
                                'data': data,
                                'type': barcode_type,
                                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                'polygon': [(int(pt[0]), int(pt[1])) for pt in pts]
                            })
        except Exception as e:
            print(f"Barcode detection error: {e}")
        
        return results
    
    def _draw_barcodes(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on detected barcodes.
        
        Args:
            image: Input image
            results: List of barcode detection results
            
        Returns:
            Annotated image
        """
        for result in results:
            # Draw polygon boundary
            points = result.get('polygon', [])
            if len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 3)
            
            # Draw rectangle
            bbox = result['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add label background
            label = f"{result['type']}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
            
            # Add label text
            cv2.putText(image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add data text below barcode
            data_text = result['data'][:50]  # Truncate if too long
            cv2.putText(image, data_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return image
    
    def _display_results(self, results: List[Dict]):
        """Display barcode results in console."""
        if not results:
            print("\n" + "="*60)
            print("No barcodes detected")
            print("="*60)
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(results)} barcode(s):")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            print(f"\nBarcode #{i}:")
            print(f"  Type: {result['type']}")
            print(f"  Data: {result['data']}")
            print(f"  Position: ({result['bbox']['x']}, {result['bbox']['y']})")
            print(f"  Size: {result['bbox']['width']}x{result['bbox']['height']}")
        
        print(f"{'='*60}\n")
    
    def _show_image(self, image: np.ndarray, window_name: str = "Image"):
        """Display image in a window."""
        # Resize if image is too large
        max_height = 800
        height, width = image.shape[:2]
        
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, max_height))
        
        cv2.imshow(window_name, image)
        print("\nPress any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _save_json_report(self, results: List[Dict], timestamp: str):
        """Save barcode results to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_barcodes': len(results),
            'barcodes': results
        }
        
        json_path = self.output_dir / f"barcode_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report saved to {json_path}")
    
    def _save_scan_history(self):
        """Save webcam scan history to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = self.output_dir / f"scan_history_{timestamp}.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.scan_history, f, indent=2)
        
        print(f"\nScan history saved to {history_path}")


def main():
    """
    Main function demonstrating different usage modes.
    """
    import sys
    
    reader = BarcodeReaderOpenCV()
    
    print("Barcode Reader (OpenCV) - Choose an option:")
    print("1. Read from image file")
    print("2. Read from webcam (real-time)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == '1':
        # Read from image
        if len(sys.argv) > 2:
            image_path = sys.argv[2]
        else:
            image_path = input("Enter image path: ").strip()
        
        results = reader.read_barcode_from_image(image_path)
        
    elif choice == '2':
        # Read from webcam
        reader.read_barcode_from_webcam()
    
    else:
        print("Invalid choice!")
        return


if __name__ == "__main__":
    main()
