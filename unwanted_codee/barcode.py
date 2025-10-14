"""
Barcode Reader using OpenCV and pyzbar
Supports reading various barcode formats including QR codes, EAN, UPC, Code128, etc.
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class BarcodeReader:
    """
    A comprehensive barcode reader that can detect and decode barcodes from images,
    video streams, or webcam feeds using OpenCV and pyzbar.
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
            print(f"Results saved to {output_path}")
        
        # Display results
        self._display_results(results)
        
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
        scanned_codes = set()  # Track already scanned codes to avoid duplicates
        
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
    
    def read_barcode_from_video(self, video_path: str, save_results: bool = True) -> List[Dict]:
        """
        Read barcodes from a video file.
        
        Args:
            video_path: Path to the video file
            save_results: Whether to save detected barcodes
            
        Returns:
            List of all detected barcodes with timestamps
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video from {video_path}")
            return []
        
        all_results = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps if fps > 0 else frame_count
            
            # Detect barcodes
            results = self._detect_and_decode(frame)
            
            # Add timestamp to results
            for result in results:
                result['video_timestamp'] = f"{timestamp:.2f}s"
                result['frame_number'] = frame_count
                all_results.append(result)
        
        cap.release()
        
        # Remove duplicates and save
        unique_results = self._remove_duplicate_scans(all_results)
        
        if save_results and unique_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_json_report(unique_results, f"video_{timestamp}")
        
        print(f"\nProcessed {frame_count} frames, found {len(unique_results)} unique barcodes")
        self._display_results(unique_results)
        
        return unique_results
    
    def _detect_and_decode(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and decode barcodes in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing barcode information
        """
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve detection
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect barcodes
        barcodes = pyzbar.decode(enhanced)
        
        # If no barcodes found, try with original image
        if not barcodes:
            barcodes = pyzbar.decode(image)
        
        results = []
        for barcode in barcodes:
            # Extract barcode data
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type
            
            # Get bounding box coordinates
            (x, y, w, h) = barcode.rect
            
            # Get polygon points for more accurate boundary
            points = barcode.polygon
            
            result = {
                'data': barcode_data,
                'type': barcode_type,
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                'polygon': [(point.x, point.y) for point in points],
                'quality': barcode.quality if hasattr(barcode, 'quality') else None
            }
            
            results.append(result)
        
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
            points = result['polygon']
            if len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 3)
            
            # Draw rectangle
            bbox = result['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add label
            label = f"{result['type']}"
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add data text below barcode
            data_text = result['data'][:30]  # Truncate if too long
            cv2.putText(image, data_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return image
    
    def _display_results(self, results: List[Dict]):
        """Display barcode results in console."""
        if not results:
            print("No barcodes detected")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(results)} barcode(s):")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            print(f"\nBarcode #{i}:")
            print(f"  Type: {result['type']}")
            print(f"  Data: {result['data']}")
            print(f"  Position: ({result['bbox']['x']}, {result['bbox']['y']})")
            if result.get('quality'):
                print(f"  Quality: {result['quality']}")
            if result.get('video_timestamp'):
                print(f"  Video Time: {result['video_timestamp']}")
        
        print(f"{'='*60}\n")
    
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
    
    def _remove_duplicate_scans(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate barcode scans from results."""
        seen = set()
        unique_results = []
        
        for result in results:
            code_id = f"{result['type']}_{result['data']}"
            if code_id not in seen:
                seen.add(code_id)
                unique_results.append(result)
        
        return unique_results


def main():
    """
    Main function demonstrating different usage modes.
    """
    import sys
    
    reader = BarcodeReader()
    
    print("Barcode Reader - Choose an option:")
    print("1. Read from image file")
    print("2. Read from webcam (real-time)")
    print("3. Read from video file")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1/2/3): ").strip()
    
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
        
    elif choice == '3':
        # Read from video
        if len(sys.argv) > 2:
            video_path = sys.argv[2]
        else:
            video_path = input("Enter video path: ").strip()
        
        results = reader.read_barcode_from_video(video_path)
    
    else:
        print("Invalid choice!")
        return


if __name__ == "__main__":
    main()
