"""
Traffic Sign Recognition System

Main entry point for the Traffic Sign Recognition project.
Detects and classifies blue circular traffic signs (Straight, Turn Left, Turn Right, Parking).

This system uses:
- HSV color space for blue color detection
- Contour analysis with circularity filtering for sign localization
- Template Matching for sign classification
"""

import cv2
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from core.detector import SignDetector
from core.classifier import SignClassifier


class TrafficSignRecognition:
    """
    Main class for Traffic Sign Recognition pipeline.
    
    Orchestrates detection and classification of traffic signs.
    """
    
    def __init__(self, 
                 templates_dir: str = 'templates',
                 confidence_threshold: float = 0.7):
        """
        Initialize the Traffic Sign Recognition system.
        
        Args:
            templates_dir (str): Path to templates directory
            confidence_threshold (float): Confidence threshold for classifications
        """
        self.detector = SignDetector()
        self.classifier = SignClassifier(
            templates_dir=templates_dir,
            template_matching_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
    
    def process_image(self, image_path: str, 
                     visualize: bool = True,
                     output_path: Optional[str] = None) -> Tuple[List[Tuple[str, float]], cv2.Mat]:
        """
        Process a single image to detect and classify traffic signs.
        
        Args:
            image_path (str): Path to input image
            visualize (bool): Whether to draw detections on the image
            output_path (Optional[str]): Path to save output image (if provided)
            
        Returns:
            Tuple containing:
                - List of classifications (sign_type, confidence)
                - Annotated image with detections
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Detect signs
        detected_signs = self.detector.detect_signs(image)
        
        if len(detected_signs) == 0:
            print(f"No signs detected in {image_path}")
            return [], image
        
        print(f"Detected {len(detected_signs)} signs in {image_path}")
        
        # Classify each detected sign
        results = []
        annotated_image = image.copy()
        
        for idx, (roi, bbox) in enumerate(detected_signs):
            # Classify the ROI with verbose output
            sign_type, confidence, all_scores = self.classifier.classify_verbose(roi)
            results.append((sign_type, confidence))
            
            print(f"\n  Sign {idx + 1}:")
            print(f"    Best Match: {sign_type.upper()}")
            print(f"    Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"    All Scores:")
            for stype in sorted(all_scores.keys()):
                score = all_scores[stype]
                status = "✓" if stype == sign_type else " "
                print(f"      {status} {stype:<12}: {score:.4f} ({score*100:.2f}%)")
            
            # Visualize on image if requested
            if visualize:
                x, y, w, h = bbox
                annotated_image = self._draw_detection(
                    annotated_image, 
                    x, y, w, h, 
                    sign_type, 
                    confidence
                )
        
        # Save output if path provided
        if output_path is not None:
            cv2.imwrite(output_path, annotated_image)
            print(f"Output saved to: {output_path}")
        
        return results, annotated_image
    
    def _draw_detection(self, image: cv2.Mat, 
                       x: int, y: int, w: int, h: int,
                       label: str, confidence: float) -> cv2.Mat:
        """
        Draw bounding box and label on the image.
        
        Args:
            image (cv2.Mat): Input image
            x, y, w, h (int): Bounding box coordinates and dimensions
            label (str): Sign type label
            confidence (float): Classification confidence
            
        Returns:
            cv2.Mat: Annotated image
        """
        # Color for bounding box based on confidence level
        if confidence >= 0.7:
            color = (0, 255, 0)    # Green - High confidence
        elif confidence >= 0.5:
            color = (0, 255, 255)  # Yellow - Medium confidence
        else:
            color = (0, 0, 255)    # Red - Low confidence
        
        # Draw bounding rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Prepare text
        text = f"{label}: {confidence:.2f}"
        
        # Draw text with background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image,
            (x, y - text_height - baseline - 5),
            (x + text_width + 5, y),
            color,
            -1  # Filled rectangle
        )
        
        # Put text
        cv2.putText(
            image,
            text,
            (x + 3, y - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness
        )
        
        return image
    
    def process_batch(self, input_dir: str, 
                     output_dir: Optional[str] = None) -> None:
        """
        Process all images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (Optional[str]): Directory to save output images
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"Input directory not found: {input_dir}")
            return
        
        # Create output directory if needed
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Process each image
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        for image_file in image_files:
            print(f"\nProcessing: {image_file.name}")
            
            output_file = None
            if output_dir is not None:
                output_file = str(Path(output_dir) / f"detected_{image_file.name}")
            
            try:
                self.process_image(str(image_file), visualize=True, output_path=output_file)
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
    
    def run_video(self, video_source: int = 0, display: bool = True) -> None:
        """
        Process video stream (webcam or video file).
        
        Args:
            video_source (int or str): Video source (0 for webcam, or path to video file)
            display (bool): Whether to display results in real-time
        """
        import time
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Failed to open video source: {video_source}")
            return
        
        print(f"Processing video from source: {video_source}")
        print("Keyboard controls:")
        print("  - 'q': Quit")
        print("  - 'c': Capture current frame")
        print("  - 's': Toggle detection statistics display")
        print()
        
        frame_count = 0
        sign_count = 0
        show_stats = True
        
        # Create output directory for captured frames
        capture_dir = Path('captures')
        capture_dir.mkdir(exist_ok=True)
        
        # FPS calculation
        prev_frame_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Video ended or failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Detect signs
            detected_signs = self.detector.detect_signs(frame)
            sign_count = len(detected_signs)
            
            # Classify and annotate
            annotated_frame = frame.copy()
            detected_info = []
            
            for idx, (roi, bbox) in enumerate(detected_signs):
                # Get verbose classification with all scores
                sign_type, confidence, all_scores = self.classifier.classify_verbose(roi)
                detected_info.append((sign_type, confidence, all_scores))
                
                x, y, w, h = bbox
                annotated_frame = self._draw_detection(
                    annotated_frame, x, y, w, h, sign_type, confidence
                )
                
                # Print to console occasionally (every 30 frames to avoid spam)
                if frame_count % 30 == 0 and idx == 0:
                    print(f"\n[Frame {frame_count}] Detected: {sign_type} ({confidence*100:.1f}%)", end="")
                    for stype in sorted(all_scores.keys()):
                        if stype != sign_type:
                            print(f" | {stype}: {all_scores[stype]*100:.1f}%", end="")
                    print()
            
            # Calculate FPS
            fps = 1 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0
            prev_frame_time = current_time
            
            # Display statistics on frame
            if show_stats:
                stats_text = f"FPS: {fps:.1f} | Signs Detected: {sign_count} | Frames: {frame_count}"
                cv2.putText(
                    annotated_frame,
                    stats_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Add background for text
                cv2.rectangle(annotated_frame, (5, 5), (500, 40), (0, 0, 0), -1)
                cv2.putText(
                    annotated_frame,
                    stats_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display
            if display:
                cv2.imshow('Traffic Sign Detection - Real-time', annotated_frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuit requested by user")
                    break
                elif key == ord('c'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = capture_dir / f"capture_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), annotated_frame)
                    print(f"✓ Frame captured and saved: {output_path}")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print(f"Statistics display: {'ON' if show_stats else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{'=' * 60}")
        print(f"Video processing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average detection rate: {sign_count} signs per frame")
        print(f"Captured frames saved in: {capture_dir.absolute()}")
        print(f"{'=' * 60}")


def main():
    """Main entry point for the Traffic Sign Recognition system."""
    
    # Initialize the system
    try:
        tsr = TrafficSignRecognition(templates_dir='templates')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the 'templates' directory exists with template images.")
        return
    
    # Real-time Camera Testing
    print("=" * 60)
    print("Traffic Sign Detection System - Real-time Webcam Mode")
    print("=" * 60)
    print("\nStarting real-time detection from webcam...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to capture current frame")
    print("\n" + "=" * 60 + "\n")
    
    tsr.run_video(video_source=0, display=True)


if __name__ == '__main__':
    main()
