"""Traffic sign detection and classification system."""
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from core.detector import SignDetector
from core.classifier import SignClassifier


class TrafficSignRecognition:
    """Detects and classifies blue circular traffic signs."""
    
    def __init__(self, templates_dir: str = 'templates', confidence_threshold: float = 0.7):
        self.detector = SignDetector()
        self.classifier = SignClassifier(
            templates_dir=templates_dir,
            template_matching_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
    
    def process_image(self, image_path: str, visualize: bool = True, output_path: Optional[str] = None) -> Tuple[List[Tuple[str, float]], cv2.Mat]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        detected_signs = self.detector.detect_signs(image)
        if len(detected_signs) == 0:
            print(f"No signs detected in {image_path}")
            return [], image
        
        print(f"Detected {len(detected_signs)} signs in {image_path}")
        results = []
        annotated_image = image.copy()
        
        for idx, (roi, bbox) in enumerate(detected_signs):
            sign_type, confidence = self.classifier.classify(roi)
            results.append((sign_type, confidence))
            print(f"  Sign {idx + 1}: {sign_type} ({confidence*100:.1f}%)")
            
            if visualize:
                x, y, w, h = bbox
                annotated_image = self._draw_detection(annotated_image, x, y, w, h, sign_type, confidence)
        
        if output_path is not None:
            cv2.imwrite(output_path, annotated_image)
            print(f"Output saved to: {output_path}")
        
        return results, annotated_image
    
    def _draw_detection(self, image: cv2.Mat, x: int, y: int, w: int, h: int, label: str, confidence: float) -> cv2.Mat:
        if confidence >= 0.7:
            color = (0, 255, 0)
        elif confidence >= 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{label}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(image, (x, y - text_height - baseline - 5), (x + text_width + 5, y), color, -1)
        cv2.putText(image, text, (x + 3, y - baseline - 2), font, font_scale, (255, 255, 255), font_thickness)
        
        return image
    
    def process_batch(self, input_dir: str, output_dir: Optional[str] = None) -> None:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Input directory not found: {input_dir}")
            return
        
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        for image_file in image_files:
            print(f"\nProcessing: {image_file.name}")
            output_file = str(Path(output_dir) / f"detected_{image_file.name}") if output_dir else None
            
            try:
                self.process_image(str(image_file), visualize=True, output_path=output_file)
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
    
    def run_video(self, video_source: int = 0, display: bool = True) -> None:
        import time
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Failed to open video source: {video_source}")
            return
        
        print(f"Processing video from source: {video_source}")
        print("Controls: 'q'=quit, 'c'=capture, 's'=toggle stats")
        
        frame_count = 0
        show_stats = True
        capture_dir = Path('captures')
        capture_dir.mkdir(exist_ok=True)
        prev_frame_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended")
                break
            
            frame_count += 1
            current_time = time.time()
            detected_signs = self.detector.detect_signs(frame)
            annotated_frame = frame.copy()
            
            for idx, (roi, bbox) in enumerate(detected_signs):
                sign_type, confidence = self.classifier.classify(roi)
                x, y, w, h = bbox
                annotated_frame = self._draw_detection(annotated_frame, x, y, w, h, sign_type, confidence)
                
                if frame_count % 30 == 0 and idx == 0:
                    print(f"[Frame {frame_count}] {sign_type} ({confidence*100:.1f}%)")
            
            fps = 1 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0
            prev_frame_time = current_time
            
            if show_stats:
                stats_text = f"FPS: {fps:.1f} | Signs: {len(detected_signs)}"
                cv2.rectangle(annotated_frame, (5, 5), (400, 40), (0, 0, 0), -1)
                cv2.putText(annotated_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if display:
                cv2.imshow('Traffic Sign Detection', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuit requested")
                    break
                elif key == ord('c'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = capture_dir / f"capture_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), annotated_frame)
                    print(f"Frame captured: {output_path.name}")
                elif key == ord('s'):
                    show_stats = not show_stats
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


def main():
    try:
        tsr = TrafficSignRecognition(templates_dir='templates')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the 'templates' directory exists.")
        return
    
    print("\nTraffic Sign Detection - Webcam Mode")
    print("Controls: 'q'=quit, 'c'=capture, 's'=toggle stats\n")
    tsr.run_video(video_source=0, display=True)


if __name__ == '__main__':
    main()
