"""
Debug and Utility Tools for Traffic Sign Recognition

This module contains all debug and tuning tools:
1. HSV Slider Tool - Adjust HSV ranges with sliders
2. Auto HSV Detector - Auto-detect optimal HSV range from blue sign
3. Detection Debugger - Step-by-step detection visualization
4. Template Matching Debugger - Shows template comparison scores

Run with: python tools.py <tool_name>
Available tools: hsv_slider, auto_hsv, detection_debug, template_debug
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from core.detector import SignDetector
from core.classifier import SignClassifier


# ============================================================================
# 1. HSV RANGE SLIDER TOOL
# ============================================================================

class HSVSliderTool:
    """Interactive HSV range adjustment with real-time visualization."""
    
    def __init__(self):
        self.h_lower = 90
        self.h_upper = 130
        self.s_lower = 50
        self.s_upper = 255
        self.v_lower = 50
        self.v_upper = 255
        self.min_area = 500
        self.circ_threshold = 0.7
    
    def nothing(self, x):
        """Dummy callback for trackbars."""
        pass
    
    def run(self):
        """Run the slider tool."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("="*70)
        print("🎚️  HSV RANGE SLIDER TOOL")
        print("="*70)
        print("\n📊 Adjust sliders until blue sign shows WHITE in left window")
        print("="*70 + "\n")
        
        cv2.namedWindow('HSV Range Sliders')
        
        cv2.createTrackbar('H_MIN', 'HSV Range Sliders', self.h_lower, 180, self.nothing)
        cv2.createTrackbar('H_MAX', 'HSV Range Sliders', self.h_upper, 180, self.nothing)
        cv2.createTrackbar('S_MIN', 'HSV Range Sliders', self.s_lower, 255, self.nothing)
        cv2.createTrackbar('S_MAX', 'HSV Range Sliders', self.s_upper, 255, self.nothing)
        cv2.createTrackbar('V_MIN', 'HSV Range Sliders', self.v_lower, 255, self.nothing)
        cv2.createTrackbar('V_MAX', 'HSV Range Sliders', self.v_upper, 255, self.nothing)
        cv2.createTrackbar('MIN_AREA', 'HSV Range Sliders', self.min_area, 5000, self.nothing)
        cv2.createTrackbar('CIRC_x100', 'HSV Range Sliders', int(self.circ_threshold * 100), 100, self.nothing)
        
        detector = SignDetector()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (960, 720))
            
            # Get slider values
            self.h_lower = cv2.getTrackbarPos('H_MIN', 'HSV Range Sliders')
            self.h_upper = cv2.getTrackbarPos('H_MAX', 'HSV Range Sliders')
            self.s_lower = cv2.getTrackbarPos('S_MIN', 'HSV Range Sliders')
            self.s_upper = cv2.getTrackbarPos('S_MAX', 'HSV Range Sliders')
            self.v_lower = cv2.getTrackbarPos('V_MIN', 'HSV Range Sliders')
            self.v_upper = cv2.getTrackbarPos('V_MAX', 'HSV Range Sliders')
            self.min_area = cv2.getTrackbarPos('MIN_AREA', 'HSV Range Sliders')
            self.circ_threshold = cv2.getTrackbarPos('CIRC_x100', 'HSV Range Sliders') / 100.0
            
            # Update detector
            detector.hsv_lower = np.array([self.h_lower, self.s_lower, self.v_lower], dtype=np.uint8)
            detector.hsv_upper = np.array([self.h_upper, self.s_upper, self.v_upper], dtype=np.uint8)
            detector.min_area = self.min_area
            detector.circularity_threshold = self.circ_threshold
            
            # Process
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, detector.hsv_lower, detector.hsv_upper)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected = 0
            vis_frame = frame.copy()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                
                circularity = detector.calculate_circularity(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                if circularity >= self.circ_threshold:
                    detected += 1
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"{circularity:.2f}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([mask_3ch, vis_frame])
            
            info_text = f"H:{self.h_lower}-{self.h_upper} S:{self.s_lower}-{self.s_upper} V:{self.v_lower}-{self.v_upper} | Area>{self.min_area} | Circ>{self.circ_threshold:.2f}"
            cv2.putText(combined, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(combined, (5, 5), (combined.shape[1]-5, 45), (0, 0, 0), -1)
            cv2.putText(combined, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            detect_text = f"DETECTED: {detected} signs"
            cv2.putText(combined, detect_text, (10, combined.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detected > 0 else (0, 0, 255), 2)
            
            cv2.imshow('HSV Range Sliders', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n✓ Current values:")
                print(f"  DETECTOR_HSV_LOWER = ({self.h_lower}, {self.s_lower}, {self.v_lower})")
                print(f"  DETECTOR_HSV_UPPER = ({self.h_upper}, {self.s_upper}, {self.v_upper})")
                print(f"  DETECTOR_MIN_AREA = {self.min_area}")
                print(f"  DETECTOR_CIRCULARITY_THRESHOLD = {self.circ_threshold}")
                print(f"\nCopy these to config.py")
                input("Press ENTER to continue...")
        
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# 2. AUTO HSV DETECTOR
# ============================================================================

class AutoHSVDetector:
    """Automatically detect optimal HSV range."""
    
    def __init__(self):
        self.frame = None
        self.hsv = None
        self.selected_region = None
        self.roi_points = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Draw rectangle by dragging."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self.roi_points = [(self.roi_points[0][0], self.roi_points[0][1]), (x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.roi_points) > 0:
                x1, y1 = self.roi_points[0]
                x2, y2 = x, y
                self.selected_region = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    def analyze_region(self):
        """Analyze HSV values in selected region."""
        if self.selected_region is None:
            return None
        
        x1, y1, x2, y2 = self.selected_region
        roi_hsv = self.hsv[y1:y2, x1:x2]
        
        if roi_hsv.size == 0:
            return None
        
        h_min, h_max = np.min(roi_hsv[:,:,0]), np.max(roi_hsv[:,:,0])
        s_min, s_max = np.min(roi_hsv[:,:,1]), np.max(roi_hsv[:,:,1])
        v_min, v_max = np.min(roi_hsv[:,:,2]), np.max(roi_hsv[:,:,2])
        
        h_lower = max(0, int(h_min - 10))
        h_upper = min(180, int(h_max + 10))
        s_lower = max(0, int(s_min - 20))
        s_upper = min(255, int(s_max + 20))
        v_lower = max(0, int(v_min - 20))
        v_upper = min(255, int(v_max + 20))
        
        print("\n" + "="*70)
        print("🎯 SUGGESTED PARAMETERS:")
        print("="*70)
        print(f"  DETECTOR_HSV_LOWER = ({h_lower}, {s_lower}, {v_lower})")
        print(f"  DETECTOR_HSV_UPPER = ({h_upper}, {s_upper}, {v_upper})")
        print("="*70 + "\n")
    
    def run(self):
        """Run the auto-detector."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("="*70)
        print("🎯 AUTO HSV RANGE DETECTOR")
        print("="*70)
        print("\n📌 INSTRUCTIONS:")
        print("  1. DRAG to select BLUE SIGN region")
        print("  2. Release mouse")
        print("  3. Press SPACE to analyze")
        print("="*70 + "\n")
        
        cv2.namedWindow('Auto HSV Detector')
        cv2.setMouseCallback('Auto HSV Detector', self.mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame = cv2.resize(frame, (960, 720))
            self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            
            display = self.frame.copy()
            
            if len(self.roi_points) == 2:
                x1, y1 = self.roi_points[0]
                x2, y2 = self.roi_points[1]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if self.selected_region:
                x1, y1, x2, y2 = self.selected_region
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cv2.imshow('Auto HSV Detector', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if self.selected_region:
                    self.analyze_region()
        
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# 3. DETECTION DEBUGGER
# ============================================================================

class DetectionDebugger:
    """Debug detection process step-by-step."""
    
    def __init__(self):
        self.detector = SignDetector()
    
    def run(self):
        """Run detection debug."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("\n" + "="*70)
        print("🔍 DETECTION DEBUGGER")
        print("="*70)
        print("\n  'SPACE': Show detection details")
        print("  'q': Quit")
        print("="*70 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (960, 720))
            detected_signs = self.detector.detect_signs(frame)
            
            display = frame.copy()
            for idx, (roi, bbox) in enumerate(detected_signs):
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display, f"Sign {idx + 1}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            info_text = f"Detected: {len(detected_signs)} | Press SPACE to analyze"
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(display, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Detection Debug', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Found {len(detected_signs)} sign(s)")
                for idx, (roi, bbox) in enumerate(detected_signs):
                    print(f"  Sign {idx + 1}: {bbox}")
        
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# 4. TEMPLATE MATCHING DEBUGGER
# ============================================================================

class TemplateMatchingDebugger:
    """Debug template matching visualization."""
    
    def __init__(self):
        self.detector = SignDetector()
        self.classifier = SignClassifier(templates_dir='templates')
    
    def run_detection_debug(self):
        """Run detection and template matching debug."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("\n" + "="*70)
        print("🔍 TEMPLATE MATCHING DEBUGGER")
        print("="*70)
        print("\n  'SPACE': Analyze current frame")
        print("  'q': Quit")
        print("="*70 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (960, 720))
            detected_signs = self.detector.detect_signs(frame)
            
            display = frame.copy()
            for idx, (roi, bbox) in enumerate(detected_signs):
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(display, f"Sign {idx + 1}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            info_text = f"Detected: {len(detected_signs)} | Press SPACE"
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(display, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Template Debug', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if len(detected_signs) > 0:
                    for idx, (roi, bbox) in enumerate(detected_signs):
                        sign_type, confidence, all_scores = self.classifier.classify_verbose(roi)
                        print(f"\nSign {idx + 1}:")
                        print(f"  Best: {sign_type} ({confidence*100:.2f}%)")
                        for stype in sorted(all_scores.keys()):
                            score = all_scores[stype]
                            print(f"    {stype:<12}: {score*100:.2f}%")
        
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================

def print_help():
    """Print help message."""
    print("\n" + "="*70)
    print("Traffic Sign Detection - Debug Tools")
    print("="*70)
    print("\nUsage: python tools.py <tool_name>")
    print("\nAvailable tools:")
    print("  1. hsv_slider      - Adjust HSV ranges with interactive sliders")
    print("  2. auto_hsv        - Auto-detect optimal HSV range by selecting region")
    print("  3. detection_debug - Debug detection process step-by-step")
    print("  4. template_debug  - Debug template matching scores")
    print("\nExample: python tools.py hsv_slider")
    print("="*70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)
    
    tool_name = sys.argv[1].lower()
    
    if tool_name == 'hsv_slider':
        tool = HSVSliderTool()
        tool.run()
    elif tool_name == 'auto_hsv':
        tool = AutoHSVDetector()
        tool.run()
    elif tool_name == 'detection_debug':
        tool = DetectionDebugger()
        tool.run()
    elif tool_name == 'template_debug':
        tool = TemplateMatchingDebugger()
        tool.run_detection_debug()
    else:
        print(f"❌ Unknown tool: {tool_name}")
        print_help()
