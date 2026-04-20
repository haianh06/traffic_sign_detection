import cv2
import numpy as np
from typing import List, Tuple


class SignDetector:
    """Detects blue circular traffic signs using HSV and contour analysis."""
    
    def __init__(self, 
                 hsv_lower: Tuple[int, int, int] = (90, 50, 50),
                 hsv_upper: Tuple[int, int, int] = (130, 255, 255),
                 min_area: int = 500,
                 max_area: int = 50000,
                 circularity_threshold: float = 0.7):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv
    
    def create_blue_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        mask = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def calculate_circularity(self, contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return 4 * np.pi * area / (perimeter ** 2)
    
    def detect_signs(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        hsv_image = self.preprocess(image)
        mask = self.create_blue_mask(hsv_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_signs = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            circularity = self.calculate_circularity(contour)
            if circularity < self.circularity_threshold:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            padding = int(0.1 * max(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            roi = image[y_start:y_end, x_start:x_end]
            detected_signs.append((roi, (x, y, w, h)))
        
        return detected_signs
    
    def get_detection_with_mask(self, image: np.ndarray) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int, int, int]]], np.ndarray]:
        hsv_image = self.preprocess(image)
        mask = self.create_blue_mask(hsv_image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_signs = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            circularity = self.calculate_circularity(contour)
            if circularity < self.circularity_threshold:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            padding = int(0.1 * max(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            roi = image[y_start:y_end, x_start:x_end]
            detected_signs.append((roi, (x, y, w, h)))
        
        return detected_signs, mask
