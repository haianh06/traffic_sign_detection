"""
Sign Detector Module

This module contains the SignDetector class which is responsible for detecting
blue circular traffic signs in an image using HSV color space and contour analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple


class SignDetector:
    """
    Detects blue circular traffic signs in an image.
    
    Uses HSV color space to mask blue colors and contour detection with
    circularity filtering to identify circular signs.
    """
    
    def __init__(self, 
                 hsv_lower: Tuple[int, int, int] = (90, 50, 50),
                 hsv_upper: Tuple[int, int, int] = (130, 255, 255),
                 min_area: int = 500,
                 max_area: int = 50000,
                 circularity_threshold: float = 0.7):
        """
        Initialize the SignDetector.
        
        Args:
            hsv_lower (Tuple[int, int, int]): Lower HSV threshold for blue color (H, S, V)
            hsv_upper (Tuple[int, int, int]): Upper HSV threshold for blue color (H, S, V)
            min_area (int): Minimum contour area to consider as a sign
            max_area (int): Maximum contour area to consider as a sign
            circularity_threshold (float): Minimum circularity (0.0-1.0) for circular signs
        """
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area = min_area
        self.max_area = max_area
        self.circularity_threshold = circularity_threshold
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image by applying Gaussian blur and converting to HSV.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Preprocessed image in HSV color space
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv
    
    def create_blue_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for blue colors in HSV space.
        
        Args:
            hsv_image (np.ndarray): Image in HSV color space
            
        Returns:
            np.ndarray: Binary mask where blue pixels are white (255)
        """
        # Create mask for blue colors
        mask = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def calculate_circularity(self, contour: np.ndarray) -> float:
        """
        Calculate the circularity of a contour.
        
        Circularity = 4π * Area / Perimeter²
        A perfect circle has circularity = 1.0
        
        Args:
            contour (np.ndarray): Contour array
            
        Returns:
            float: Circularity value (0.0 to 1.0)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity
    
    def detect_signs(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect blue circular signs in the image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]: List of tuples containing:
                - Cropped ROI (region of interest)
                - Bounding box (x, y, w, h)
        """
        # Preprocess the image
        hsv_image = self.preprocess(image)
        
        # Create blue color mask
        mask = self.create_blue_mask(hsv_image)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_signs = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            # Filter by circularity
            circularity = self.calculate_circularity(contour)
            if circularity < self.circularity_threshold:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract ROI with some padding
            padding = int(0.1 * max(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            roi = image[y_start:y_end, x_start:x_end]
            
            # Store ROI and original bounding box
            detected_signs.append((roi, (x, y, w, h)))
        
        return detected_signs
    
    def get_detection_with_mask(self, image: np.ndarray) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int, int, int]]], np.ndarray]:
        """
        Detect signs and return both detections and the generated mask.
        
        Useful for debugging and visualization.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            Tuple containing:
                - List of detected signs (ROI and bounding box)
                - Binary mask used for detection
        """
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
