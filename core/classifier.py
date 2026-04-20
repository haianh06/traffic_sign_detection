import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class SignClassifier:
    """Classifies traffic signs using template matching and HOG features."""
    
    def __init__(self, 
                 templates_dir: str = 'templates',
                 template_size: Tuple[int, int] = (64, 64),
                 template_matching_threshold: float = 0.7):
        self.templates_dir = Path(templates_dir)
        self.template_size = template_size
        self.threshold = template_matching_threshold
        self.template_mapping = {
            'straight': 'up.png',
            'left': 'left.png',
            'right': 'right.png',
            'parking': 'p.png'
        }
        self.templates = {}
        self.load_templates()
    
    def load_templates(self) -> None:
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
        
        for sign_type, template_name in self.template_mapping.items():
            template_path = self.templates_dir / template_name
            if not template_path.exists():
                print(f"Warning: Template file not found: {template_path}")
                self.templates[sign_type] = None
                continue
            
            template = cv2.imread(str(template_path))
            if template is None:
                print(f"Warning: Failed to load template: {template_path}")
                self.templates[sign_type] = None
                continue
            
            self.templates[sign_type] = cv2.resize(template, self.template_size)
    
    def template_matching(self, roi: np.ndarray) -> Tuple[str, float]:
        roi_resized = cv2.resize(roi, self.template_size)
        best_match = None
        best_confidence = -1.0
        
        for sign_type, template in self.templates.items():
            if template is None:
                continue
            result = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = sign_type
        
        return best_match, best_confidence
    
    def extract_hog_features(self, roi: np.ndarray) -> np.ndarray:
        roi_resized = cv2.resize(roi, self.template_size)
        if len(roi_resized.shape) == 3:
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_resized
        
        hog = cv2.HOGDescriptor(
            (64, 64), (16, 16), (8, 8), (8, 8), 9, 1, -1.0, 0, 0.2, True
        )
        return hog.compute(roi_gray)
    
    def classify_hog_features(self, roi: np.ndarray) -> Tuple[str, float]:
        hog_features = self.extract_hog_features(roi)
        return 'hog_not_trained', 0.0
    
    def classify(self, 
                 roi: np.ndarray, 
                 method: str = 'template_matching') -> Tuple[str, float]:
        if method == 'template_matching':
            return self.template_matching(roi)
        elif method == 'hog_features':
            return self.classify_hog_features(roi)
        else:
            return self.template_matching(roi)
    
    def batch_classify(self, 
                      rois: List[np.ndarray],
                      method: str = 'template_matching') -> List[Tuple[str, float]]:
        """
        Classify multiple ROIs.
        
        Args:
            rois (List[np.ndarray]): List of regions of interest
            method (str): Classification method
            
        Returns:
            List[Tuple[str, float]]: List of classification results
        """
        results = []
        for roi in rois:
            result = self.classify(roi, method=method)
            results.append(result)
        
        return results
    
    def classify_verbose(self, roi: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify with detailed output showing all template scores.
        
        Args:
            roi (np.ndarray): Region of Interest to classify
            
        Returns:
            Tuple containing:
                - Best match sign type
                - Best match confidence
                - Dictionary with all template scores
        """
        # Resize ROI to match template size
        roi_resized = cv2.resize(roi, self.template_size)
        
        all_scores = {}
        best_match = None
        best_confidence = -1.0
        
        for sign_type, template in self.templates.items():
            if template is None:
                all_scores[sign_type] = 0.0
                continue
            
            # Perform template matching
            result = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
            
            # Get the maximum correlation value
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            all_scores[sign_type] = max_val
            
            # Update best match if this confidence is higher
            if max_val > best_confidence:
                best_confidence = max_val
                best_match = sign_type
        
        # Always return best match (no unknown classification)
        return best_match, best_confidence, all_scores
