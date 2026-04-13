"""
Configuration File for Traffic Sign Recognition System

This module contains all configurable parameters for the detection and classification pipeline.
Modify these values to tune the system for your specific environment and use case.
"""

# ============================================================================
# DETECTOR CONFIGURATION - HSV Color Space Settings
# ============================================================================

# HSV thresholds for blue color detection
# OpenCV HSV: H (0-180), S (0-255), V (0-255)
# Standard blue: H=90-130
DETECTOR_HSV_LOWER = (90, 50, 50)      # (H_min, S_min, V_min)
DETECTOR_HSV_UPPER = (130, 255, 255)   # (H_max, S_max, V_max)

# Area filtering
DETECTOR_MIN_AREA = 500                # Minimum contour area in pixels²
DETECTOR_MAX_AREA = 50000              # Maximum contour area in pixels²

# Circularity filtering
# Circularity = 4π × Area / Perimeter²
# Circle = 1.0, Square = 0.785, varied shapes = lower values
DETECTOR_CIRCULARITY_THRESHOLD = 0.7   # Range: 0.0-1.0 (higher = stricter)

# ============================================================================
# CLASSIFIER CONFIGURATION - Template Matching Settings
# ============================================================================

# Template directory and size
CLASSIFIER_TEMPLATES_DIR = 'templates'
CLASSIFIER_TEMPLATE_SIZE = (64, 64)    # (width, height) for resizing templates

# Template confidence threshold
# Template matching correlation values: -1.0 to 1.0
CLASSIFIER_TEMPLATE_THRESHOLD = 0.7    # Range: 0.0-1.0 (higher = stricter)

# Classification method: 'template_matching' or 'hog_features'
CLASSIFIER_PRIMARY_METHOD = 'template_matching'

# ============================================================================
# MAIN PIPELINE CONFIGURATION
# ============================================================================

# Input/Output directories
INPUT_IMAGE_DIR = 'test_images'
OUTPUT_IMAGE_DIR = 'results'
TEMPLATES_DIR = 'templates'

# Visualization settings
ENABLE_VISUALIZATION = True
SAVE_ANNOTATED_IMAGES = True

# Video processing settings
VIDEO_DISPLAY_ENABLED = True
VIDEO_FRAME_SKIP = 1                  # Process every Nth frame (1 = all frames)

# ============================================================================
# DEBUG & LOGGING SETTINGS
# ============================================================================

DEBUG_MODE = False                     # Print detailed detection information
SAVE_DEBUG_MASKS = False               # Save intermediate HSV masks for debugging
SHOW_DETECTION_CONFIDENCE = True       # Display confidence scores in visualization

# ============================================================================
# HARDWARE OPTIMIZATION
# ============================================================================

# Multi-threading for batch processing
ENABLE_MULTIPROCESSING = False         # Not yet implemented
NUM_WORKERS = 4                        # Threads for parallel processing

# ============================================================================
# ADVANCED PARAMETERS
# ============================================================================

# Morphological operations for mask cleaning
MORPH_KERNEL_SIZE = (5, 5)            # Kernel size for erosion/dilation
MORPH_ITERATIONS = 1                   # Number of morphological operation iterations

# Gaussian blur parameters
GAUSSIAN_BLUR_KERNEL = (5, 5)         # Kernel size for blur
GAUSSIAN_BLUR_SIGMA = 0.0             # Standard deviation

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

PRESETS = {
    'strict': {
        'min_area': 1000,
        'max_area': 30000,
        'circularity_threshold': 0.85,
        'template_threshold': 0.8,
    },
    'balanced': {
        'min_area': 500,
        'max_area': 50000,
        'circularity_threshold': 0.7,
        'template_threshold': 0.7,
    },
    'lenient': {
        'min_area': 300,
        'max_area': 60000,
        'circularity_threshold': 0.6,
        'template_threshold': 0.6,
    },
    'small_signs': {
        'min_area': 200,
        'max_area': 20000,
        'circularity_threshold': 0.65,
        'template_threshold': 0.65,
    },
    'large_signs': {
        'min_area': 5000,
        'max_area': 100000,
        'circularity_threshold': 0.75,
        'template_threshold': 0.75,
    },
}


def apply_preset(preset_name: str) -> dict:
    """
    Apply a preset configuration.
    
    Args:
        preset_name (str): Name of preset ('strict', 'balanced', 'lenient', etc.)
        
    Returns:
        dict: Configuration dictionary for the preset
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
from config import apply_preset, DETECTOR_HSV_LOWER, DETECTOR_HSV_UPPER

# Using default configuration
from core.detector import SignDetector
detector = SignDetector(
    hsv_lower=DETECTOR_HSV_LOWER,
    hsv_upper=DETECTOR_HSV_UPPER,
    min_area=DETECTOR_MIN_AREA,
    max_area=DETECTOR_MAX_AREA,
    circularity_threshold=DETECTOR_CIRCULARITY_THRESHOLD
)

# Or use a preset
preset = apply_preset('balanced')
detector = SignDetector(
    min_area=preset['min_area'],
    max_area=preset['max_area'],
    circularity_threshold=preset['circularity_threshold']
)
"""
