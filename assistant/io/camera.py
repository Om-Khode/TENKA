"""
camera.py — Webcam capture for vision tasks.

Lazy-loads opencv-python to avoid overhead when camera is disabled or unused.
Captures a single frame, resizes it, and returns a base64 JPEG.
"""

import logging
import base64
from io import BytesIO

logger = logging.getLogger("camera")

def capture_camera_frame_base64(camera_index: int = 0, quality: int = 75) -> str | None:
    """
    Open the camera, capture one frame, release it, and return as base64 JPEG.
    Resizes the image to a maximum of 1280px wide to save bandwidth.
    Returns None if capture fails.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        logger.error("opencv-python or Pillow is not installed.")
        return None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {camera_index}")
        return None

    try:
        # Read a single frame
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to read frame from camera")
            return None

        # OpenCV returns BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for resizing and saving
        img = Image.fromarray(frame_rgb)
        
        # Resize if wider than 1280px (preserve aspect ratio)
        max_width = 1280
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = int((float(img.height) * float(ratio)))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Save to BytesIO as JPEG
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        
        # Convert to base64
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str

    except Exception as e:
        logger.error(f"Error during camera capture: {e}")
        return None
    finally:
        # Always release the camera immediately
        cap.release()

def capture_camera_frame_numpy(camera_index: int = 0):
    """
    Capture a single webcam frame and return it as an RGB numpy array.
    Returns None on any failure.
    Used by face recognition (face_recognition library expects numpy arrays).
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python is not installed.")
        return None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {camera_index}")
        return None

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to read frame from camera")
            return None

        # OpenCV returns BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    except Exception as e:
        logger.error(f"Error during camera capture (numpy): {e}")
        return None
    finally:
        cap.release()

def numpy_frame_to_base64(frame_rgb, quality: int = 75) -> str | None:
    """
    Convert an RGB numpy array (already captured) to a base64 JPEG string.
    Used when we already have a frame from capture_camera_frame_numpy()
    and need to send it to a vision LLM without reopening the camera.
    Returns base64 string or None on failure.
    """
    try:
        from PIL import Image
        import io
        import base64
        from .. import config as _config

        img = Image.fromarray(frame_rgb)
        
        max_width = _config.CAMERA_MAX_WIDTH
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = int((float(img.height) * float(ratio)))
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str
    except Exception as e:
        logger.error(f"Error converting numpy frame to base64: {e}")
        return None
