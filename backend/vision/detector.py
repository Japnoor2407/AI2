"""
Local Video-Based Vehicle Detection and Tracking Pipeline (Phase 4).

Provides lightweight, fully local (no cloud/paid APIs) vehicle detection,
tracking, and virtual tripwire counting using OpenCV.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2


class VehicleDetection:
    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        self.bbox = bbox  # (x, y, w, h)
        self.class_name = class_name
        self.confidence = confidence
        self.center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)


class LocalVehicleDetector:
    """
    Local vehicle detector using adaptive background subtraction and morphological
    contour classification for traffic surveillance cameras.
    """

    def __init__(self, min_contour_area: int = 500):
        self.min_contour_area = min_contour_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=32, detectShadows=True
        )

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Detect moving vehicles in a frame."""
        if frame is None or frame.size == 0 or np.count_nonzero(frame) == 0:
            return []

        # Preprocessing: Grayscale + Gaussian Blur
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blurred)

        # Remove shadows (value 127 in MOG2)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio and area heuristic classification
            aspect_ratio = float(w) / max(1, h)
            if area < 1500:
                class_name = "motorbike"
            elif area < 6000:
                class_name = "car"
            elif aspect_ratio > 1.8 or area > 12000:
                class_name = "truck"
            else:
                class_name = "bus"

            conf = min(0.99, round(0.5 + (area / 15000.0) * 0.45, 2))
            detections.append(VehicleDetection((x, y, w, h), class_name, conf))

        return detections
