"""
Skin region extraction using MediaPipe FaceLandmarker.

Creates a mask that covers only skin-bearing areas by:
  1. Drawing the face oval polygon
  2. Punching out eyes, eyebrows, and lips
"""

import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ---------------------------------------------------------------------------
# Landmark index groups
# ---------------------------------------------------------------------------
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EYEBROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
RIGHT_EYEBROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]

LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def ensure_landmarker_model() -> None:
    """Download face_landmarker.task if not already present."""
    MODELS_DIR.mkdir(exist_ok=True)
    if not LANDMARKER_PATH.exists():
        print("Downloading face_landmarker.task …")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("face_landmarker.task ready.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_skin_crop(image_rgb: np.ndarray) -> np.ndarray | None:
    """
    Return a masked copy of *image_rgb* containing only skin pixels.
    Non-skin pixels are set to black.
    Returns None if no face is detected.
    """
    ensure_landmarker_model()

    h, w = image_rgb.shape[:2]

    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    def pts(indices: list[int]) -> np.ndarray:
        return np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
            dtype=np.int32,
        )

    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill face oval
    cv2.fillPoly(mask, [pts(FACE_OVAL)], 255)

    # Punch out non-skin regions
    for region in [LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS]:
        cv2.fillPoly(mask, [pts(region)], 0)

    # Apply mask
    skin_crop = image_rgb.copy()
    skin_crop[mask == 0] = 0
    return skin_crop
