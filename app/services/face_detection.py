import base64
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app.schemas import DetectionResponse, FaceDetail

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_detector.tflite")

# Forehead / side padding ratios
PAD_TOP = 0.30   # 30% of bbox height added above (covers forehead)
PAD_SIDE = 0.05  # 5% of bbox width added on each side


def detect_faces(image_array: np.ndarray, confidence: float) -> DetectionResponse:
    h, w = image_array.shape[:2]

    options = mp_vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=os.path.abspath(MODEL_PATH)),
        min_detection_confidence=confidence,
    )

    face_count = 0
    face_details = []
    annotated = image_array.copy()

    with mp_vision.FaceDetector.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
        result = detector.detect(mp_image)

        for detection in result.detections:
            face_count += 1
            bbox = detection.bounding_box

            pad_top = int(bbox.height * PAD_TOP)
            pad_side = int(bbox.width * PAD_SIDE)

            x_min = max(bbox.origin_x - pad_side, 0)
            y_min = max(bbox.origin_y - pad_top, 0)
            x_max = min(bbox.origin_x + bbox.width + pad_side, w)
            y_max = min(bbox.origin_y + bbox.height, h)

            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (102, 126, 234), 3)

            score = detection.categories[0].score
            cv2.putText(
                annotated,
                f"{score:.2%}",
                (x_min, max(y_min - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (102, 126, 234),
                2,
            )

            face_details.append(FaceDetail(
                face_id=face_count,
                confidence=score,
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
            ))

    _, buffer = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return DetectionResponse(
        face_count=face_count,
        face_details=face_details,
        annotated_image=annotated_b64,
    )
