"""
POST /analyze — full skin analysis pipeline.

Pipeline:
  1. FaceLandmarker → skin mask crop
  2a. OpenCV feature extraction  ┐ ThreadPoolExecutor
  2b. EfficientNet + skintel     │ (3 threads, wall time ≈ slowest)
  2c. Face detection             ┘
  3. Rule-based skin type classification (from OpenCV features, ~0ms)
  4. Scoring engine → 10 concern scores
"""

import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.schemas import AnalysisResponse, ConcernResult, OpenCVFeatures, PipelineTiming
from app.services.face_detection import detect_faces
from app.services.feature_extraction import extract_opencv_features
from app.services.model_inference import _models_available, run_parallel_inference
from app.services.scoring import calculate_concerns
from app.services.skin_crop import extract_skin_crop
from app.services.skin_type import classify_skin_type

router = APIRouter(prefix="/analyze", tags=["analysis"])
log = logging.getLogger("skinscope.pipeline")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Sri Lanka Standard Time = UTC+5:30
_SL_TZ = timezone(timedelta(hours=5, minutes=30))


def _encode(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode("utf-8")


@router.post("", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    t_start  = time.perf_counter()
    dt_start = datetime.now(_SL_TZ)

    # ── 1. Decode image ───────────────────────────────────────────────────────
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_rgb = np.array(image)
    h, w = image_rgb.shape[:2]

    log.info("═" * 65)
    log.info("  SKINSCOPE ANALYSIS PIPELINE")
    log.info(f"  START   {dt_start.strftime('%Y-%m-%d %H:%M:%S')} (Sri Lanka Time)")
    log.info("═" * 65)
    log.info(f"  INPUT   filename={file.filename}  size={w}×{h}px  "
             f"conf_threshold={confidence}")
    log.info(f"  MODELS  {' | '.join(k for k, v in _models_available.items() if v) or 'OpenCV only'}")

    # ── 2. Skin crop ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    skin_crop = extract_skin_crop(image_rgb)
    skin_s = time.perf_counter() - t0

    if skin_crop is None:
        log.warning("  SKIN CROP  no face detected — aborting")
        raise HTTPException(status_code=422, detail="No face detected in image.")

    skin_px = int(np.any(skin_crop > 0, axis=2).sum())
    log.info(f"  SKIN CROP  {skin_px:,} skin pixels extracted  ({skin_s:.3f}s)")

    # ── 3. Parallel feature extraction (3 threads) ───────────────────────────
    log.info("  PARALLEL THREADS  starting OpenCV + EfficientNet + Detection …")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_opencv = pool.submit(extract_opencv_features, skin_crop)
        f_ml     = pool.submit(run_parallel_inference,  image_rgb, skin_crop)
        f_detect = pool.submit(detect_faces,            image_rgb, confidence)

        opencv_features  = f_opencv.result()
        ml_features      = f_ml.result()
        detection_result = f_detect.result()

    parallel_s = time.perf_counter() - t0
    all_features = {**opencv_features, **ml_features}

    log.info(f"  PARALLEL THREADS  done in {parallel_s:.3f}s")
    log.info("  ┌─ RAW FEATURES ───────────────────────────────────────")
    log.info(f"  │  OpenCV")
    for k, v in opencv_features.items():
        bar = "▓" * int(v * 20)
        log.info(f"  │    {k:<28} = {v:.4f}  {bar}")
    log.info(f"  │  ML Models")
    for k, v in ml_features.items():
        bar = "▓" * int(v * 20)
        log.info(f"  │    {k:<28} = {v:.4f}  {bar}")
    log.info(f"  │  Face Detection")
    log.info(f"  │    faces_detected           = {detection_result.face_count}")
    log.info("  └──────────────────────────────────────────────────────")

    # ── 4. Rule-based skin type classification ────────────────────────────────
    skin_type = classify_skin_type(opencv_features)

    # ── 5. Score 10 concerns ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    concerns = calculate_concerns(all_features)
    scoring_s = time.perf_counter() - t0
    log.info(f"  SCORING  done in {scoring_s:.3f}s")

    # ── 6. Build response ─────────────────────────────────────────────────────
    models_used = ["OpenCV"]
    if _models_available.get("effnet"):
        models_used.append("EfficientNet-B0")
    if _models_available.get("skintel"):
        models_used.append("skintelligent-acne")

    annotated_bytes = base64.b64decode(detection_result.annotated_image)
    annotated_array = np.array(Image.open(io.BytesIO(annotated_bytes)).convert("RGB"))

    total_s  = time.perf_counter() - t_start
    dt_end   = datetime.now(_SL_TZ)

    log.info(f"  OUTPUT  top concern = {concerns[0].name} ({concerns[0].score}/95)")
    log.info(f"  OUTPUT  skin type   = {skin_type}")
    log.info(f"  ┌─ TIMING BREAKDOWN ───────────────────────────────────")
    log.info(f"  │  Skin Crop        : {skin_s:.3f}s")
    log.info(f"  │  Parallel Threads : {parallel_s:.3f}s  (wall time, 3 threads)")
    log.info(f"  │  Scoring          : {scoring_s:.3f}s")
    log.info(f"  │  TOTAL            : {total_s:.3f}s")
    log.info(f"  └──────────────────────────────────────────────────────")
    log.info(f"  END     {dt_end.strftime('%Y-%m-%d %H:%M:%S')} (Sri Lanka Time)")
    log.info("═" * 65)

    return AnalysisResponse(
        concerns=[
            ConcernResult(
                name=c.name,
                score=c.score,
                severity=c.severity,
                description=c.description,
            )
            for c in concerns
        ],
        features=OpenCVFeatures(**{
            k: v for k, v in opencv_features.items()
            if k in OpenCVFeatures.model_fields
        }),
        skin_crop_image=_encode(skin_crop),
        annotated_image=_encode(annotated_array),
        models_used=models_used,
        skin_type=skin_type,
        timing=PipelineTiming(
            skin_crop_s=round(skin_s, 3),
            parallel_pipeline_s=round(parallel_s, 3),
            scoring_s=round(scoring_s, 3),
            total_s=round(total_s, 3),
        ),
    )
