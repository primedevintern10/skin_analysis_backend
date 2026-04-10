import io

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from app.schemas import DetectionResponse
from app.services.face_detection import detect_faces

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)
    return detect_faces(image_array, confidence)

