from pydantic import BaseModel
from typing import List


# ---------------------------------------------------------------------------
# Detection schemas
# ---------------------------------------------------------------------------
class FaceDetail(BaseModel):
    face_id: int
    confidence: float
    x: int
    y: int
    width: int
    height: int


class DetectionResponse(BaseModel):
    face_count: int
    face_details: List[FaceDetail]
    annotated_image: str          # base64-encoded PNG


# ---------------------------------------------------------------------------
# Analysis schemas
# ---------------------------------------------------------------------------
class ConcernResult(BaseModel):
    name: str
    score: float                  # 10–95
    severity: str                 # Low | Moderate | High
    description: str


class PipelineTiming(BaseModel):
    skin_crop_s: float
    parallel_pipeline_s: float
    scoring_s: float
    total_s: float


class OpenCVFeatures(BaseModel):
    redness: float
    oiliness: float
    brightness: float
    texture_variance: float
    color_variance: float
    pigmentation: float
    dark_spot_ratio: float
    local_redness_clusters: float
    saturation_inv: float
    flakiness: float
    lab_uniformity: float


class AnalysisResponse(BaseModel):
    concerns: List[ConcernResult]
    features: OpenCVFeatures
    skin_crop_image: str          # base64-encoded PNG of masked skin region
    annotated_image: str          # base64-encoded PNG with face box
    models_used: List[str]        # which ML models contributed
    skin_type: str                # normal | oily | dry | combination | sensitive
    timing: PipelineTiming
