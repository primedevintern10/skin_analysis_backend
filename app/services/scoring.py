"""
Scoring engine — maps raw features to 10 skin concern scores.

Score range: 10 (minimal concern) → 95 (severe concern).
Each concern is derived from a weighted blend of 1–3 features.
Weights are intentionally explicit so they are easy to tune.
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("skinscope.scoring")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ConcernScore:
    name: str
    score: float          # 10–95
    severity: str         # "Low" | "Moderate" | "High"
    description: str
    breakdown: dict       # feature contributions for logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SCORE_MIN = 10.0
_SCORE_MAX = 95.0
_SCORE_RANGE = _SCORE_MAX - _SCORE_MIN


def _scale(value: float) -> float:
    return round(_SCORE_MIN + float(value) * _SCORE_RANGE, 1)


def _severity(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 65:
        return "Moderate"
    return "High"


def _blend(**kwargs: tuple[float, float]) -> tuple[float, dict]:
    """
    Weighted blend: pass keyword args as feature=(value, weight).
    Returns (blended_value [0,1], breakdown_dict).
    """
    total_weight = sum(w for _, w in kwargs.values())
    blended = sum(v * w for v, w in kwargs.values()) / total_weight
    breakdown = {k: round(v, 4) for k, (v, _) in kwargs.items()}
    return min(blended, 1.0), breakdown


# ---------------------------------------------------------------------------
# Concern calculators
# ---------------------------------------------------------------------------
def _acne(f: dict) -> ConcernScore:
    """
    Three-source acne signal:

    1. YOLO density       — direct lesion detection (best when it works)
    2. OpenCV clusters    — discrete small red blobs (works for mild/moderate)
    3. Texture-gated redness — redness × texture_variance
                             Rosacea = high redness + LOW texture (smooth flush)
                             Acne    = high redness + HIGH texture (bumpy lesions)
                             This is the fallback for severe/widespread acne where
                             relative detection fails because the whole face is red.

    Rosacea guard is now TEXTURE-based, not redness-level-based.
    """
    skintel_signal = f.get("skintel_acne", 0.0)
    redness        = f["redness"]
    texture        = f["texture_variance"]

    # Texture factor: bumpy skin (acne) vs smooth skin (flush/oily pores)
    texture_factor = float(np.clip(texture * 3.5, 0.0, 1.0))

    # Texture-gated redness fallback for severe/widespread acne where
    # the ViT model may underestimate due to the masked skin crop
    texture_redness = redness * texture_factor * 0.65

    # Primary: skintelligent ViT  |  Fallback: texture × redness
    combined = float(np.clip(max(skintel_signal, texture_redness), 0.0, 1.0))

    value, bd = _blend(
        combined=(combined,                  0.90),
        redness =(redness * texture_factor,  0.10),
    )
    score = _scale(value)
    return ConcernScore("Acne / Breakouts", score, _severity(score),
                        "Active lesions and breakout activity", bd)


def _redness(f: dict) -> ConcernScore:
    value, bd = _blend(
        redness  =(f["redness"],  0.75),
        oiliness =(f["oiliness"], 0.25),
    )
    score = _scale(value)
    return ConcernScore("Redness / Inflammation", score, _severity(score),
                        "Skin redness, inflammation, and reactive areas", bd)


def _dark_spots(f: dict) -> ConcernScore:
    value, bd = _blend(
        dark_spots   =(f["dark_spot_ratio"], 0.55),
        pigmentation =(f["pigmentation"],    0.45),
    )
    score = _scale(value)
    return ConcernScore("Dark Spots / Hyperpigmentation", score, _severity(score),
                        "Post-acne marks, sun spots, and uneven pigment", bd)


def _enlarged_pores(f: dict) -> ConcernScore:
    value, bd = _blend(
        sharpness=(f["effnet_sharpness"],   0.50),
        texture  =(f["texture_variance"],   0.50),
    )
    score = _scale(value)
    return ConcernScore("Enlarged Pores", score, _severity(score),
                        "Visible pore size and skin texture coarseness", bd)


def _wrinkles(f: dict) -> ConcernScore:
    value, bd = _blend(
        texture    =(f["texture_variance"],   0.55),
        complexity =(f["effnet_complexity"],  0.45),
    )
    score = _scale(value)
    return ConcernScore("Wrinkles / Fine Lines", score, _severity(score),
                        "Fine lines, expression lines, and surface texture", bd)


def _oiliness(f: dict) -> ConcernScore:
    value, bd = _blend(
        oiliness   =(f["oiliness"],    0.75),
        brightness =(f["brightness"],  0.25),
    )
    score = _scale(value)
    return ConcernScore("Excess Oiliness", score, _severity(score),
                        "Sebum production, shine, and greasy areas", bd)


def _dryness(f: dict) -> ConcernScore:
    """
    Proper dryness signals:
    - saturation_inv : desaturated skin = dehydrated
    - flakiness      : micro-texture patches = flaky/dry surface
    - lab_uniformity : flat tonal range = moisture-depleted skin
    - inv_oiliness   : no shine = not oily (but only a secondary signal now)

    NOT based on texture_variance alone — rough ≠ dry.
    """
    value, bd = _blend(
        saturation_inv  =(f["saturation_inv"],   0.40),
        flakiness       =(f["flakiness"],        0.30),
        lab_uniformity  =(f["lab_uniformity"],   0.20),
        inv_oiliness    =(1.0 - f["oiliness"],   0.10),
    )
    score = _scale(value)
    return ConcernScore("Dryness / Dehydration", score, _severity(score),
                        "Lack of moisture, flakiness, and tightness", bd)


def _uneven_tone(f: dict) -> ConcernScore:
    value, bd = _blend(
        color_var    =(f["color_variance"], 0.55),
        pigmentation =(f["pigmentation"],   0.45),
    )
    score = _scale(value)
    return ConcernScore("Uneven Skin Tone", score, _severity(score),
                        "Colour inconsistencies and blotchy patches", bd)


def _dullness(f: dict) -> ConcernScore:
    value, bd = _blend(
        inv_brightness=(1.0 - f["brightness"], 0.60),
        inv_redness   =(1.0 - f["redness"],    0.40),
    )
    score = _scale(value)
    return ConcernScore("Dullness", score, _severity(score),
                        "Lack of radiance, vitality, and healthy glow", bd)


def _rough_texture(f: dict) -> ConcernScore:
    value, bd = _blend(
        texture    =(f["texture_variance"],  0.55),
        complexity =(f["effnet_complexity"], 0.45),
    )
    score = _scale(value)
    return ConcernScore("Rough Texture", score, _severity(score),
                        "Surface roughness, bumps, and uneven skin surface", bd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calculate_concerns(features: dict) -> list[ConcernScore]:
    """
    Accepts the merged feature dict from feature_extraction + model_inference.
    Returns a list of 10 ConcernScore objects, ordered by score (desc).
    """
    calculators = [
        _acne, _redness, _dark_spots, _enlarged_pores, _wrinkles,
        _oiliness, _dryness, _uneven_tone, _dullness, _rough_texture,
    ]
    concerns = [fn(features) for fn in calculators]
    concerns.sort(key=lambda c: c.score, reverse=True)

    # ── Terminal score breakdown ─────────────────────────────────────────────
    log.info("─" * 60)
    log.info("SCORING BREAKDOWN")
    log.info("─" * 60)
    for c in concerns:
        bar = "█" * int((c.score - 10) / 85 * 20)
        log.info(f"  {c.name:<35} {c.score:>5.1f}/95  [{c.severity:<8}]  {bar}")
        for feat, val in c.breakdown.items():
            log.info(f"      └─ {feat:<28} = {val:.4f}")
    log.info("─" * 60)

    return concerns
