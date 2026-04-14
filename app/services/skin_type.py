"""
Rule-based skin type classifier.

Derives skin type from OpenCV features already extracted by the pipeline.
No extra model or inference time required.

Rules (thresholds tuned to normalised 0–1 feature range):
  oily        : oiliness high, redness low
  dry         : flakiness high, oiliness low
  sensitive   : redness high (inflammation markers)
  combination : moderate oiliness with uneven texture/color variance
  normal      : all features in balanced range
"""

import logging

log = logging.getLogger("skinscope.skintype")

# Thresholds
_OILY_THRESH       = 0.30   # oiliness >= this → leaning oily
_DRY_OILY_MAX      = 0.15   # oiliness <= this for dry classification
_FLAKY_THRESH      = 0.40   # flakiness >= this → dry indicator
_REDNESS_SENSITIVE = 0.45   # redness >= this → sensitive
_COMBO_OILY        = 0.20   # moderate oiliness for combination
_COLOR_VAR_COMBO   = 0.45   # color variance for combination (uneven zones)


def classify_skin_type(features: dict[str, float]) -> str:
    """
    Classify skin type from extracted OpenCV features.

    Args:
        features: dict containing at minimum:
            oiliness, redness, flakiness, brightness,
            texture_variance, color_variance, saturation_inv

    Returns:
        One of: "oily" | "dry" | "sensitive" | "combination" | "normal"
    """
    oiliness      = features.get("oiliness",        0.0)
    redness       = features.get("redness",          0.0)
    flakiness     = features.get("flakiness",        0.0)
    color_var     = features.get("color_variance",   0.0)
    saturation_inv = features.get("saturation_inv",  0.0)

    # ── Sensitive: high redness is the dominant signal ────────────────────────
    if redness >= _REDNESS_SENSITIVE:
        skin_type = "sensitive"

    # ── Oily: high oiliness, redness not dominant ────────────────────────────
    elif oiliness >= _OILY_THRESH and redness < _REDNESS_SENSITIVE:
        skin_type = "oily"

    # ── Dry: low oiliness + high flakiness or high saturation drop ───────────
    elif oiliness <= _DRY_OILY_MAX and (flakiness >= _FLAKY_THRESH or saturation_inv >= 0.35):
        skin_type = "dry"

    # ── Combination: moderate oiliness + uneven color zones ──────────────────
    elif oiliness >= _COMBO_OILY and color_var >= _COLOR_VAR_COMBO:
        skin_type = "combination"

    # ── Normal: nothing stands out ────────────────────────────────────────────
    else:
        skin_type = "normal"

    log.info(
        f"  SKIN TYPE  {skin_type:<12}  "
        f"oiliness={oiliness:.3f}  redness={redness:.3f}  "
        f"flakiness={flakiness:.3f}  color_var={color_var:.3f}  "
        f"saturation_inv={saturation_inv:.3f}"
    )
    return skin_type
