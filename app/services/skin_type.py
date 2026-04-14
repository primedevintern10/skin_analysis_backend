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
_OILY_SPECULAR        = 0.20   # oiliness (specular) >= this → oily
_OILY_BRIGHT_SAT      = 0.62   # brightness >= this → possible shine
_OILY_SAT_INV_MAX     = 0.25   # saturation_inv <= this → high saturation (oily)
_OILY_FLAKY_MAX       = 0.65   # flakiness must be below this for brightness-based oily
_DRY_OILY_MAX         = 0.15   # oiliness <= this for dry classification
_DRY_BRIGHT_MAX       = 0.60   # brightness must be below this to classify dry
_FLAKY_THRESH         = 0.40   # flakiness >= this → dry indicator
_REDNESS_SENSITIVE    = 0.45   # redness >= this (used with clusters)
_CLUSTERS_SENSITIVE   = 0.60   # local_redness_clusters >= this → true inflammation
_COMBO_OILY           = 0.18   # moderate oiliness for combination
_COLOR_VAR_COMBO      = 0.45   # color variance for combination (uneven zones)


def classify_skin_type(features: dict[str, float]) -> str:
    """
    Classify skin type from extracted OpenCV features.

    Args:
        features: dict containing at minimum:
            oiliness, redness, flakiness, brightness,
            texture_variance, color_variance, saturation_inv,
            local_redness_clusters

    Returns:
        One of: "oily" | "dry" | "sensitive" | "combination" | "normal"
    """
    oiliness         = features.get("oiliness",               0.0)
    redness          = features.get("redness",                 0.0)
    flakiness        = features.get("flakiness",               0.0)
    brightness       = features.get("brightness",              0.0)
    color_var        = features.get("color_variance",          0.0)
    saturation_inv   = features.get("saturation_inv",          0.0)
    clusters         = features.get("local_redness_clusters",  0.0)

    # ── Oily: high specular highlights OR bright + high saturation + not flaky ─
    # Second condition catches diffuse shine that the specular detector misses
    oily_by_specular = oiliness >= _OILY_SPECULAR
    oily_by_shine    = (brightness >= _OILY_BRIGHT_SAT
                        and saturation_inv <= _OILY_SAT_INV_MAX
                        and flakiness < _OILY_FLAKY_MAX)
    if (oily_by_specular or oily_by_shine) and redness < _REDNESS_SENSITIVE:
        skin_type = "oily"

    # ── Sensitive: requires BOTH high redness AND inflamed clusters ───────────
    # Rules out warm skin tones which have redness but no actual clusters
    elif redness >= _REDNESS_SENSITIVE and clusters >= _CLUSTERS_SENSITIVE:
        skin_type = "sensitive"

    # ── Dry: low oiliness + high flakiness/saturation drop + not bright ───────
    # Brightness guard prevents bright oily faces from being misclassified
    elif (oiliness <= _DRY_OILY_MAX
          and brightness < _DRY_BRIGHT_MAX
          and (flakiness >= _FLAKY_THRESH or saturation_inv >= 0.35)):
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
