"""
OpenCV-based skin feature extraction.

All returned values are normalised to [0, 1] so the scoring engine
can treat them uniformly regardless of image resolution.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _skin_pixels(image_rgb: np.ndarray) -> np.ndarray:
    """Return only the non-black (skin) pixels as (N, 3) array."""
    mask = np.any(image_rgb > 0, axis=2)
    return image_rgb[mask]


def _skin_mask(image_rgb: np.ndarray) -> np.ndarray:
    return np.any(image_rgb > 0, axis=2).astype(np.uint8)


def _safe_norm(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------
def compute_redness(image_rgb: np.ndarray) -> float:
    """Proportion of skin pixels in the red HSV zone."""
    pixels = _skin_pixels(image_rgb)
    if len(pixels) == 0:
        return 0.0
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    red_mask = ((h <= 10) | (h >= 160)) & (s >= 50) & (v >= 50)
    return float(red_mask.sum() / len(pixels))


def compute_oiliness(image_rgb: np.ndarray) -> float:
    """Ratio of specular-highlight pixels (very bright, low saturation)."""
    pixels = _skin_pixels(image_rgb)
    if len(pixels) == 0:
        return 0.0
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    s, v = hsv[:, 1], hsv[:, 2]
    specular_mask = (v >= 220) & (s <= 40)
    return float(specular_mask.sum() / len(pixels))


def compute_brightness(image_rgb: np.ndarray) -> float:
    """Mean LAB L* of skin pixels, normalised to [0, 1]."""
    pixels = _skin_pixels(image_rgb)
    if len(pixels) == 0:
        return 0.0
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return float(lab[:, 0].mean() / 255.0)


def compute_texture_variance(image_rgb: np.ndarray) -> float:
    """Normalised Laplacian variance — high → rough/porous skin."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = _skin_mask(image_rgb)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_masked = lap[mask == 1]
    if len(lap_masked) == 0:
        return 0.0
    return _safe_norm(float(lap_masked.var()), 0.0, 3000.0)


def compute_color_variance(image_rgb: np.ndarray) -> float:
    """Mean std-dev across RGB channels of skin pixels — high → blotchy tone."""
    pixels = _skin_pixels(image_rgb).astype(np.float32)
    if len(pixels) == 0:
        return 0.0
    return _safe_norm(float(pixels.std(axis=0).mean()), 0.0, 80.0)


def compute_pigmentation(image_rgb: np.ndarray) -> float:
    """LAB b* std-dev — high → dark spots / uneven pigment."""
    pixels = _skin_pixels(image_rgb)
    if len(pixels) == 0:
        return 0.0
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return _safe_norm(float(lab[:, 2].std()), 0.0, 25.0)


def compute_dark_spot_ratio(image_rgb: np.ndarray) -> float:
    """Proportion of skin pixels significantly darker than the median L*."""
    pixels = _skin_pixels(image_rgb).astype(np.float32)
    if len(pixels) == 0:
        return 0.0
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_RGB2LAB).reshape(-1, 3)
    l = lab[:, 0].astype(np.float32)
    return float((l < float(np.median(l)) * 0.72).sum() / len(pixels))


# ---------------------------------------------------------------------------
# NEW: Acne — local redness cluster detection
# ---------------------------------------------------------------------------
def compute_local_redness_clusters(image_rgb: np.ndarray) -> float:
    """
    Detect spatially concentrated inflamed lesion blobs.

    Uses skin-tone-RELATIVE detection instead of absolute HSV thresholds.
    This correctly catches pink/salmon acne on Asian skin that absolute
    red-zone detection misses entirely.

    Method:
      1. Compute median skin hue and saturation as the "baseline"
      2. Find pixels that are MORE red/pink than the skin baseline
      3. Cluster those pixels into blobs — each blob = a potential lesion
    """
    mask = _skin_mask(image_rgb)
    if mask.sum() == 0:
        return 0.0

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch = hsv[:, :, 0]   # 0-180
    s_ch = hsv[:, :, 1]   # 0-255
    v_ch = hsv[:, :, 2]   # 0-255

    # Baseline skin hue and saturation from median of skin pixels
    skin_h = h_ch[mask == 1]
    skin_s = s_ch[mask == 1]
    baseline_h = float(np.median(skin_h))
    baseline_s = float(np.median(skin_s))

    # A pixel is an "inflamed" candidate if:
    #   - Its hue is shifted toward red/pink relative to baseline
    #     (lower hue = more red; most skin sits around hue 10-25)
    #   - Its saturation is higher than baseline (inflammation = more vivid)
    #   - It is not too dark (not a shadow) and not too bright (not specular)
    hue_diff  = baseline_h - h_ch           # positive = redder than baseline
    sat_diff  = s_ch - baseline_s           # positive = more saturated than baseline

    inflamed = (
        (hue_diff > 6)        # stricter: 6° redder than skin tone
        & (sat_diff > 20)     # stricter: clearly more saturated than baseline
        & (v_ch > 60)
        & (v_ch < 230)
        & (mask == 1)
    ).astype(np.uint8) * 255

    # Small kernel: prevents merging distant pixels into one large blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(inflamed, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed)

    # Acne lesions: small discrete blobs (30–400px).
    # < 30px = noise.  > 400px = diffuse flush / rosacea patch.
    valid_blobs = sum(
        1 for i in range(1, num_labels)
        if 30 <= stats[i, cv2.CC_STAT_AREA] <= 400
    )

    # Normalise: 8+ lesion blobs = max score
    return float(np.clip(valid_blobs / 8.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# NEW: Dryness — saturation drop + flakiness + LAB flatness
# ---------------------------------------------------------------------------
def compute_saturation_level(image_rgb: np.ndarray) -> float:
    """
    Skin-tone-RELATIVE saturation drop — measures how much saturation
    varies WITHIN the face rather than the absolute level.

    Why relative: Asian, darker, and lighter skin tones all have
    different baseline saturations. Absolute desaturation falsely flags
    naturally lower-saturation skin tones as "dry".

    Signal: coefficient of variation of saturation across skin pixels.
    Low CV  = uniform saturation = well-hydrated, consistent moisture.
    High CV = patchy saturation = localised dry/dehydrated zones.

    Additionally penalises if many patches fall below 60% of the median
    saturation (genuine desaturated patches = dry areas).
    """
    pixels = _skin_pixels(image_rgb)
    if len(pixels) == 0:
        return 0.0

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    s = hsv[:, 1].astype(np.float32)

    median_s = float(np.median(s))
    if median_s < 1:
        return 0.0

    # CV of saturation — high = uneven moisture distribution
    cv_sat = float(s.std() / (median_s + 1e-6))
    cv_norm = _safe_norm(cv_sat, 0.0, 1.2)

    # Proportion of pixels with saturation < 60% of median (genuinely dry patches)
    dry_patch_ratio = float((s < median_s * 0.60).sum() / len(s))

    return float(np.clip(cv_norm * 0.5 + dry_patch_ratio * 0.5, 0.0, 1.0))


def compute_flakiness(image_rgb: np.ndarray) -> float:
    """
    Detects fine high-frequency micro-texture patches — characteristic of
    flaky/peeling dry skin.  Uses variance of local Laplacian in small windows.
    High → flaky / dry surface.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = _skin_mask(image_rgb)

    # Local std in 7x7 windows — captures micro-texture
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    local_var = cv2.absdiff(gray, blur)
    local_var[mask == 0] = 0

    skin_px = local_var[mask == 1]
    if len(skin_px) == 0:
        return 0.0

    # High variance patches (> 75th percentile) = potential flaking
    threshold = float(np.percentile(skin_px, 75))
    flaky_ratio = float((skin_px > threshold * 1.5).sum() / len(skin_px))
    return float(np.clip(flaky_ratio * 3.0, 0.0, 1.0))   # scale up — rare signal


def compute_lab_uniformity(image_rgb: np.ndarray) -> float:
    """
    Detects patchy brightness variation — a sign of dehydration/dryness.
    Splits the skin region into 16×16 blocks and measures how much the
    mean L* differs between blocks (high inter-block variance = patchy).

    This is skin-tone-independent: it measures VARIATION, not absolute level.
    """
    pixels = _skin_pixels(image_rgb)
    if len(pixels) == 0:
        return 0.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = _skin_mask(image_rgb)

    block = 16
    h, w = gray.shape
    block_means = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            region_mask = mask[y:y+block, x:x+block]
            if region_mask.sum() < block * block * 0.5:
                continue   # skip mostly-non-skin blocks
            region = gray[y:y+block, x:x+block]
            block_means.append(float(region[region_mask == 1].mean()))

    if len(block_means) < 4:
        return 0.0

    block_cv = float(np.std(block_means) / (np.mean(block_means) + 1e-6))
    # High CV between blocks = patchy brightness = dehydrated
    # Range calibrated: normal faces sit 0.05–0.20, patchy/dry 0.20–0.50
    return _safe_norm(block_cv, 0.05, 0.40)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_opencv_features(skin_crop: np.ndarray) -> dict[str, float]:
    """
    Run all OpenCV feature functions on the masked skin crop.
    Returns a dict with all keys normalised to [0, 1].
    """
    return {
        # Core features
        "redness":                  compute_redness(skin_crop),
        "oiliness":                 compute_oiliness(skin_crop),
        "brightness":               compute_brightness(skin_crop),
        "texture_variance":         compute_texture_variance(skin_crop),
        "color_variance":           compute_color_variance(skin_crop),
        "pigmentation":             compute_pigmentation(skin_crop),
        "dark_spot_ratio":          compute_dark_spot_ratio(skin_crop),
        # Acne-specific
        "local_redness_clusters":   compute_local_redness_clusters(skin_crop),
        # Dryness-specific
        "saturation_inv":           compute_saturation_level(skin_crop),
        "flakiness":                compute_flakiness(skin_crop),
        "lab_uniformity":           compute_lab_uniformity(skin_crop),
    }
