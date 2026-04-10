"""
ML model inference — YOLOv11 (acne detection) + EfficientNet-B0 (texture features).

Models are loaded once at startup as module-level singletons.
run_parallel_inference() runs both models concurrently via ThreadPoolExecutor
so total latency ≈ max(yolo_time, effnet_time) instead of their sum.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Lazy singletons with thread-safe initialisation
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_yolo: Any = None
_effnet: Any = None
_effnet_transforms: Any = None
_skintel: Any = None           # imfarzanansari/skintelligent-acne ViT
_skintel_processor: Any = None
_models_available = {"yolo": False, "effnet": False, "skintel": False}

# skintelligent-acne label → severity score (0=clear … 1=severe)
# Labels are resolved at runtime from model.config.id2label so any
# naming convention is handled automatically.
_SKINTEL_SEVERITY_BY_INDEX = [0.00, 0.25, 0.50, 0.75, 1.00]

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def _load_yolo() -> None:
    global _yolo
    try:
        from ultralytics import YOLO
        model_path = MODELS_DIR / "yolov11s_acne.pt"
        if not model_path.exists():
            print("Downloading YOLOv11 acne model from HuggingFace …")
            downloaded = hf_hub_download(
                repo_id="KhaMinh/yolov11s-acne-detection",
                filename="best.pt",
                local_dir=str(MODELS_DIR),
            )
            Path(downloaded).rename(model_path)
        _yolo = YOLO(str(model_path))
        _models_available["yolo"] = True
        print("YOLOv11 acne model ready.")
    except Exception as e:
        print(f"[WARN] YOLOv11 unavailable: {e}. Acne detection will be skipped.")


def _load_efficientnet() -> None:
    global _effnet, _effnet_transforms
    try:
        import timm
        import torch
        from torchvision import transforms

        _effnet = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,          # feature extractor — no classifier head
        )
        _effnet.eval()

        _effnet_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        _models_available["effnet"] = True
        print("EfficientNet-B0 ready.")
    except Exception as e:
        print(f"[WARN] EfficientNet unavailable: {e}. Texture ML features will be skipped.")


def _load_skintelligent() -> None:
    global _skintel, _skintel_processor
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_id = "imfarzanansari/skintelligent-acne"
        print("Loading skintelligent-acne ViT model …")
        _skintel_processor = AutoImageProcessor.from_pretrained(model_id)
        _skintel = AutoModelForImageClassification.from_pretrained(model_id)
        _skintel.eval()
        _models_available["skintel"] = True
        print("skintelligent-acne ready.")
    except Exception as e:
        print(f"[WARN] skintelligent-acne unavailable: {e}.")


def load_all_models() -> None:
    """Call once at application startup to pre-warm all models."""
    with _lock:
        t1 = threading.Thread(target=_load_yolo,            daemon=True)
        t2 = threading.Thread(target=_load_efficientnet,    daemon=True)
        t3 = threading.Thread(target=_load_skintelligent,   daemon=True)
        t1.start(); t2.start(); t3.start()
        t1.join();  t2.join();  t3.join()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _crop_to_face(image_rgb: np.ndarray, skin_crop_rgb: np.ndarray) -> np.ndarray:
    """Extract tight bounding box of the face from the skin crop."""
    mask = np.any(skin_crop_rgb > 0, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return image_rgb
    pad = 10
    h, w = image_rgb.shape[:2]
    y1 = max(rows[0] - pad, 0)
    y2 = min(rows[-1] + pad, h)
    x1 = max(cols[0] - pad, 0)
    x2 = min(cols[-1] + pad, w)
    return image_rgb[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Individual inference functions
# ---------------------------------------------------------------------------
def _run_yolo(image_rgb: np.ndarray, skin_crop_rgb: np.ndarray, skin_area_px: int) -> dict[str, float]:
    """
    Returns:
        acne_count       — number of detected lesions
        acne_density     — lesions per 10 000 skin pixels (normalised 0-1)
        acne_confidence  — mean detection confidence (0-1)
    """
    if not _models_available["yolo"]:
        return {"acne_count": 0.0, "acne_density": 0.0, "acne_confidence": 0.0}

    # Crop to face region and upscale for small images so lesions are detectable.
    # On a 474×315px image, a papule is ~5px — below YOLO's effective resolution.
    face_img = _crop_to_face(image_rgb, skin_crop_rgb)
    h_f, w_f = face_img.shape[:2]
    if max(h_f, w_f) < 640:
        scale = 640 / max(h_f, w_f)
        face_img = cv2.resize(face_img, (int(w_f * scale), int(h_f * scale)),
                              interpolation=cv2.INTER_LINEAR)

    results = _yolo.predict(face_img, verbose=False, conf=0.10)[0]
    boxes = results.boxes

    if len(boxes) == 0:
        return {"acne_count": 0.0, "acne_density": 0.0, "acne_confidence": 0.0}

    confidences = boxes.conf.cpu().numpy()
    classes     = boxes.cls.cpu().numpy().astype(int)

    SEVERITY_WEIGHTS = {
        0: 0.5,   # Blackhead
        1: 1.0,   # Conglobata
        2: 1.0,   # Cystic
        3: 0.3,   # Flat_wart
        4: 0.6,   # Folliculitis
        5: 0.8,   # Keloid
        6: 0.3,   # Milium
        7: 0.7,   # Papular
        8: 0.8,   # Purulent
        9: 0.5,   # Scars
        10: 0.3,  # Syringoma
        11: 0.5,  # Whitehead
    }

    # Weight each detection by BOTH severity AND confidence.
    # Low-confidence detections (rosacea false positives at ~0.10-0.15)
    # contribute far less than genuine high-confidence lesions (0.4+).
    weighted_count = sum(
        SEVERITY_WEIGHTS.get(c, 0.5) * float(conf)
        for c, conf in zip(classes, confidences)
    )

    # Log detected classes to terminal for debugging
    import logging
    log = logging.getLogger("skinscope.yolo")
    class_names = {0:"Blackhead",1:"Conglobata",2:"Cystic",3:"Flat_wart",
                   4:"Folliculitis",5:"Keloid",6:"Milium",7:"Papular",
                   8:"Purulent",9:"Scars",10:"Syringoma",11:"Whitehead"}
    detections = [f"{class_names[c]}({conf:.2f})" for c, conf in zip(classes, confidences)]
    log.info(f"  YOLO  {len(boxes)} detections: {', '.join(detections)}")
    log.info(f"  YOLO  weighted_count={weighted_count:.3f}  (conf×severity)")

    # Normalise: 5 confidence-weighted lesions = max density
    density_norm = float(np.clip(weighted_count / 5.0, 0.0, 1.0))

    return {
        "acne_count":      float(len(boxes)),
        "acne_density":    density_norm,
        "acne_confidence": float(confidences.mean()),
    }


def _run_skintelligent(skin_crop_rgb: np.ndarray) -> dict[str, float]:
    """
    imfarzanansari/skintelligent-acne ViT classifier.
    Returns skintel_acne in [0, 1]: 0=clear, 0.25=almost clear,
    0.5=mild, 0.75=moderate, 1.0=severe.
    """
    if not _models_available["skintel"]:
        return {"skintel_acne": 0.0}
    try:
        import torch
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(skin_crop_rgb)
        inputs  = _skintel_processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            logits = _skintel(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        # Parse severity from label name: "level -1"→0.00, "level 0"→0.15,
        # "level 1"→0.35, "level 2"→0.65, "level 3"→1.00
        LEVEL_SEVERITY = {-1: 0.00, 0: 0.15, 1: 0.35, 2: 0.65, 3: 1.00}

        score = 0.0
        for idx, label in _skintel.config.id2label.items():
            try:
                level_num = int(label.strip().split()[-1])   # "level 3" → 3
                severity  = LEVEL_SEVERITY.get(level_num, 0.0)
                score    += float(probs[idx]) * severity
            except (ValueError, IndexError):
                pass

        import logging
        logging.getLogger("skinscope.skintel").info(
            f"  skintelligent  score={score:.3f}  "
            f"top={_skintel.config.id2label[int(probs.argmax())]}({float(probs.max()):.2f})"
        )
        return {"skintel_acne": float(np.clip(score, 0.0, 1.0))}
    except Exception as e:
        return {"skintel_acne": 0.0}


def _run_efficientnet(skin_crop_rgb: np.ndarray) -> dict[str, float]:
    """
    Extracts a 1280-dim feature vector from EfficientNet-B0 and derives:
        effnet_complexity  — overall feature activation diversity
        effnet_sharpness   — peak high-frequency response (pores/wrinkles proxy)
    """
    if not _models_available["effnet"]:
        return {"effnet_complexity": 0.0, "effnet_sharpness": 0.0}

    import torch

    tensor = _effnet_transforms(skin_crop_rgb).unsqueeze(0)   # (1, 3, 224, 224)
    with torch.no_grad():
        features = _effnet(tensor).squeeze().numpy()           # (1280,)

    pos_features = features[features > 0]
    if len(pos_features) == 0:
        return {"effnet_complexity": 0.0, "effnet_sharpness": 0.0}

    complexity = float(np.clip(pos_features.std() / 2.0, 0.0, 1.0))
    sharpness  = float(np.clip(pos_features.max()  / 8.0, 0.0, 1.0))

    return {
        "effnet_complexity": complexity,
        "effnet_sharpness":  sharpness,
    }


# ---------------------------------------------------------------------------
# Public API — parallel execution
# ---------------------------------------------------------------------------
def run_parallel_inference(
    image_rgb: np.ndarray,
    skin_crop_rgb: np.ndarray,
) -> dict[str, float]:
    """
    Runs YOLOv11 and EfficientNet concurrently.
    Total wall time ≈ max(yolo_latency, effnet_latency).

    Returns merged dict of all ML features.
    """
    skin_area_px = int(np.any(skin_crop_rgb > 0, axis=2).sum())

    futures: dict = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures["yolo"]    = pool.submit(_run_yolo,           image_rgb, skin_crop_rgb, skin_area_px)
        futures["effnet"]  = pool.submit(_run_efficientnet,   skin_crop_rgb)
        futures["skintel"] = pool.submit(_run_skintelligent,  skin_crop_rgb)

    defaults = {
        "yolo":    {"acne_count": 0.0, "acne_density": 0.0, "acne_confidence": 0.0},
        "effnet":  {"effnet_complexity": 0.0, "effnet_sharpness": 0.0},
        "skintel": {"skintel_acne": 0.0},
    }
    results: dict[str, float] = {}
    for key, future in futures.items():
        try:
            results.update(future.result())
        except Exception as e:
            print(f"[WARN] {key} inference failed: {e}")
            results.update(defaults[key])

    return results
