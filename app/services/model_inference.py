"""
ML model inference — EfficientNet-B0 (texture) + skintelligent-acne ViT (acne).

Models are loaded once at startup as module-level singletons.
run_parallel_inference() runs both models concurrently via ThreadPoolExecutor.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import numpy as np

# ---------------------------------------------------------------------------
# Lazy singletons with thread-safe initialisation
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_effnet: Any = None
_effnet_transforms: Any = None
_skintel: Any = None
_skintel_processor: Any = None
_models_available = {"effnet": False, "skintel": False}

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def _load_efficientnet() -> None:
    global _effnet, _effnet_transforms
    try:
        import timm
        import torch
        from torchvision import transforms

        _effnet = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
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
        t1 = threading.Thread(target=_load_efficientnet,  daemon=True)
        t2 = threading.Thread(target=_load_skintelligent, daemon=True)
        t1.start(); t2.start()
        t1.join();  t2.join()


# ---------------------------------------------------------------------------
# Individual inference functions
# ---------------------------------------------------------------------------
def _run_skintelligent(skin_crop_rgb: np.ndarray) -> dict[str, float]:
    """
    imfarzanansari/skintelligent-acne ViT classifier.
    Returns skintel_acne in [0, 1]: 0=clear … 1=severe.
    Labels: "level -1" (clear) → "level 3" (severe).
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

        LEVEL_SEVERITY = {-1: 0.00, 0: 0.15, 1: 0.35, 2: 0.65, 3: 1.00}

        score = 0.0
        for idx, label in _skintel.config.id2label.items():
            try:
                level_num = int(label.strip().split()[-1])
                score    += float(probs[idx]) * LEVEL_SEVERITY.get(level_num, 0.0)
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
    Extracts a 1280-dim feature vector from EfficientNet-B0:
        effnet_complexity  — texture diversity
        effnet_sharpness   — peak high-frequency response (pores/wrinkles proxy)
    """
    if not _models_available["effnet"]:
        return {"effnet_complexity": 0.0, "effnet_sharpness": 0.0}

    import torch

    tensor = _effnet_transforms(skin_crop_rgb).unsqueeze(0)
    with torch.no_grad():
        features = _effnet(tensor).squeeze().numpy()

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
    Runs skintelligent-acne and EfficientNet concurrently.
    Total wall time ≈ max(skintel_latency, effnet_latency).
    """
    futures: dict = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures["effnet"]  = pool.submit(_run_efficientnet,  skin_crop_rgb)
        futures["skintel"] = pool.submit(_run_skintelligent, skin_crop_rgb)

    defaults = {
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
