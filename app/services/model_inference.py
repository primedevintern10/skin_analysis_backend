"""
ML model inference — EfficientNet-B0 (texture) + skintelligent-acne ViT (acne)
                   + Qwen3-VL-2B-Instruct (skin type + concern scoring).

Models are loaded once at startup as module-level singletons.
run_parallel_inference() runs all models concurrently via ThreadPoolExecutor.
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Lazy singletons with thread-safe initialisation
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_effnet: Any = None
_effnet_transforms: Any = None
_skintel: Any = None
_skintel_processor: Any = None
_qwen: Any = None
_qwen_processor: Any = None
_models_available = {"effnet": False, "skintel": False, "qwen_vlm": False}

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


def _load_qwen_vlm() -> None:
    global _qwen, _qwen_processor
    try:
        print("Loading Qwen3-VL-2B-Instruct VLM …")
        # Try Qwen2.5-VL class first (used by Qwen3-VL), fall back to Qwen2-VL
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            _qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-2B-Instruct",
                torch_dtype="auto",
                device_map="auto",
            )
        except (ImportError, AttributeError):
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            _qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-2B-Instruct",
                torch_dtype="auto",
                device_map="auto",
            )
        _qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        _qwen.eval()
        _models_available["qwen_vlm"] = True
        print("Qwen3-VL-2B-Instruct ready.")
    except Exception as e:
        print(f"[WARN] Qwen3-VL unavailable: {e}. VLM skin type/concerns will be skipped.")


def load_all_models() -> None:
    """Call once at application startup to pre-warm all models."""
    with _lock:
        t1 = threading.Thread(target=_load_efficientnet,  daemon=True)
        t2 = threading.Thread(target=_load_skintelligent, daemon=True)
        t3 = threading.Thread(target=_load_qwen_vlm,      daemon=True)
        t1.start(); t2.start(); t3.start()
        t1.join();  t2.join();  t3.join()


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
# VLM — Qwen3-VL-2B-Instruct
# ---------------------------------------------------------------------------

_VLM_CONCERN_NAMES = [
    "Acne / Breakouts",
    "Redness / Inflammation",
    "Dark Spots / Hyperpigmentation",
    "Enlarged Pores",
    "Wrinkles / Fine Lines",
    "Excess Oiliness",
    "Dryness / Dehydration",
    "Uneven Skin Tone",
    "Dullness",
    "Rough Texture",
]

_VLM_PROMPT = """You are a professional dermatologist analyzing a facial skin image.

Task 1 — Skin Type: Classify the skin type as EXACTLY one of:
  normal, oily, dry, combination, sensitive

Task 2 — Skin Concerns: Score each concern on a scale of 10 to 95
  (10 = completely clear, 95 = very severe).

Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.
Format:
{
  "skin_type": "<one of: normal | oily | dry | combination | sensitive>",
  "concerns": {
    "Acne / Breakouts": <int 10-95>,
    "Redness / Inflammation": <int 10-95>,
    "Dark Spots / Hyperpigmentation": <int 10-95>,
    "Enlarged Pores": <int 10-95>,
    "Wrinkles / Fine Lines": <int 10-95>,
    "Excess Oiliness": <int 10-95>,
    "Dryness / Dehydration": <int 10-95>,
    "Uneven Skin Tone": <int 10-95>,
    "Dullness": <int 10-95>,
    "Rough Texture": <int 10-95>
  }
}"""


def _parse_vlm_output(text: str) -> dict:
    """Extract JSON from VLM output robustly."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _vlm_severity(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 65:
        return "Moderate"
    return "High"


def run_vlm_analysis(skin_crop_rgb: np.ndarray) -> dict:
    """
    Run Qwen3-VL-2B-Instruct on the skin crop.
    Returns:
        skin_type: str
        vlm_concerns: list of dicts {name, score, severity}
        vlm_inference_ms: float
    """
    default = {
        "skin_type": "unknown",
        "vlm_concerns": [],
        "vlm_inference_ms": 0.0,
    }

    if not _models_available["qwen_vlm"]:
        return default

    try:
        import torch
        from PIL import Image as PILImage

        t0 = time.perf_counter()

        pil_img = PILImage.fromarray(skin_crop_rgb)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": _VLM_PROMPT},
                ],
            }
        ]

        # Build inputs using the processor's chat template
        text_input = _qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # process_vision_info is part of qwen_vl_utils; fall back to manual if missing
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = _qwen_processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            inputs = _qwen_processor(
                text=[text_input],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(_qwen.device)

        with torch.no_grad():
            output_ids = _qwen.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_text = _qwen_processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        vlm_inference_ms = (time.perf_counter() - t0) * 1000

        import logging
        logging.getLogger("skinscope.vlm").info(
            f"  Qwen3-VL raw output: {raw_text[:200]}"
        )

        parsed = _parse_vlm_output(raw_text)
        skin_type = parsed.get("skin_type", "unknown").strip().lower()
        if skin_type not in {"normal", "oily", "dry", "combination", "sensitive"}:
            skin_type = "unknown"

        raw_concerns = parsed.get("concerns", {})
        vlm_concerns = []
        for name in _VLM_CONCERN_NAMES:
            score = float(raw_concerns.get(name, 10))
            score = max(10.0, min(95.0, score))
            vlm_concerns.append({
                "name": name,
                "score": score,
                "severity": _vlm_severity(score),
            })

        return {
            "skin_type": skin_type,
            "vlm_concerns": vlm_concerns,
            "vlm_inference_ms": vlm_inference_ms,
        }

    except Exception as e:
        import logging
        logging.getLogger("skinscope.vlm").warning(f"  Qwen3-VL inference failed: {e}")
        return default


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
