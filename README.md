---
title: SkinScope Backend
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# SkinScope — Backend

FastAPI backend for the SkinScope AI skin analysis pipeline.

## Live API

| | URL |
|---|---|
| **Interactive Docs (Swagger UI)** | https://primeintern10-skinscope-backend.hf.space/docs |
| **Base URL** | https://primeintern10-skinscope-backend.hf.space |

## Stack
- **FastAPI** + Uvicorn
- **MediaPipe** — face detection + landmark masking
- **skintelligent-acne** (ViT) — acne severity classification
- **EfficientNet-B0** — texture feature extraction
- **OpenCV** — photometric features (redness, oiliness, brightness, etc.)

## Setup

```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Models
Downloaded automatically on first run:
| Model | Source | Size |
|---|---|---|
| `face_landmarker.task` | Google MediaPipe | ~28MB |
| `face_detector.tflite` | Google MediaPipe | ~1MB |
| `skintelligent-acne` | imfarzanansari/skintelligent-acne (HF) | ~330MB |
| `efficientnet_b0` | timm (auto) | ~21MB |

## Run locally

```bash
python run.py
# API available at http://localhost:8000/docs
```

## Endpoints
| Method | Path | Description |
|---|---|---|
| `POST` | `/detect` | Face detection only |
| `POST` | `/analyze` | Full skin analysis pipeline |
