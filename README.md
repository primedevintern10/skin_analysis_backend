# SkinScope — Backend

FastAPI backend for the SkinScope AI skin analysis pipeline.

## Stack
- **FastAPI** + Uvicorn
- **MediaPipe** — face detection + landmark masking
- **YOLOv11** — acne lesion detection
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
| `yolov11s_acne.pt` | KhaMinh/yolov11s-acne-detection (HF) | ~19MB |
| `skintelligent-acne` | imfarzanansari/skintelligent-acne (HF) | ~330MB |
| `efficientnet_b0` | timm (auto) | ~21MB |

## Run

```bash
python run.py
# API available at http://localhost:8000
```

## Endpoints
| Method | Path | Description |
|---|---|---|
| `POST` | `/detect` | Face detection only |
| `POST` | `/analyze` | Full skin analysis pipeline |
