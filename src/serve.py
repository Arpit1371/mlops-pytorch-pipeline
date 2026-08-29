import io
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model

CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/configs/training_config.yaml"))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

app = FastAPI(title="mlops-pytorch-pipeline serving")
_state = {"model": None, "device": None}


def _load_model() -> None:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device
    print(json.dumps({"event": "model_loaded", "checkpoint": str(CHECKPOINT_PATH)}), flush=True)


@app.on_event("startup")
def startup() -> None:
    try:
        _load_model()
    except Exception as exc:
        # Keep the process up so /health reports "not ready" instead of the
        # pod crash-looping before its startup logs are even visible.
        print(json.dumps({"event": "model_load_failed", "error": str(exc)}), flush=True)


@app.get("/health")
def health():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    tensor = _TRANSFORM(img).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    ranked = sorted(zip(CIFAR10_CLASSES, probs), key=lambda pair: -pair[1])
    return {"predictions": [{"class": cls, "probability": round(p, 6)} for cls, p in ranked]}
