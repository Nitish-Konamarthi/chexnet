# app.py
import io, collections, os, sys
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import uvicorn

# If DensenetModels.py is in the same folder, this will import:
try:
    from DensenetModels import DenseNet121
except Exception as e:
    raise RuntimeError("Cannot import DenseNet121 from DensenetModels.py: " + str(e))

app = FastAPI(title="CheXNet Prototype Inference")

#MODEL_PATH = os.environ.get("MODEL_PATH", "chexnet/models/m-25012018-123527.pth.tar")
MODEL_PATH = os.environ.get("MODEL_PATH", "chexnet/models/m-30012020-104001.pth.tar")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = collections.OrderedDict()
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    return new_state

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Put checkpoint at models/... or set MODEL_PATH env var.")

state_dict = load_state(MODEL_PATH)

NUM_CLASSES = 14
model = DenseNet121(NUM_CLASSES, True)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

CLASS_NAMES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
               'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
               'Fibrosis','Pleural_Thickening','Hernia']

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
])

class LabelProb(BaseModel):
    label: str
    prob: float

class PredictResponse(BaseModel):
    model_version: str
    predictions: List[LabelProb]
    notes: str

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "File must be an image")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(400, f"Unable to read image: {e}")

    arr = np.stack([np.array(img)]*3, axis=-1).astype(np.uint8)
    inp = preprocess(Image.fromarray(arr)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)
        probs = torch.sigmoid(out).squeeze(0).cpu().numpy()

    pairs = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)[:top_k]
    return PredictResponse(
        model_version=os.path.basename(MODEL_PATH),
        predictions=[LabelProb(label=p[0], prob=float(p[1])) for p in pairs],
        notes="Prototype: probabilistic outputs only; not a clinical diagnosis."
    )

@app.get("/health")
def health():
    return {"status":"ok", "device": str(DEVICE)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT",8080)), log_level="info")