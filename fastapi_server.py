import json

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from nn.inference.decode import beam_decode
from nn.inference.predictor import Predictor
from nn.models import load_lprnet, load_stn, load_yolo
from nn.settings import settings

app = FastAPI()

device = torch.device("cpu")


class Image(BaseModel):
    data: str


yolo = load_yolo(
    weights=settings.YOLO.WEIGHTS,
    confidence=settings.YOLO.CONFIDENCE,
    device=device,
)
lprnet = load_lprnet(
    weights=settings.LPRNET.WEIGHTS,
    num_classes=settings.LPRNET.NUM_CLASSES,
    out_indices=settings.LPRNET.OUT_INDICES,
    device=device,
)
stn = load_stn(weights=settings.STN.WEIGHTS, device=device)

predictor = Predictor(
    yolo=yolo, lprn=lprnet, stn=stn, device=device, decode_fn=beam_decode
)


@app.post("/recognise")
async def recognise_array(image: Image) -> list:
    decoded_image = np.asarray(json.loads(image.data), dtype=np.uint8)
    results = predictor.predict(decoded_image)
    return [i for i in results if len(i) > 6]


if __name__ == '__main__':
    uvicorn.run("fastapi_server:app", host="localhost", port=8888, reload=True, log_level="debug")
