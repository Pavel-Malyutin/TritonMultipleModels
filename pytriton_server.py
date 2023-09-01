import logging

import numpy as np
import torch
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from nn.models import load_lprnet, load_stn, load_yolo
from nn.settings import settings
from nn.utils import prepare_recognition_input

logger = logging.getLogger("examples.mlp_random_tensorflow2.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

device = torch.device("cpu")

yolo = load_yolo(
    weights=settings.YOLO.WEIGHTS,
    confidence=settings.YOLO.CONFIDENCE,
    device=device,
)

stn = load_stn(weights=settings.STN.WEIGHTS, device=device)

lprn = load_lprnet(
    weights=settings.LPRNET.WEIGHTS,
    num_classes=settings.LPRNET.NUM_CLASSES,
    out_indices=settings.LPRNET.OUT_INDICES,
    device=device,
)


@batch
def _yolo(img):
    detection = yolo(img[0], size=settings.YOLO.PREDICT_SIZE)
    df_results = detection.pandas().xyxy[0]
    license_plate_batch = prepare_recognition_input(
        df_results, img[0], return_torch=True, device=device
    )
    np_array = license_plate_batch.detach().cpu().numpy()
    return [np.expand_dims(np_array, axis=0)]


@batch
def _stn(license_plate_batch):
    license_plate_batch = torch.tensor(license_plate_batch[0], dtype=torch.float32)
    transfer = stn(license_plate_batch)
    np_array = transfer.detach().cpu().numpy()
    return [np.expand_dims(np_array, axis=0)]


@batch
def _lprnet(transfer):
    transfer = torch.tensor(transfer[0], dtype=torch.float32)
    predictions = lprn(transfer)
    predictions = predictions.cpu().detach().numpy()
    return [np.expand_dims(predictions, axis=0)]


with Triton() as triton:
    logger.info("Loading Yolo model")
    triton.bind(
        model_name="Yolo",
        infer_func=_yolo,
        inputs=[
            Tensor(name="img", dtype=np.uint8, shape=(1080, 1920, 3)),
        ],
        outputs=[
            Tensor(name="license_plate_batch", dtype=np.float32, shape=(-1, -1, 24, 94)),
        ],
        config=ModelConfig(max_batch_size=6220800),
        strict=False
    )
    logger.info("Loading STN model")
    triton.bind(
        model_name="STN",
        infer_func=_stn,
        inputs=[
            Tensor(name="license_plate_batch", dtype=np.float32, shape=(-1, -1, 24, 94)),
        ],
        outputs=[
            Tensor(name="transfer", dtype=np.float32, shape=(-1, -1, 24, 94)),
        ],
        config=ModelConfig(max_batch_size=13536),
        strict=False,
    )
    logger.info("Loading LPR model")
    triton.bind(
        model_name="LPR",
        infer_func=_lprnet,
        inputs=[
            Tensor(name="transfer", dtype=np.float32, shape=(-1, -1, 24, 94)),
        ],
        outputs=[
            Tensor(name="predictions", dtype=np.float32, shape=(-1, 23, 18)),
        ],
        config=ModelConfig(max_batch_size=13536),
        strict=False,
    )
    triton.serve()
