import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
from pytriton.client import ModelClient

from nn.inference.decode import beam_decode
from nn.settings import settings
from nn.utils import prepare_detection_input

logger = logging.getLogger("Triton inference")
logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def call_server(img: np.array) -> list:
    with ModelClient("localhost", "Yolo") as client_y:
        with ModelClient("localhost", "STN") as client_stn:
            with ModelClient("localhost", "LPR") as client_lpr:
                license_plate_batch = client_y.infer_batch(img)
                transfer = client_stn.infer_batch(license_plate_batch['license_plate_batch'])
                predictions = client_lpr.infer_batch(transfer['transfer'])

                labels, log_likelihood, _ = beam_decode(predictions['predictions'][0], settings.VOCAB.VOCAB)
                return labels


def run(data_folder: str):
    images_path = Path(__file__).resolve().parents[0] / data_folder
    for image in os.listdir(images_path):

        img_array = prepare_detection_input(images_path / image)
        img = np.expand_dims(img_array, axis=0)
        labels = call_server(img)
        [logger.info(i) for i in labels if len(i) > 6]


if __name__ == '__main__':
    start = dt.datetime.now()
    run('cars')
    logger.info(f"Total time is {dt.datetime.now() - start}")
