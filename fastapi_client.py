import datetime as dt
import json
import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm

from nn.inference.predictor import prepare_detection_input

logger = logging.getLogger("Python inference")
logging.basicConfig(level=logging.INFO, format="\n%(asctime)s - %(levelname)s - %(name)s: %(message)s")


if __name__ == '__main__':
    images_path = Path(__file__).resolve().parents[0] / 'cars'
    start = dt.datetime.now()
    for image in tqdm(os.listdir(images_path)):
        data = prepare_detection_input(images_path / image)
        encoded_img = json.dumps(data.tolist())
        response = requests.post("http://localhost:8888/recognise", json={'data': encoded_img})
        logger.info(f"Response is {response.json()}")
    logger.info(f"Total time is {dt.datetime.now() - start}")
