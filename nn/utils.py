from pathlib import Path
from typing import Optional, Union, List

import cv2
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from torch import Tensor, device

from nn.settings import settings


def prepare_recognition_input(
    df_results: pd.DataFrame,
    image: np.ndarray,
    return_torch: bool = True,
    device: Optional[device] = None,
) -> Union[ndarray, Tensor]:
    cropped_images = []
    for row in range(df_results.shape[0]):
        height = df_results.iloc[row][3] - df_results.iloc[row][1]

        top_x = int(df_results.iloc[row][0])
        top_y = int(df_results.iloc[row][1] - height * 0.2)
        bottom_x = int(df_results.iloc[row][2])
        bottom_y = int(df_results.iloc[row][3] + height * 0.2)

        license_plate = image[top_y:bottom_y, top_x:bottom_x]
        license_plate = cv2.resize(
            license_plate, settings.LPRNET.IMG_SIZE, interpolation=cv2.INTER_CUBIC
        )
        license_plate = (
            np.transpose(np.float32(license_plate), (2, 0, 1)) - 127.5
        ) * 0.0078125
        cropped_images.append(license_plate)

    cropped_images = np.array(cropped_images)

    if return_torch:
        return torch.from_numpy(cropped_images).to(device)
    else:
        return cropped_images


def prepare_detection_input(image: Union[ndarray, str, Path]) -> ndarray:
    """Prepares input for detector. Operates with numpy or image path

    Arguments:
        image -- image or path to image

    Returns:
        input to detector
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    assert image.ndim == 3

    image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
    image = image[:, :, ::-1]
    return image

