import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_json(path: str) -> dict:
    with Path(path).open("r") as f:
        info = json.load(f)
    return info


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"{path} not found"

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(path: str, image: np.ndarray):
    folder = Path(path).parent
    if not folder.exists():
        raise FileNotFoundError(f"Parent folder not Found: {folder}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)



def mkdir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize(image: np.ndarray, target_size: Tuple[int, int], interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    image = cv2.resize(image, target_size, interpolation)
    return image
