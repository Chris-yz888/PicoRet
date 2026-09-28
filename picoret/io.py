from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_index(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = {path.stem: path for path in sorted(root.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS}
    if not result:
        raise FileNotFoundError(f"No images found in {root}")
    return result


def read_rgb(path: str | Path) -> Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def tensor_to_rgb(tensor: Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Expected one image")
        tensor = tensor[0]
    return tensor.detach().float().cpu().permute(1, 2, 0).numpy()


def save_rgb(path: str | Path, tensor: Tensor) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.rint(np.clip(tensor_to_rgb(tensor), 0, 1) * 255).astype(np.uint8)
    if not cv2.imwrite(str(destination), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
        raise OSError(destination)
