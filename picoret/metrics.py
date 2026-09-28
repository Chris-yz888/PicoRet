from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


def _ssim_channel(first: np.ndarray, second: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    first, second = first.astype(np.float64), second.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    mean_first = cv2.filter2D(first, -1, window)[5:-5, 5:-5]
    mean_second = cv2.filter2D(second, -1, window)[5:-5, 5:-5]
    var_first = cv2.filter2D(first**2, -1, window)[5:-5, 5:-5] - mean_first**2
    var_second = cv2.filter2D(second**2, -1, window)[5:-5, 5:-5] - mean_second**2
    covariance = cv2.filter2D(first * second, -1, window)[5:-5, 5:-5] - mean_first * mean_second
    score = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
        (mean_first**2 + mean_second**2 + c1) * (var_first + var_second + c2)
    )
    return float(score.mean())


def paired_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0, 1)
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference**2))
    return {
        "psnr": -10 * math.log10(max(mse, 1e-12)),
        "ssim": float(np.mean([_ssim_channel(prediction[..., i], target[..., i]) for i in range(3)])),
        "delta_e00": float(deltaE_ciede2000(rgb2lab(prediction), rgb2lab(target)).mean()),
    }
