"""Compact single-GPU PicoRet training entry point."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from .io import image_index, read_rgb, tensor_to_rgb
from .metrics import paired_metrics
from .model import PicoRet


class PairedDataset(Dataset):
    def __init__(self, low: str, target: str, crop: int | None, split: str | None = None):
        low_index, target_index = image_index(low), image_index(target)
        if split:
            with open(split, newline="", encoding="utf-8") as handle:
                names = [row["id"] for row in csv.DictReader(handle)]
        else:
            names = sorted(set(low_index) & set(target_index))
        self.pairs = [(low_index[name], target_index[name], name) for name in names]
        self.crop = crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        low_path, target_path, identifier = self.pairs[index]
        low, target = read_rgb(low_path), read_rgb(target_path)
        if low.shape != target.shape:
            raise ValueError(f"Pair shape mismatch: {identifier}")
        if self.crop:
            height, width = low.shape[-2:]
            pad_h, pad_w = max(0, self.crop - height), max(0, self.crop - width)
            if pad_h or pad_w:
                low = F.pad(low, (0, pad_w, 0, pad_h), mode="replicate")
                target = F.pad(target, (0, pad_w, 0, pad_h), mode="replicate")
            height, width = low.shape[-2:]
            top = random.randint(0, height - self.crop)
            left = random.randint(0, width - self.crop)
            low = low[:, top : top + self.crop, left : left + self.crop]
            target = target[:, top : top + self.crop, left : left + self.crop]
            if random.random() < 0.5:
                low, target = low.flip(-1), target.flip(-1)
            if random.random() < 0.5:
                low, target = low.flip(-2), target.flip(-2)
            if random.random() < 0.5:
                low, target = low.transpose(-1, -2), target.transpose(-1, -2)
        return low.contiguous(), target.contiguous(), identifier


class HybridLoss(nn.Module):
    def __init__(self, pixel: float = 1.0, perceptual: float = 0.01, ms_ssim_weight: float = 0.2):
        super().__init__()
        self.pixel_weight = pixel
        self.perceptual_weight = perceptual
        self.ms_ssim_weight = ms_ssim_weight
        self.vgg: nn.Module | None = None

    def _build_vgg(self, device: torch.device) -> nn.Module:
        from torchvision import models

        network = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        return network.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.pixel_weight * F.mse_loss(prediction, target)
        if self.perceptual_weight:
            if self.vgg is None:
                self.vgg = self._build_vgg(prediction.device)
            loss = loss + self.perceptual_weight * F.mse_loss(self.vgg(prediction), self.vgg(target))
        if self.ms_ssim_weight:
            from pytorch_msssim import ms_ssim

            structural = 1 - ms_ssim(
                prediction, target, data_range=1.0, size_average=True, weights=[0.99, 0.009, 0.001], win_size=5
            )
            loss = loss + self.ms_ssim_weight * structural
        return loss


def learning_rate(step: int, initial: float = 4e-4) -> float:
    if step <= 40_000:
        minimum, position, period = 3e-4, step, 40_000
    else:
        minimum, position, period = 1e-6, step - 40_000, 60_000
    return minimum + 0.5 * (initial - minimum) * (1 + math.cos(math.pi * position / period))


def mixup(low: torch.Tensor, target: torch.Tensor, beta: float = 1.2):
    if random.randint(0, 1):
        return low, target
    coefficient = torch.distributions.Beta(beta, beta).sample().item()
    order = torch.randperm(low.shape[0], device=low.device)
    return coefficient * low + (1 - coefficient) * low[order], coefficient * target + (1 - coefficient) * target[order]


@torch.inference_mode()
def validate(model: PicoRet, dataset: PairedDataset, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    rows = []
    for low, target, _ in DataLoader(dataset, batch_size=1, shuffle=False):
        prediction = model(low.to(device))
        rows.append(paired_metrics(tensor_to_rgb(prediction), tensor_to_rgb(target)))
    model.train(was_training)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def save_checkpoint(path: Path, model: PicoRet, step: int, dataset: str, metrics: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model": "PicoRet",
            "parameters": model.parameter_count,
            "iteration": step,
            "dataset": dataset,
            "validation": metrics,
        },
        path,
    )


def train(config_path: str, output: str, device_name: str, steps: int | None = None) -> None:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(config.get("seed", 100))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    train_set = PairedDataset(
        config["train_low"], config["train_target"], int(config.get("crop_size", 256)), config.get("train_split")
    )
    val_set = PairedDataset(config["train_low"], config["train_target"], None, config.get("val_split"))
    loader = DataLoader(
        train_set,
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(config.get("workers", 4)),
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = PicoRet().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
    objective = HybridLoss().to(device)
    total_steps = int(config.get("total_steps", 100_000) if steps is None else steps)
    validation_interval = int(config.get("validation_interval", 5_000))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    iterator = iter(loader)
    best_psnr = -float("inf")

    for step in range(1, total_steps + 1):
        try:
            low, target, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            low, target, _ = next(iterator)
        low, target = low.to(device), target.to(device)
        low, target = mixup(low, target)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(low)
        loss = objective(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            record = {"step": step, "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"]}
            with (destination / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(record, flush=True)
        if step % validation_interval == 0 or step == total_steps:
            scores = validate(model, val_set, device)
            save_checkpoint(destination / f"step_{step:06d}.pth", model, step, config["dataset"], scores)
            if scores["psnr"] > best_psnr:
                best_psnr = scores["psnr"]
                save_checkpoint(destination / "best.pth", model, step, config["dataset"], scores)
            print({"validation": scores, "best_psnr": best_psnr}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, help="Optional short-run override for smoke testing")
    args = parser.parse_args()
    train(args.config, args.output, args.device, args.steps)


if __name__ == "__main__":
    main()
