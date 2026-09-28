from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .io import image_index, read_rgb, save_rgb, tensor_to_rgb
from .metrics import paired_metrics
from .model import load_model


@torch.inference_mode()
def run(checkpoint: str, inputs: str, output: str, targets: str | None, device: str) -> dict:
    model = load_model(checkpoint, device)
    source = image_index(inputs)
    ground_truth = image_index(targets) if targets else None
    output_root = Path(output)
    rows = []
    for identifier, path in source.items():
        low = read_rgb(path)[None].to(device)
        prediction = model(low)
        save_rgb(output_root / "enhanced" / f"{identifier}.png", prediction)
        row: dict[str, str | float] = {"id": identifier}
        if ground_truth is not None:
            if identifier not in ground_truth:
                raise FileNotFoundError(f"Missing target for {identifier}")
            target = tensor_to_rgb(read_rgb(ground_truth[identifier]))
            row.update(paired_metrics(tensor_to_rgb(prediction), target))
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in rows[0] if key != "id"]
    summary = {
        "images": len(rows),
        "mean": {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric},
        "model": "PicoRet",
        "parameters": model.parameter_count,
        "protocol": "full-resolution clipped-float RGB; no TTA; no target-mean correction",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.input, args.output, args.target, args.device), indent=2))


if __name__ == "__main__":
    main()
