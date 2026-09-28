# PicoRet

PicoRet is an ultra-tiny low-light image enhancer with **817 trainable
parameters**. It combines compact rank-2 prior filtering, a five-channel
dual-stream representation, illumination-conditioned color exchange, coarse
illumination context, and product-plus-direct residual prediction.

This repository is intentionally small: it contains the standalone model,
single-GPU training and inference code, dataset download helpers, fixed LOL-v1
split files, and compact pretrained weights. It does not depend on BasicSR.

## Where everything is

| Item | Location | Description |
|---|---|---|
| Model code | `picoret/model.py` | Standalone 817-parameter `PicoRet` architecture and checkpoint loader |
| Training code | `picoret/train.py` | Single-GPU training, validation, checkpoint selection and hybrid loss |
| Inference code | `picoret/infer.py` | Full-resolution inference and paired evaluation |
| Pretrained weights | `weights/` | LOL-v1 and LOL-v2 Synthetic checkpoints plus SHA-256 hashes |
| Training configs | `configs/` | Dataset paths, split manifests and training-control values |
| Dataset splits | `splits/` | Fixed internal training/validation manifests |
| Dataset helper | `scripts/download_datasets.py` | Downloads public paired and no-reference datasets |

## Architecture: Know, Form, Correct

The source code follows the same three-stage organization as the paper:

1. **Know** constructs four illumination priors and five chromatic priors from
   the input image. Ratio-based chromatic priors are stabilized near black.
2. **Form** uses pointwise projection and rank-2 spatial guidance to form two
   compact, full-resolution five-channel feature streams.
3. **Correct** applies coarse illumination context and one-way
   illumination-to-color conditioning, then predicts a structured correction
   and a direct RGB residual:

   ```text
   enhanced = input + delta_luma * delta_color + delta_direct
   ```

## Pretrained weights

The compact checkpoints in `weights/` are provided to verify installation,
inference and evaluation. They are earlier reproducible checkpoints and are
**not the final checkpoints used for the paper tables**. Final paper weights
will be added after the evaluation protocol and release artifacts are frozen.

## Installation

```bash
conda create -n picoret python=3.10 -y
conda activate picoret
# Install the PyTorch build appropriate for your CUDA version first.
pip install -e .
```

## Inference and paired evaluation

Verify the released checkpoints from the repository root:

```bash
sha256sum -c weights/SHA256SUMS.txt
```

```bash
python -m picoret.infer \
  --checkpoint weights/picoret_lolv1.pth \
  --input data/LOLv1/Test/input \
  --target data/LOLv1/Test/target \
  --output results/lolv1_test15 \
  --device cuda:0
```

For images without targets, omit `--target`. Enhanced PNGs are written to
`OUTPUT/enhanced`, with `per_image.csv` and `summary.json` beside them.

## Download datasets

```bash
pip install gdown
python scripts/download_datasets.py lolv1 lolv2
python scripts/download_datasets.py dicm mef npe lime
```

See [data/README.md](data/README.md) for the expected directory structure and
source links. Dataset licenses remain with their respective owners.

## Training

```bash
python -m picoret.train \
  --config configs/lolv1.yaml \
  --output runs/picoret_lolv1_seed42 \
  --device cuda:0
```

For LOL-v2 Synthetic, use `configs/lolv2_synthetic.yaml`.

Recommended paper-training recipe:

```text
seed                 42
optimizer            Adam, betas=(0.9, 0.999)
initial learning rate 4e-4
schedule             40K + 60K cosine
training steps       100K
crop / batch         256x256 / 8 on one GPU
augmentation         MixUp(beta=1.2), including identity
gradient clipping    0.01
loss                  MSE + 0.01 VGG19 + 0.2 MS-SSIM
validation           every 5K steps on the fixed internal split
model selection      highest internal-validation PSNR
test protocol        full resolution, no TTA, no tiling, no GT correction
```

Dataset paths, split manifests and run-level controls are in `configs/`; the
optimizer, scheduler and objective are implemented in `picoret/train.py`.
Official test images are never used for training or checkpoint selection.

## Repository layout

```text
Picoret/
├── picoret/          # standalone model, inference, metrics, training
├── configs/          # reproducible training configuration
├── splits/           # fixed internal split; no official test images
├── scripts/          # public dataset downloader
├── weights/          # compact pretrained checkpoints
├── tests/            # architecture smoke tests
└── results/          # optional local evaluation outputs
```

## License

This release is for noncommercial research and education under the included
PolyForm Noncommercial License. The analytical-prior provenance is documented
next to its implementation in `picoret/model.py`.
