"""Standalone 817-parameter PicoRet model.

The implementation follows the three stages used in the PicoRet paper:

1. Know: construct four illumination and five chromatic analytical priors.
2. Form: project and spatially gate the priors into two compact feature streams.
3. Correct: condition the streams on image context and predict complementary
   structured and direct residual corrections.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IlluminationPriors(nn.Module):
    """Know stage: construct four parameter-free illumination descriptors."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("w_rec709", torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1))
        # Retained in the state layout for checkpoint compatibility.
        self.register_buffer("w_ycgco", torch.tensor([0.25, 0.5, 0.25]).view(1, 3, 1, 1))

    def forward(self, image: Tensor) -> Tensor:
        red, green, blue = image.split(1, dim=1)
        maximum = image.amax(1, keepdim=True)
        minimum = image.amin(1, keepdim=True)
        return torch.cat(
            [
                (image * self.w_rec709).sum(1, keepdim=True),
                maximum,
                (maximum + minimum) / 2,
                torch.sqrt(red.square() + green.square() + blue.square() + self.eps),
            ],
            dim=1,
        )


class BoundedScalar(nn.Module):
    def __init__(self, value: float = 0.003, lower: float = 1e-6, upper: float = 0.1):
        super().__init__()
        self.lower, self.upper = lower, upper
        position = (value - lower) / (upper - lower)
        self.logit = nn.Parameter(torch.tensor(math.log(position / (1 - position))))

    def forward(self) -> Tensor:
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.logit)


class ColorPriors(nn.Module):
    """Know stage: construct five chromatic descriptors with dark-region stabilization."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("yuv_U", torch.tensor([-0.14713, -0.28886, 0.436]).view(1, 3, 1, 1))
        self.register_buffer("yuv_V", torch.tensor([0.615, -0.51499, -0.10001]).view(1, 3, 1, 1))
        self.register_buffer("ycbcr_Cb", torch.tensor([-0.168736, -0.331264, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("ycbcr_Cr", torch.tensor([0.5, -0.418688, -0.081312]).view(1, 3, 1, 1))
        self.register_buffer("opp_O1", torch.tensor([1.0, -1.0, 0.0]).view(1, 3, 1, 1) / math.sqrt(2))
        self.register_buffer("opp_O2", torch.tensor([1.0, 1.0, -2.0]).view(1, 3, 1, 1) / math.sqrt(6))
        self.rg_threshold = BoundedScalar()
        self.s_threshold = BoundedScalar()

    def forward(self, image: Tensor) -> Tensor:
        work = image.float() if image.dtype in (torch.float16, torch.bfloat16) else image
        red, green, blue = work.split(1, dim=1)
        tau_rg = self.rg_threshold().to(work)
        denominator = (red + green + blue + tau_rg).clamp_min(self.eps)
        maximum, minimum = work.amax(1, keepdim=True), work.amin(1, keepdim=True)
        tau_s = self.s_threshold().to(work)
        return torch.cat(
            [
                (work * self.ycbcr_Cb).sum(1, keepdim=True),
                (work * self.ycbcr_Cr).sum(1, keepdim=True),
                (red + tau_rg / 3) / denominator,
                (green + tau_rg / 3) / denominator,
                (maximum - minimum) / (maximum + tau_s).clamp_min(self.eps),
            ],
            dim=1,
        )


class LowRankDepthwise(nn.Module):
    """Form stage: rank-2 separable spatial guidance with a 7x7 receptive field."""

    def __init__(self, channels: int, rank: int = 2, kernel: int = 7):
        super().__init__()
        self.channels, self.rank = channels, rank
        self.vertical = nn.Conv2d(
            channels, channels * rank, (kernel, 1), padding=(kernel // 2, 0), groups=channels, bias=False
        )
        self.horizontal = nn.Conv2d(
            channels * rank,
            channels * rank,
            (1, kernel),
            padding=(0, kernel // 2),
            groups=channels * rank,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.horizontal(self.vertical(inputs))
        batch, _, height, width = values.shape
        values = values.reshape(batch, self.channels, self.rank, height, width).sum(2)
        return values + self.bias[None, :, None, None]


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.act(self.bn(self.pw(self.dw(inputs))))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, 1)
        self.fc2 = nn.Linear(1, channels)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, _, _ = inputs.shape
        values = self.pool(inputs).reshape(batch, channels)
        values = torch.tanh(self.fc2(F.relu(self.fc1(values))))
        return inputs * values.reshape(batch, channels, 1, 1)


class FeatureRefinement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.se_attn = SqueezeExcitation(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.depthwise_conv(inputs) * self.se_attn(inputs) + inputs


class CoarseIlluminationContext(nn.Module):
    """Correct stage: predict a low-frequency illumination gain field at 1/8 scale."""

    def __init__(self, channels: int = 4, hidden: int = 4, factor: int = 8):
        super().__init__()
        self.factor = factor
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, priors: Tensor) -> Tensor:
        height, width = priors.shape[-2:]
        pooled = F.adaptive_avg_pool2d(
            priors, (max(1, height // self.factor), max(1, width // self.factor))
        )
        native = torch.tanh(self.net(pooled))
        return F.interpolate(native, size=(height, width), mode="bilinear", align_corners=False)


class PicoRet(nn.Module):
    """PicoRet: an 817-parameter Know--Form--Correct low-light enhancer."""

    def __init__(self) -> None:
        super().__init__()
        channels = 5

        # Stage 1 -- Know: analytical illumination and chromatic knowledge.
        # The nine-prior construction is adapted from Multinex-Nano
        # (https://github.com/albrateanu/multinex). PicoRet stabilizes the
        # dark-region chromatic ratios; the subsequent Form and Correct stages
        # are PicoRet-specific designs.
        # These modules convert RGB into four illumination priors and five
        # color priors; only the two bounded dark-region stabilizers are learned.
        self.illum_extractor = IlluminationPriors()
        self.chroma_extractor = ColorPriors()

        # Stage 2 -- Form (PGFF): compact prior projection and rank-2 spatial
        # guidance form two branch-specific five-channel representations.
        self.illum_stem = nn.Conv2d(4, channels, 1)
        self.chroma_stem = nn.Conv2d(5, channels, 1)
        self.illum_att = LowRankDepthwise(4)
        self.chroma_att = LowRankDepthwise(5)
        self.illum_att_proj = nn.Conv2d(4, channels, 1)
        self.chroma_att_proj = nn.Conv2d(5, channels, 1)
        self.illum_mid_seq = nn.Sequential(
            nn.Sequential(DepthwiseSeparableConv(channels), FeatureRefinement(channels))
        )
        self.chroma_mid_seq = nn.Sequential(
            nn.Sequential(DepthwiseSeparableConv(channels), FeatureRefinement(channels))
        )

        # Stage 3 -- Correct (SAC): coarse illumination context and one-way
        # illumination-to-color conditioning precede two complementary heads.
        # The structured head predicts Delta_L * Delta_C, while the direct head
        # predicts an unconstrained RGB residual Delta_D.
        self.head_luma = nn.Conv2d(channels, 1, 1)
        self.head_chroma = nn.Conv2d(channels, 3, 1)
        self.cross_proj = nn.Conv2d(channels, channels, 1)
        self.direct_head = nn.Conv2d(2 * channels, 3, 1)
        self.coarse_context = CoarseIlluminationContext()

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("PicoRet expects Bx3xHxW RGB tensors")

        # Know: P_L in R^(4xHxW), P_C in R^(5xHxW).
        illumination = self.illum_extractor(image)
        color = self.chroma_extractor(image)

        # Form: pointwise projection followed by content-dependent spatial
        # gating. Both branches retain the full input resolution.
        features_l = self.illum_stem(illumination)
        features_c = self.chroma_stem(color)
        features_l = features_l * torch.sigmoid(self.illum_att_proj(self.illum_att(illumination)))
        features_c = features_c * torch.sigmoid(self.chroma_att_proj(self.chroma_att(color)))

        # Correct: illumination first conditions color and supplies a coarse
        # gain field; each branch is then independently refined.
        features_c = features_c + self.cross_proj(features_l)
        features_l = features_l * (1 + self.coarse_context(illumination))
        features_l = self.illum_mid_seq(features_l)
        features_c = self.chroma_mid_seq(features_c)

        # I_hat = I + Delta_L * Delta_C + Delta_D.
        product = self.head_chroma(features_c) * self.head_luma(features_l)
        direct = self.direct_head(torch.cat((features_l, features_c), dim=1))
        return image + product + direct

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> PicoRet:
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("params_ema", payload.get("params", payload))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = PicoRet()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
