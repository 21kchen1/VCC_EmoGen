"""
Feature-engineering regularizers for EmoGen.

This module adds differentiable, handcrafted feature losses that can be
back-propagated through the frozen Stable Diffusion U-Net/text encoder into
the trainable EmoGen mapper.

Included feature engineering methods:
1) RGB soft color histogram + color moments
2) HOG-like Sobel gradient orientation histogram
3) Low-frequency 2D DCT coefficients
4) PCA-whitened CLIP attribute loss helper
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from [-1, 1] or [0, 1] to [0, 1]."""
    if x.min().detach() < -0.05:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: Bx3xHxW in [0,1], returns Bx1xHxW."""
    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create orthonormal DCT-II transform matrix of size n x n."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    mat = torch.cos(math.pi / n * (i + 0.5) * k)
    mat[0, :] *= math.sqrt(1.0 / n)
    mat[1:, :] *= math.sqrt(2.0 / n)
    return mat


class FeatureEngineeringLoss(nn.Module):
    """
    Differentiable feature-engineering losses.

    pred_img and target_img should be Bx3xHxW images in [-1,1] or [0,1].
    The returned total loss is already weighted by color_weight, hog_weight,
    and dct_weight.
    """

    def __init__(
        self,
        color_weight: float = 0.03,
        hog_weight: float = 0.02,
        dct_weight: float = 0.01,
        hist_bins: int = 16,
        hog_bins: int = 9,
        feature_size: int = 128,
        dct_size: int = 64,
        dct_keep: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.color_weight = float(color_weight)
        self.hog_weight = float(hog_weight)
        self.dct_weight = float(dct_weight)
        self.hist_bins = int(hist_bins)
        self.hog_bins = int(hog_bins)
        self.feature_size = int(feature_size)
        self.dct_size = int(dct_size)
        self.dct_keep = int(dct_keep)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def color_histogram(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_01(x.float())
        x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode="bilinear", align_corners=False)
        b, c, h, w = x.shape
        values = x.flatten(2)  # B,C,N

        centers = torch.linspace(0.0, 1.0, self.hist_bins, device=x.device, dtype=x.dtype)
        width = 1.0 / max(self.hist_bins - 1, 1)
        # Soft triangular bins: B,C,N,K
        weights = torch.relu(1.0 - torch.abs(values.unsqueeze(-1) - centers) / (width + self.eps))
        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=-1, keepdim=True) + self.eps)
        return hist.flatten(1)  # B, C*K

    def color_moments(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_01(x.float())
        x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode="bilinear", align_corners=False)
        flat = x.flatten(2)
        mean = flat.mean(dim=-1)
        std = flat.std(dim=-1).clamp_min(self.eps)
        skew = (((flat - mean.unsqueeze(-1)) / std.unsqueeze(-1)) ** 3).mean(dim=-1)
        return torch.cat([mean, std, skew], dim=-1)

    def hog_histogram(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_01(x.float())
        x = F.interpolate(x, size=(self.feature_size, self.feature_size), mode="bilinear", align_corners=False)
        gray = _rgb_to_gray(x)
        gx = F.conv2d(gray, self.sobel_x.float(), padding=1)
        gy = F.conv2d(gray, self.sobel_y.float(), padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        # unsigned orientation in [0, pi)
        ori = torch.atan2(gy, gx).remainder(math.pi)

        centers = torch.linspace(0.0, math.pi, self.hog_bins + 1, device=x.device, dtype=x.dtype)[:-1]
        bin_width = math.pi / self.hog_bins
        # Circular distance to orientation centers
        diff = torch.abs(ori.unsqueeze(-1) - centers)
        diff = torch.minimum(diff, math.pi - diff)
        weights = torch.relu(1.0 - diff / (bin_width + self.eps))
        hist = (weights * mag.unsqueeze(-1)).flatten(2).sum(dim=2)  # B,1,K
        hist = hist / (hist.sum(dim=-1, keepdim=True) + self.eps)
        return hist.flatten(1)

    def dct_lowfreq(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_01(x.float())
        x = F.interpolate(x, size=(self.dct_size, self.dct_size), mode="bilinear", align_corners=False)
        gray = _rgb_to_gray(x).squeeze(1)  # B,H,W
        n = self.dct_size
        dct = _dct_matrix(n, x.device, torch.float32)
        coeff = torch.matmul(torch.matmul(dct, gray.float()), dct.t())
        coeff = coeff[:, : self.dct_keep, : self.dct_keep]
        # Drop the DC term from the loss scale? Keep it because color/global brightness matters for FID.
        coeff = coeff.flatten(1)
        return F.normalize(coeff, dim=-1, eps=self.eps)

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred = _to_01(pred_img)
        target = _to_01(target_img)

        pred_hist = self.color_histogram(pred)
        target_hist = self.color_histogram(target)
        pred_mom = self.color_moments(pred)
        target_mom = self.color_moments(target)
        loss_color = F.l1_loss(pred_hist, target_hist) + 0.5 * F.l1_loss(pred_mom, target_mom)

        loss_hog = F.l1_loss(self.hog_histogram(pred), self.hog_histogram(target))
        loss_dct = F.smooth_l1_loss(self.dct_lowfreq(pred), self.dct_lowfreq(target))

        total = (
            self.color_weight * loss_color
            + self.hog_weight * loss_hog
            + self.dct_weight * loss_dct
        )
        logs = {
            "loss_fe": total.detach(),
            "loss_color": loss_color.detach(),
            "loss_hog": loss_hog.detach(),
            "loss_dct": loss_dct.detach(),
        }
        return total, logs


class PCAProjector(nn.Module):
    """PCA projection for whitening/noise reduction of CLIP attribute embeddings."""

    def __init__(self, n_components: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.n_components = int(n_components)
        self.eps = float(eps)
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))
        self.register_buffer("scale", torch.empty(0))

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> "PCAProjector":
        x = x.float()
        n_components = min(self.n_components, x.shape[0] - 1, x.shape[1])
        mean = x.mean(dim=0, keepdim=True)
        xc = x - mean
        # q slightly larger than target dimension improves stability.
        q = min(n_components + 16, min(xc.shape))
        _, s, v = torch.pca_lowrank(xc, q=q, center=False)
        comps = v[:, :n_components]
        scale = s[:n_components].clamp_min(self.eps)
        self.mean = mean
        self.components = comps
        self.scale = scale
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.components.numel() == 0:
            raise RuntimeError("PCAProjector must be fitted before transform().")
        y = (x.float() - self.mean.to(x.device)) @ self.components.to(x.device)
        y = y / self.scale.to(x.device)
        return F.normalize(y, dim=-1)


def pca_attribute_ce(
    project_semantic: torch.Tensor,
    total_attr_embed: torch.Tensor,
    index_attr: torch.Tensor,
    projector: PCAProjector,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Cross-entropy over cosine similarities in PCA-whitened CLIP attribute space."""
    attr_z = projector.transform(total_attr_embed)          # K,Dp
    sem_z = projector.transform(project_semantic)           # B,Dp
    logits = torch.matmul(sem_z, attr_z.t()) * temperature  # B,K
    return F.cross_entropy(logits, index_attr.view(-1).long())


def predict_x0_from_noise(
    noise_scheduler,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    model_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Recover predicted clean latent x_0 from U-Net prediction.

    Stable Diffusion v1.x uses epsilon prediction. v_prediction is included for
    compatibility with newer schedulers.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha = alpha_t.sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()

    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        pred_x0 = (noisy_latents - sqrt_one_minus_alpha * model_pred) / sqrt_alpha.clamp_min(1e-6)
    elif pred_type == "v_prediction":
        pred_x0 = sqrt_alpha * noisy_latents - sqrt_one_minus_alpha * model_pred
    else:
        raise ValueError(f"Unsupported prediction_type: {pred_type}")

    return pred_x0
