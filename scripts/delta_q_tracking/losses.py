from __future__ import annotations

import torch

from utils.loss_utils import ssim


def masked_l1_rgb(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    """Masked RGB L1. Tensors are CxHxW, mask is 1xHxW or HxW in {0,1}."""
    if object_mask.ndim == 2:
        object_mask = object_mask[None]
    mask = object_mask.to(device=pred_rgb.device, dtype=pred_rgb.dtype).clamp(0, 1)
    denom = mask.sum() * pred_rgb.shape[0]
    if float(denom.detach().cpu()) <= 0:
        raise ValueError("Object mask is empty")
    return (torch.abs(pred_rgb - gt_rgb) * mask).sum() / denom


def _mask_like_rgb(pred_rgb: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    if object_mask.ndim == 2:
        object_mask = object_mask[None]
    return object_mask.to(device=pred_rgb.device, dtype=pred_rgb.dtype).clamp(0, 1)


def masked_ssim_rgb(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    """SSIM on masked RGB tensors. Returns a similarity value where higher is better."""
    mask = _mask_like_rgb(pred_rgb, object_mask)
    pred = (pred_rgb * mask).unsqueeze(0)
    gt = (gt_rgb * mask).unsqueeze(0)
    return ssim(pred, gt)


def loss_config(config: dict[str, object]) -> dict[str, object]:
    cfg = config.get("loss", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "use_l1": bool(cfg.get("use_l1", True)),
        "use_ssim": bool(cfg.get("use_ssim", False)),
        "lambda_ssim": float(cfg.get("lambda_ssim", 0.0)),
    }


def masked_rgb_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    object_mask: torch.Tensor,
    config: dict[str, object],
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = loss_config(config)
    use_l1 = bool(cfg["use_l1"])
    use_ssim = bool(cfg["use_ssim"])
    lambda_ssim = float(cfg["lambda_ssim"]) if use_ssim else 0.0
    if not use_l1 and not use_ssim:
        raise ValueError("At least one RGB loss term must be enabled")
    if not use_l1:
        lambda_ssim = 1.0
    if not use_ssim:
        lambda_ssim = 0.0

    l1_value = masked_l1_rgb(pred_rgb, gt_rgb, object_mask) if use_l1 else pred_rgb.sum() * 0.0
    ssim_value = masked_ssim_rgb(pred_rgb, gt_rgb, object_mask) if use_ssim else pred_rgb.new_tensor(1.0)
    loss = (1.0 - lambda_ssim) * l1_value + lambda_ssim * (1.0 - ssim_value)
    return loss, {
        "loss_l1": float(l1_value.detach().cpu()),
        "ssim": float(ssim_value.detach().cpu()),
        "lambda_ssim": lambda_ssim,
        "use_l1": use_l1,
        "use_ssim": use_ssim,
    }
