from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render

from scripts.delta_q_tracking.articulation import deform_rotation_relative_continuous, deform_xyz_relative_continuous
from scripts.delta_q_tracking.deformed_gaussian import DeformedGaussian
from scripts.delta_q_tracking.io_utils import (
    build_colmap_camera,
    default_pipeline,
    ensure_colmap_source,
    ensure_output_dir,
    image_to_cuda_rgb,
    load_enriched_ply_metadata,
    load_gaussian_model,
    load_mask_frame,
    load_rgb_frame,
    load_simple_yaml,
    mask_to_cuda,
    save_json,
    save_overlays,
    tensor_to_pil_rgb,
    warn_debug_image_shift,
)
from scripts.delta_q_tracking.losses import loss_config, masked_rgb_loss, maybe_shift_render


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one frame-to-frame delta_q scalar.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--source-frame", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    rotation_mode = str(cfg.get("rotation_mode", "none"))
    if rotation_mode not in {"none", "rigid"}:
        raise ValueError(f"Unsupported rotation_mode={rotation_mode}")
    debug_shift = warn_debug_image_shift(cfg)
    source = ensure_colmap_source(cfg)
    out_dir = ensure_output_dir(Path(cfg["output_root"]) / "02_fit_single_step" / f"cam_{args.cam:03d}")
    model_ply = Path(cfg["model_path"]) / "point_cloud" / f"iteration_{int(cfg['iteration'])}" / "point_cloud.ply"

    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]))
    meta = load_enriched_ply_metadata(model_ply)
    moving_mask = meta["part_ids"] == int(cfg["moving_part_id"])
    camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.target_frame)
    target_rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, args.target_frame)
    target_mask_pil = load_mask_frame(cfg["mask_root"], args.cam, args.target_frame)
    target_rgb = image_to_cuda_rgb(target_rgb_pil)
    target_mask = mask_to_cuda(target_mask_pil)

    delta_q = torch.nn.Parameter(torch.zeros((), dtype=gaussians.get_xyz.dtype, device="cuda"))
    optimizer = torch.optim.Adam([delta_q], lr=float(cfg["lr"]))
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    loss_curve = []
    l1_curve = []
    ssim_curve = []
    delta_curve = []
    grad_curve = []
    final_raw_render = None
    final_shifted_render = None

    for _ in range(int(cfg["num_iters"])):
        optimizer.zero_grad(set_to_none=True)
        xyz = deform_xyz_relative_continuous(gaussians.get_xyz, moving_mask, meta["joint_origin"], meta["joint_axis"], delta_q)
        rotation = None
        if rotation_mode == "rigid":
            rotation = deform_rotation_relative_continuous(gaussians.get_rotation, moving_mask, meta["joint_axis"], delta_q)
        rendered = render(camera, DeformedGaussian(gaussians, xyz, rotation), default_pipeline(), bg)["render"]
        shifted_rendered = maybe_shift_render(rendered, debug_shift)
        loss, loss_parts = masked_rgb_loss(shifted_rendered, target_rgb, target_mask, cfg)
        loss.backward()
        if delta_q.grad is None:
            raise RuntimeError("delta_q.grad is None; deformation is not connected to the loss")
        grad = float(delta_q.grad.detach().cpu())
        if not torch.isfinite(delta_q.grad):
            raise RuntimeError("delta_q.grad is not finite")
        optimizer.step()
        loss_curve.append(float(loss.detach().cpu()))
        l1_curve.append(loss_parts["loss_l1"])
        ssim_curve.append(loss_parts["ssim"])
        delta_curve.append(float(delta_q.detach().cpu()))
        grad_curve.append(grad)
        final_raw_render = rendered.detach()
        final_shifted_render = shifted_rendered.detach()

    if final_raw_render is None or final_shifted_render is None:
        raise RuntimeError("No optimization iterations were run")
    final_raw_img = tensor_to_pil_rgb(final_raw_render)
    final_shifted_img = tensor_to_pil_rgb(final_shifted_render)
    prefix = f"cam_{args.cam:03d}_frame_{args.target_frame:06d}_fit"
    save_overlays(final_raw_img, target_rgb_pil, target_mask_pil, out_dir, f"{prefix}_raw")
    save_overlays(final_shifted_img, target_rgb_pil, target_mask_pil, out_dir, f"{prefix}_shifted")
    grad_nonzero = any(abs(g) > 1e-10 for g in grad_curve)
    if not grad_nonzero:
        raise RuntimeError("delta_q.grad stayed zero for all iterations")
    result = {
        "warning": "Current USB static alignment gate fails and debug image-space shift may be enabled; this delta_q is for plumbing/debug only.",
        "debug_image_shift": debug_shift,
        "rotation_mode": rotation_mode,
        "camera": f"cam_{args.cam:03d}",
        "source_frame": args.source_frame,
        "target_frame": args.target_frame,
        "delta_q": float(delta_q.detach().cpu()),
        "q_ref_t1": None,
        "q_ref_t1_note": "No frame-level ground-truth q source is configured; this script estimates delta_q only.",
        "num_iters": int(cfg["num_iters"]),
        "lr": float(cfg["lr"]),
        "loss_config": loss_config(cfg),
        "loss_curve": loss_curve,
        "l1_curve": l1_curve,
        "ssim_curve": ssim_curve,
        "delta_curve": delta_curve,
        "grad_curve": grad_curve,
        "grad_nonzero": grad_nonzero,
    }
    save_json(result, out_dir / "fit_result.json")
    print(f"delta_q={result['delta_q']:+.8f}")
    print(f"final_loss={loss_curve[-1]:.8f}")
    print(f"grad_nonzero={result['grad_nonzero']}")
    print(f"saved_dir={out_dir}")


if __name__ == "__main__":
    main()
