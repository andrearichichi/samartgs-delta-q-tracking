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
    load_enriched_ply_metadata,
    load_gaussian_model,
    load_mask_frame,
    load_rgb_frame,
    load_simple_yaml,
    save_json,
    save_overlays,
    support_metrics,
    tensor_to_pil_rgb,
    warn_debug_image_shift,
)
from scripts.delta_q_tracking.losses import maybe_shift_render


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual +/- delta_q render check.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--target-frame", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--rotation-mode", choices=["config", "none", "rigid"], default="config")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    rotation_mode = str(cfg.get("rotation_mode", "none")) if args.rotation_mode == "config" else args.rotation_mode
    if rotation_mode not in {"none", "rigid"}:
        raise ValueError(f"Unsupported rotation_mode={rotation_mode}")
    debug_shift = warn_debug_image_shift(cfg)
    source = ensure_colmap_source(cfg)
    out_dir = ensure_output_dir(Path(cfg["output_root"]) / "01_manual_deltaq" / f"cam_{args.cam:03d}" / f"rotation_{rotation_mode}")
    model_path = Path(cfg["model_path"]) / "point_cloud" / f"iteration_{int(cfg['iteration'])}" / "point_cloud.ply"

    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]))
    meta = load_enriched_ply_metadata(model_path)
    moving_mask = meta["part_ids"] == int(cfg["moving_part_id"])
    camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.target_frame)
    rgb = load_rgb_frame(cfg["rgb_root"], args.cam, args.target_frame)
    mask = load_mask_frame(cfg["mask_root"], args.cam, args.target_frame)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    summary = {}

    for label, delta in [("zero", 0.0), ("plus", args.delta), ("minus", -args.delta)]:
        dq = torch.tensor(delta, dtype=gaussians.get_xyz.dtype, device="cuda")
        xyz = deform_xyz_relative_continuous(gaussians.get_xyz, moving_mask, meta["joint_origin"], meta["joint_axis"], dq)
        rotation = None
        if rotation_mode == "rigid":
            rotation = deform_rotation_relative_continuous(gaussians.get_rotation, moving_mask, meta["joint_axis"], dq)
        rendered = render(camera, DeformedGaussian(gaussians, xyz, rotation), default_pipeline(), bg)["render"]
        shifted_rendered = maybe_shift_render(rendered, debug_shift)
        raw_img = tensor_to_pil_rgb(rendered)
        shifted_img = tensor_to_pil_rgb(shifted_rendered)
        prefix = f"cam_{args.cam:03d}_frame_{args.target_frame:06d}_{label}"
        save_overlays(raw_img, rgb, mask, out_dir, f"{prefix}_raw")
        save_overlays(shifted_img, rgb, mask, out_dir, f"{prefix}_shifted")
        summary[label] = {
            "delta_q": delta,
            "raw_metrics": support_metrics(raw_img, mask),
            "shifted_metrics": support_metrics(shifted_img, mask),
        }
    summary["debug_image_shift"] = debug_shift
    summary["rotation_mode"] = rotation_mode
    save_json(summary, out_dir / "manual_deltaq_metrics.json")
    print(f"saved_dir={out_dir}")
    print(f"rotation_mode={rotation_mode}")
    for label in ("zero", "plus", "minus"):
        payload = summary[label]
        raw_shift = payload["raw_metrics"]["centroid_shift_render_minus_mask_px"]
        shifted_shift = payload["shifted_metrics"]["centroid_shift_render_minus_mask_px"]
        print(
            f"{label}: delta_q={payload['delta_q']:+.5f} "
            f"raw_iou={payload['raw_metrics']['support_mask_iou']:.6f} "
            f"raw_shift=({raw_shift[0]:+.2f},{raw_shift[1]:+.2f}) "
            f"shifted_iou={payload['shifted_metrics']['support_mask_iou']:.6f} "
            f"shifted_shift=({shifted_shift[0]:+.2f},{shifted_shift[1]:+.2f})"
        )


if __name__ == "__main__":
    main()
