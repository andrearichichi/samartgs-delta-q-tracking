from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render

from scripts.delta_q_tracking.io_utils import (
    build_colmap_camera,
    default_pipeline,
    ensure_colmap_source,
    ensure_output_dir,
    load_gaussian_model,
    load_mask_frame,
    load_rgb_frame,
    load_simple_yaml,
    save_csv,
    save_json,
    save_overlays,
    support_metrics,
    tensor_to_pil_rgb,
    warn_debug_image_shift,
)
from scripts.delta_q_tracking.losses import maybe_shift_render


def main() -> None:
    parser = argparse.ArgumentParser(description="Static delta-q alignment gate at delta_q=0.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    debug_shift = warn_debug_image_shift(cfg)
    source = ensure_colmap_source(cfg)
    out_dir = ensure_output_dir(Path(cfg["output_root"]) / "00_static_alignment" / f"cam_{args.cam:03d}")

    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]))
    camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.frame)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        rendered = render(camera, gaussians, default_pipeline(), bg)["render"]
        shifted_rendered = maybe_shift_render(rendered, debug_shift)
    raw_img = tensor_to_pil_rgb(rendered)
    shifted_img = tensor_to_pil_rgb(shifted_rendered)
    rgb = load_rgb_frame(cfg["rgb_root"], args.cam, args.frame)
    mask = load_mask_frame(cfg["mask_root"], args.cam, args.frame)
    raw_metrics = support_metrics(raw_img, mask)
    shifted_metrics = support_metrics(shifted_img, mask)

    save_overlays(raw_img, rgb, mask, out_dir, f"cam_{args.cam:03d}_frame_{args.frame:06d}_raw")
    save_overlays(shifted_img, rgb, mask, out_dir, f"cam_{args.cam:03d}_frame_{args.frame:06d}_shifted")
    save_json({"raw": raw_metrics, "shifted": shifted_metrics, "debug_image_shift": debug_shift}, out_dir / "metrics.json")
    rows = []
    for version, metrics in [("raw", raw_metrics), ("shifted", shifted_metrics)]:
        rows.append(
            {
                "version": version,
                "camera": f"cam_{args.cam:03d}",
                "frame": args.frame,
                "support_mask_iou": metrics["support_mask_iou"],
                "centroid_dx": metrics["centroid_shift_render_minus_mask_px"][0],
                "centroid_dy": metrics["centroid_shift_render_minus_mask_px"][1],
                "bbox_tl_dx": metrics["bbox_top_left_shift_render_minus_mask_px"][0],
                "bbox_tl_dy": metrics["bbox_top_left_shift_render_minus_mask_px"][1],
                "bbox_br_dx": metrics["bbox_bottom_right_shift_render_minus_mask_px"][0],
                "bbox_br_dy": metrics["bbox_bottom_right_shift_render_minus_mask_px"][1],
            }
        )
    save_csv(rows, out_dir / "metrics.csv")

    metrics = shifted_metrics if debug_shift["enabled"] else raw_metrics
    shift = metrics["centroid_shift_render_minus_mask_px"]
    shift_norm = (shift[0] ** 2 + shift[1] ** 2) ** 0.5
    iou_ok = metrics["support_mask_iou"] >= float(cfg["alignment_iou_threshold"])
    shift_ok = shift_norm <= float(cfg["alignment_max_centroid_shift_px"])
    status = "PASS" if iou_ok and shift_ok else "FAIL"
    raw_shift = raw_metrics["centroid_shift_render_minus_mask_px"]
    shifted_shift = shifted_metrics["centroid_shift_render_minus_mask_px"]
    print(f"STATIC_ALIGNMENT_{status}")
    print(f"camera=cam_{args.cam:03d} frame={args.frame:06d}")
    print(f"raw_iou={raw_metrics['support_mask_iou']:.6f}")
    print(f"raw_centroid_shift=({raw_shift[0]:+.3f},{raw_shift[1]:+.3f})")
    print(f"shifted_iou={shifted_metrics['support_mask_iou']:.6f}")
    print(f"shifted_centroid_shift=({shifted_shift[0]:+.3f},{shifted_shift[1]:+.3f}) norm={shift_norm:.3f}px")
    print(f"output_dir={out_dir}")
    if status != "PASS":
        print("WARNING: static alignment gate failed; delta_q estimates should not be trusted yet.")


if __name__ == "__main__":
    main()
