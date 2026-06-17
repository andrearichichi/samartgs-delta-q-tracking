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
    resolve_gaussian_ply,
    save_csv,
    save_json,
    save_overlays,
    support_metrics,
    tensor_to_pil_rgb,
)
from scripts.delta_q_tracking.losses import masked_rgb_loss


def candidate_values(q_min: float, q_max: float, num_samples: int) -> list[float]:
    if num_samples < 2:
        values = [0.0]
    else:
        values = torch.linspace(q_min, q_max, num_samples).tolist()
    values.append(0.0)
    return sorted({round(float(v), 12) for v in values})


def plot_curves(rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    q = [float(row["q_candidate"]) for row in rows]

    def finish(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    for key, ylabel, title, name in [
        ("total_loss", "total loss", "Initial q candidate vs total loss", "q_vs_total_loss.png"),
        ("support_iou", "support IoU", "Initial q candidate vs support IoU", "q_vs_support_iou.png"),
        ("rgb_loss", "RGB/L1 loss", "Initial q candidate vs RGB loss", "q_vs_rgb_loss.png"),
    ]:
        plt.figure(figsize=(7.5, 4.0))
        plt.plot(q, [float(row[key]) for row in rows], linewidth=2)
        plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("q candidate")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        finish(name)

    depth_values = [row.get("depth_loss_if_available", "") for row in rows]
    if any(value not in {"", None} for value in depth_values):
        plt.figure(figsize=(7.5, 4.0))
        plt.plot(q, [float(row["depth_loss_if_available"]) for row in rows], linewidth=2)
        plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("q candidate")
        plt.ylabel("depth loss")
        plt.title("Initial q candidate vs depth loss")
        plt.grid(True, alpha=0.3)
        finish("q_vs_depth_loss.png")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained 1D calibration of initial q_ref(0).")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--q-min", type=float, default=-0.05)
    parser.add_argument("--q-max", type=float, default=0.05)
    parser.add_argument("--num-samples", type=int, default=101)
    parser.add_argument("--output-subdir", default="initial_q_calibration")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    rotation_mode = str(cfg.get("rotation_mode", "rigid"))
    if rotation_mode not in {"none", "rigid"}:
        raise ValueError(f"Unsupported rotation_mode={rotation_mode}")
    source = ensure_colmap_source(cfg)
    out_dir = ensure_output_dir(Path(cfg["output_root"]) / args.output_subdir / f"cam_{args.cam:03d}" / f"frame_{args.frame:06d}")
    gaussian_source = str(cfg.get("gaussian_source", "point_cloud"))
    model_ply = resolve_gaussian_ply(cfg["model_path"], int(cfg["iteration"]), gaussian_source)

    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]), gaussian_source=gaussian_source)
    meta = load_enriched_ply_metadata(model_ply)
    moving_mask = meta["part_ids"] == int(cfg["moving_part_id"])
    base_xyz = gaussians.get_xyz.detach()
    base_rotation = gaussians.get_rotation
    camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.frame)
    target_rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, args.frame)
    target_mask_pil = load_mask_frame(cfg["mask_root"], args.cam, args.frame)
    target_rgb = image_to_cuda_rgb(target_rgb_pil)
    target_mask = mask_to_cuda(target_mask_pil)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    rows: list[dict[str, object]] = []
    rendered_by_q: dict[float, object] = {}
    for q_value in candidate_values(args.q_min, args.q_max, args.num_samples):
        q_tensor = torch.tensor(q_value, dtype=base_xyz.dtype, device=base_xyz.device)
        with torch.no_grad():
            xyz = deform_xyz_relative_continuous(base_xyz, moving_mask, meta["joint_origin"], meta["joint_axis"], q_tensor)
            rotation = None
            if rotation_mode == "rigid":
                rotation = deform_rotation_relative_continuous(base_rotation, moving_mask, meta["joint_axis"], q_tensor)
            rendered = render(camera, DeformedGaussian(gaussians, xyz, rotation), default_pipeline(), bg)["render"]
            loss, parts = masked_rgb_loss(rendered, target_rgb, target_mask, cfg)
        image = tensor_to_pil_rgb(rendered)
        metrics = support_metrics(image, target_mask_pil)
        rendered_by_q[q_value] = image
        rows.append(
            {
                "q_candidate": q_value,
                "rgb_loss": parts["loss_l1"],
                "ssim_loss": 1.0 - parts["ssim"],
                "total_loss": float(loss.detach().cpu()),
                "support_iou": metrics["support_mask_iou"],
                "moving_iou_if_available": "",
                "depth_loss_if_available": "",
            }
        )

    best_loss_row = min(rows, key=lambda row: float(row["total_loss"]))
    best_iou_row = max(rows, key=lambda row: float(row["support_iou"]))
    zero_row = min(rows, key=lambda row: abs(float(row["q_candidate"])))
    save_csv(rows, out_dir / "initial_q_scan.csv")
    plot_paths = plot_curves(rows, out_dir)

    overlay_specs = [
        ("q_zero", float(zero_row["q_candidate"])),
        ("best_total_loss", float(best_loss_row["q_candidate"])),
    ]
    if float(best_iou_row["q_candidate"]) != float(best_loss_row["q_candidate"]):
        overlay_specs.append(("best_support_iou", float(best_iou_row["q_candidate"])))
    for label, q_value in overlay_specs:
        save_overlays(rendered_by_q[q_value], target_rgb_pil, target_mask_pil, out_dir, f"{label}_q_{q_value:+.6f}")

    total_improvement = float(zero_row["total_loss"]) - float(best_loss_row["total_loss"])
    iou_improvement = float(best_iou_row["support_iou"]) - float(zero_row["support_iou"])
    summary = {
        "camera": f"cam_{args.cam:03d}",
        "frame": args.frame,
        "q_min": args.q_min,
        "q_max": args.q_max,
        "num_candidates": len(rows),
        "gaussian_source": gaussian_source,
        "gaussian_ply": str(model_ply),
        "rotation_mode": rotation_mode,
        "best_q_by_total_loss": best_loss_row,
        "best_q_by_support_iou": best_iou_row,
        "q_zero": zero_row,
        "total_loss_improvement_vs_q0": total_improvement,
        "support_iou_improvement_vs_q0": iou_improvement,
        "q0_is_best_total_loss": float(best_loss_row["q_candidate"]) == float(zero_row["q_candidate"]),
        "q0_is_best_support_iou": float(best_iou_row["q_candidate"]) == float(zero_row["q_candidate"]),
        "plots": [str(path) for path in plot_paths],
    }
    save_json(summary, out_dir / "initial_q_calibration_summary.json")

    print(f"output_dir={out_dir}")
    print(f"best_q_by_total_loss={float(best_loss_row['q_candidate']):+.8f}")
    print(f"best_total_loss={float(best_loss_row['total_loss']):.8f}")
    print(f"best_q_by_support_iou={float(best_iou_row['q_candidate']):+.8f}")
    print(f"best_support_iou={float(best_iou_row['support_iou']):.8f}")
    print(f"q0_total_loss={float(zero_row['total_loss']):.8f}")
    print(f"q0_support_iou={float(zero_row['support_iou']):.8f}")
    print(f"total_loss_improvement_vs_q0={total_improvement:.8f}")
    print(f"support_iou_improvement_vs_q0={iou_improvement:.8f}")
    print(f"q0_is_best_total_loss={summary['q0_is_best_total_loss']}")
    print(f"q0_is_best_support_iou={summary['q0_is_best_support_iou']}")


if __name__ == "__main__":
    main()
