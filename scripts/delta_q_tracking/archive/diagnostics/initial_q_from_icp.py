from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)

from scripts.delta_q_tracking.articulation import (
    deform_rotation_relative_continuous,
    deform_xyz_relative_continuous,
    rodrigues,
)
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
    resolve_path,
    save_csv,
    save_json,
    save_overlays,
    support_metrics,
    tensor_to_pil_rgb,
)
from scripts.delta_q_tracking.losses import masked_rgb_loss


class OpacityMaskedGaussian(DeformedGaussian):
    """Renderer wrapper that keeps only selected Gaussians visible."""

    def __init__(self, base, deformed_xyz, deformed_rotation, visible_mask):
        super().__init__(base, deformed_xyz, deformed_rotation)
        self.visible_mask = visible_mask.reshape(-1, 1).to(device=base.get_opacity.device, dtype=torch.bool)

    @property
    def get_opacity(self):
        zeros = torch.zeros_like(self.base.get_opacity)
        return torch.where(self.visible_mask, self.base.get_opacity, zeros)


def read_colmap(source: Path):
    sparse = source / "sparse" / "0"
    try:
        extr = read_extrinsics_binary(str(sparse / "images.bin"))
        intr = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        extr = read_extrinsics_text(str(sparse / "images.txt"))
        intr = read_intrinsics_text(str(sparse / "cameras.txt"))
    return extr, intr


def colmap_frame_camera(source: Path, cam_idx: int):
    extrinsics, intrinsics = read_colmap(source)
    wanted = f"cam_{cam_idx:03d}/frame_000000.png"
    extr = next(e for e in extrinsics.values() if e.name == wanted)
    intr = intrinsics[extr.camera_id]
    if intr.model == "PINHOLE":
        fx, fy, cx, cy = map(float, intr.params[:4])
    elif intr.model == "SIMPLE_PINHOLE":
        fx = fy = float(intr.params[0])
        cx, cy = map(float, intr.params[1:3])
    else:
        raise ValueError(f"Unsupported COLMAP camera model {intr.model}")
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "Rcw": qvec2rotmat(extr.qvec).astype(np.float64),
        "tcw": np.asarray(extr.tvec, dtype=np.float64),
        "image_name": extr.name,
        "camera_model": intr.model,
    }


def project_world(points_world: np.ndarray, cam: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cam_pts = points_world @ cam["Rcw"].T + cam["tcw"].reshape(1, 3)
    z = cam_pts[:, 2]
    u = cam["fx"] * cam_pts[:, 0] / z + cam["cx"]
    v = cam["fy"] * cam_pts[:, 1] / z + cam["cy"]
    return np.stack([u, v], axis=1), z


def unproject_depth_to_world(depth: np.ndarray, mask: np.ndarray, cam: dict[str, Any]) -> np.ndarray:
    ys, xs = np.where(mask & np.isfinite(depth) & (depth > 0))
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cam["cx"]) * z / cam["fx"]
    y = (ys.astype(np.float64) - cam["cy"]) * z / cam["fy"]
    cam_pts = np.stack([x, y, z], axis=1)
    world = (cam_pts - cam["tcw"].reshape(1, 3)) @ cam["Rcw"]
    return world


def draw_projected_mask(
    points_world: np.ndarray,
    cam: dict[str, Any],
    radius: int,
    full_mask: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    uv, z = project_world(points_world, cam)
    valid = (
        np.isfinite(z)
        & (z > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < cam["width"])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < cam["height"])
    )
    image = Image.new("L", (cam["width"], cam["height"]), 0)
    draw = ImageDraw.Draw(image)
    for u, v in uv[valid]:
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=255)
    projected = np.asarray(image, dtype=np.uint8) > 0
    target = projected & full_mask & np.isfinite(depth) & (depth > 0)
    return target, {
        "projected_points_total": int(len(points_world)),
        "projected_points_in_image": int(valid.sum()),
        "projected_mask_pixels": int(projected.sum()),
        "target_moving_mask_pixels": int(target.sum()),
        "mask_radius_px": int(radius),
    }


def deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    h = src_centered.T @ tgt_centered
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tgt_centroid - r @ src_centroid
    return r, t


def run_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int,
    trim_percentile: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    current = source.copy()
    r_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    history: list[dict[str, float]] = []
    prev_rmse: float | None = None
    for iteration in range(max_iters):
        distances, indices = tree.query(current, k=1, workers=-1)
        cutoff = np.percentile(distances, trim_percentile)
        keep = distances <= cutoff
        if int(keep.sum()) < 6:
            break
        r_delta, t_delta = best_fit_transform(current[keep], target[indices[keep]])
        current = current @ r_delta.T + t_delta.reshape(1, 3)
        r_total = r_delta @ r_total
        t_total = r_delta @ t_total + t_delta
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        history.append(
            {
                "iteration": float(iteration + 1),
                "rmse": rmse,
                "mean_distance": float(distances[keep].mean()),
                "median_distance": float(np.median(distances[keep])),
                "kept_correspondences": float(keep.sum()),
                "distance_cutoff": float(cutoff),
            }
        )
        if prev_rmse is not None and abs(prev_rmse - rmse) < 1.0e-8:
            break
        prev_rmse = rmse
    return r_total, t_total, history


def signed_angle_about_axis(rotation: np.ndarray, axis: np.ndarray) -> float:
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    sin_axis_vec = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) * 0.5
    sin_along_axis = float(np.dot(axis, sin_axis_vec))
    return float(math.atan2(sin_along_axis, cos_angle))


def rotation_angle(rotation: np.ndarray) -> float:
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cos_angle))


def joint_transform_numpy(origin: np.ndarray, axis: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    axis_t = torch.as_tensor(axis, dtype=torch.float64)
    q_t = torch.tensor(q, dtype=torch.float64)
    r = rodrigues(axis_t, q_t).cpu().numpy()
    t = origin - r @ origin
    return r, t


def save_point_cloud(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.9g} {y:.9g} {z:.9g}\n")


def mask_iou_from_images(render_rgb: Image.Image, mask: Image.Image, threshold: int = 5) -> float:
    support = np.asarray(render_rgb.convert("L"), dtype=np.float32) > threshold
    mask_b = np.asarray(mask.convert("L")) > 0
    union = np.logical_or(support, mask_b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(support, mask_b).sum() / union)


def render_candidate(
    q_value: float,
    gaussians,
    base_xyz: torch.Tensor,
    base_rotation: torch.Tensor,
    moving_mask: torch.Tensor,
    meta: dict[str, Any],
    rotation_mode: str,
    camera,
    bg: torch.Tensor,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    cfg: dict[str, Any],
    target_rgb_pil: Image.Image,
    target_mask_pil: Image.Image,
    moving_target_mask_pil: Image.Image,
) -> dict[str, Any]:
    q_tensor = torch.tensor(q_value, dtype=base_xyz.dtype, device=base_xyz.device)
    with torch.no_grad():
        xyz = deform_xyz_relative_continuous(base_xyz, moving_mask, meta["joint_origin"], meta["joint_axis"], q_tensor)
        rotation = None
        if rotation_mode == "rigid":
            rotation = deform_rotation_relative_continuous(base_rotation, moving_mask, meta["joint_axis"], q_tensor)
        rendered = render(camera, DeformedGaussian(gaussians, xyz, rotation), default_pipeline(), bg)["render"]
        moving_rendered = render(
            camera,
            OpacityMaskedGaussian(gaussians, xyz, rotation, moving_mask),
            default_pipeline(),
            bg,
        )["render"]
        loss, parts = masked_rgb_loss(rendered, target_rgb, target_mask, cfg)
    render_image = tensor_to_pil_rgb(rendered)
    moving_image = tensor_to_pil_rgb(moving_rendered)
    support = support_metrics(render_image, target_mask_pil)
    moving_iou = mask_iou_from_images(moving_image, moving_target_mask_pil)
    return {
        "q": float(q_value),
        "render_image": render_image,
        "moving_render_image": moving_image,
        "rgb_loss": parts["loss_l1"],
        "ssim": parts["ssim"],
        "ssim_loss": 1.0 - parts["ssim"],
        "total_loss": float(loss.detach().cpu()),
        "support_iou": support["support_mask_iou"],
        "moving_iou_if_available": moving_iou,
        "support_metrics": support,
        "target_rgb": target_rgb_pil,
        "target_mask": target_mask_pil,
        "moving_target_mask": moving_target_mask_pil,
    }


def save_candidate_outputs(result: dict[str, Any], out_dir: Path, label: str) -> None:
    q = float(result["q"])
    prefix = f"{label}_q_{q:+.6f}"
    save_overlays(result["render_image"], result["target_rgb"], result["target_mask"], out_dir, prefix)
    result["moving_render_image"].save(out_dir / f"{prefix}_moving_render.png")
    Image.blend(result["target_rgb"], result["moving_render_image"], 0.5).save(
        out_dir / f"{prefix}_moving_render_rgb_blend_50.png"
    )
    save_overlays(result["moving_render_image"], result["target_rgb"], result["moving_target_mask"], out_dir, f"{prefix}_moving")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate initial q_ref(0) from moving-part ICP, then validate by rendering.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--icp-iters", type=int, default=40)
    parser.add_argument("--trim-percentile", type=float, default=80.0)
    parser.add_argument("--mask-radius", type=int, default=5)
    parser.add_argument("--max-source-points", type=int, default=8000)
    parser.add_argument("--max-target-points", type=int, default=12000)
    parser.add_argument("--output-subdir", default="initial_q_icp_calibration")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    rotation_mode = str(cfg.get("rotation_mode", "rigid"))
    if rotation_mode not in {"none", "rigid"}:
        raise ValueError(f"Unsupported rotation_mode={rotation_mode}")
    source_path = ensure_colmap_source(cfg)
    out_dir = ensure_output_dir(
        Path(cfg["output_root"]) / args.output_subdir / f"cam_{args.cam:03d}" / f"frame_{args.frame:06d}"
    )

    gaussian_source = str(cfg.get("gaussian_source", "point_cloud"))
    model_ply = resolve_gaussian_ply(cfg["model_path"], int(cfg["iteration"]), gaussian_source)
    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]), gaussian_source=gaussian_source)
    meta = load_enriched_ply_metadata(model_ply)
    moving_mask = meta["part_ids"] == int(cfg["moving_part_id"])
    base_xyz = gaussians.get_xyz.detach()
    base_rotation = gaussians.get_rotation
    source_points_all = base_xyz[moving_mask].detach().cpu().numpy().astype(np.float64)
    source_points = deterministic_sample(source_points_all, args.max_source_points)

    cam_info = colmap_frame_camera(source_path, args.cam)
    rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, args.frame)
    mask_pil = load_mask_frame(cfg["mask_root"], args.cam, args.frame)
    full_mask = np.asarray(mask_pil.convert("L")) > 0
    depth_path = resolve_path(cfg.get("depth_root", "../dataset/usb_rgbdm/depth_npy")) / f"cam_{args.cam:03d}" / f"frame_{args.frame:06d}.npy"
    if not depth_path.exists():
        depth_path = resolve_path("../dataset/usb_rgbdm/depth_npy") / f"cam_{args.cam:03d}" / f"frame_{args.frame:06d}.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth npy for ICP target: {depth_path}")
    depth = np.load(depth_path).astype(np.float64)

    moving_target_mask, moving_mask_stats = draw_projected_mask(
        source_points_all,
        cam_info,
        args.mask_radius,
        full_mask,
        depth,
    )
    moving_mask_pil = Image.fromarray((moving_target_mask.astype(np.uint8) * 255))
    moving_mask_pil.save(out_dir / "approx_projected_moving_target_mask.png")
    mask_pil.save(out_dir / "target_full_mask.png")
    rgb_pil.save(out_dir / "target_rgb.png")

    target_points_all = unproject_depth_to_world(depth, moving_target_mask, cam_info)
    target_points = deterministic_sample(target_points_all, args.max_target_points)
    if len(source_points) < 6 or len(target_points) < 6:
        raise ValueError(
            f"Not enough ICP points: source={len(source_points)}, target={len(target_points)}. "
            "Check depth and approximate moving mask."
        )

    r_icp, t_icp, icp_history = run_icp(source_points, target_points, args.icp_iters, args.trim_percentile)
    transformed_source = source_points @ r_icp.T + t_icp.reshape(1, 3)
    save_point_cloud(out_dir / "moving_gaussian_source_sample.ply", source_points)
    save_point_cloud(out_dir / "observed_moving_target_sample.ply", target_points)
    save_point_cloud(out_dir / "moving_gaussian_source_after_icp_sample.ply", transformed_source)

    axis = meta["joint_axis"].detach().cpu().numpy().astype(np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
    origin = meta["joint_origin"].detach().cpu().numpy().astype(np.float64)
    q_from_rotation = signed_angle_about_axis(r_icp, axis)
    candidate_qs = [0.0, q_from_rotation, -q_from_rotation]
    candidate_qs = sorted({round(float(q), 12) for q in candidate_qs})

    camera = build_colmap_camera(source_path, cfg["rgb_root"], args.cam, args.frame)
    target_rgb_tensor = image_to_cuda_rgb(rgb_pil)
    target_mask_tensor = mask_to_cuda(mask_pil)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    candidate_results = [
        render_candidate(
            q,
            gaussians,
            base_xyz,
            base_rotation,
            moving_mask,
            meta,
            rotation_mode,
            camera,
            bg,
            target_rgb_tensor,
            target_mask_tensor,
            cfg,
            rgb_pil,
            mask_pil,
            moving_mask_pil,
        )
        for q in candidate_qs
    ]
    best_by_loss = min(candidate_results, key=lambda item: float(item["total_loss"]))
    best_by_iou = max(candidate_results, key=lambda item: float(item["support_iou"]))
    q_zero = min(candidate_results, key=lambda item: abs(float(item["q"])))

    for label, result in [
        ("q_zero", q_zero),
        ("q_icp_best_loss", best_by_loss),
        ("q_icp_best_iou", best_by_iou),
    ]:
        save_candidate_outputs(result, out_dir, label)

    rows = []
    for result in candidate_results:
        r_joint, t_joint = joint_transform_numpy(origin, axis, float(result["q"]))
        rot_residual = rotation_angle(r_icp @ r_joint.T)
        translation_residual = float(np.linalg.norm(t_icp - t_joint))
        rows.append(
            {
                "q_candidate": float(result["q"]),
                "candidate_kind": (
                    "q_zero"
                    if abs(float(result["q"])) < 1.0e-12
                    else "signed_icp_angle"
                    if float(result["q"]) == round(float(q_from_rotation), 12)
                    else "opposite_signed_icp_angle"
                ),
                "rgb_loss": result["rgb_loss"],
                "ssim": result["ssim"],
                "ssim_loss": result["ssim_loss"],
                "total_loss": result["total_loss"],
                "support_iou": result["support_iou"],
                "moving_iou_if_available": result["moving_iou_if_available"],
                "joint_translation_residual_to_icp": translation_residual,
                "joint_rotation_residual_to_icp_rad": rot_residual,
            }
        )
    save_csv(rows, out_dir / "initial_q_icp_candidate_summary.csv")
    save_csv(icp_history, out_dir / "icp_history.csv")

    total_loss_improvement = float(q_zero["total_loss"]) - float(best_by_loss["total_loss"])
    support_iou_improvement = float(best_by_iou["support_iou"]) - float(q_zero["support_iou"])
    q0_recommendation_threshold = 1.0e-4
    report = {
        "camera": f"cam_{args.cam:03d}",
        "frame": args.frame,
        "config": {
            "model_path": cfg["model_path"],
            "gaussian_source": gaussian_source,
            "gaussian_ply": str(model_ply),
            "rotation_mode": rotation_mode,
            "debug_image_shift_enabled": bool(cfg.get("debug_image_shift", {}).get("enabled", False)),
        },
        "limitation": (
            "No dense moving-part mask was found in usb_rgbdm. The target moving point cloud was built from "
            "depth pixels inside the full object mask and an approximate moving mask obtained by projecting "
            "moving Gaussian centers at q=0."
        ),
        "moving_mask_stats": moving_mask_stats,
        "point_counts": {
            "moving_gaussian_centers_all": int(len(source_points_all)),
            "moving_gaussian_centers_sampled": int(len(source_points)),
            "target_depth_points_all": int(len(target_points_all)),
            "target_depth_points_sampled": int(len(target_points)),
        },
        "icp": {
            "iterations_run": len(icp_history),
            "trim_percentile": args.trim_percentile,
            "R_icp": r_icp.tolist(),
            "t_icp": t_icp.tolist(),
            "T_icp_4x4": np.block([[r_icp, t_icp.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
            "final_rmse": icp_history[-1]["rmse"] if icp_history else None,
            "signed_angle_about_joint_axis": q_from_rotation,
        },
        "joint": {
            "joint_type_id": int(meta["joint_type_id"]),
            "joint_type_id_map": meta["joint_type_id_map"],
            "origin": origin.tolist(),
            "axis": axis.tolist(),
        },
        "validation": {
            "q_zero": {k: v for k, v in q_zero.items() if k not in {"render_image", "moving_render_image", "target_rgb", "target_mask", "moving_target_mask"}},
            "best_by_total_loss": {
                k: v
                for k, v in best_by_loss.items()
                if k not in {"render_image", "moving_render_image", "target_rgb", "target_mask", "moving_target_mask"}
            },
            "best_by_support_iou": {
                k: v
                for k, v in best_by_iou.items()
                if k not in {"render_image", "moving_render_image", "target_rgb", "target_mask", "moving_target_mask"}
            },
            "total_loss_improvement_vs_q0": total_loss_improvement,
            "support_iou_improvement_vs_q0": support_iou_improvement,
            "q0_from_icp": float(best_by_loss["q"]),
            "q_zero_already_near_optimal": abs(total_loss_improvement) < q0_recommendation_threshold,
        },
        "outputs": {
            "output_dir": str(out_dir),
            "source_point_cloud": str(out_dir / "moving_gaussian_source_sample.ply"),
            "target_point_cloud": str(out_dir / "observed_moving_target_sample.ply"),
            "icp_transformed_source": str(out_dir / "moving_gaussian_source_after_icp_sample.ply"),
            "candidate_csv": str(out_dir / "initial_q_icp_candidate_summary.csv"),
            "icp_history_csv": str(out_dir / "icp_history.csv"),
        },
    }
    save_json(report, out_dir / "initial_q_icp_report.json")

    print(f"output_dir={out_dir}")
    print(f"gaussian_ply={model_ply}")
    print(f"moving_target_limitation={report['limitation']}")
    print(f"source_points={len(source_points)} target_points={len(target_points)}")
    print(f"icp_iterations={len(icp_history)} final_rmse={report['icp']['final_rmse']}")
    print(f"signed_icp_angle_about_axis={q_from_rotation:+.8f}")
    print(f"best_q_by_total_loss={float(best_by_loss['q']):+.8f}")
    print(f"best_total_loss={float(best_by_loss['total_loss']):.8f}")
    print(f"q0_total_loss={float(q_zero['total_loss']):.8f}")
    print(f"total_loss_improvement_vs_q0={total_loss_improvement:.8f}")
    print(f"best_q_by_support_iou={float(best_by_iou['q']):+.8f}")
    print(f"best_support_iou={float(best_by_iou['support_iou']):.8f}")
    print(f"q0_support_iou={float(q_zero['support_iou']):.8f}")
    print(f"support_iou_improvement_vs_q0={support_iou_improvement:.8f}")
    print(f"q_zero_already_near_optimal={report['validation']['q_zero_already_near_optimal']}")


if __name__ == "__main__":
    main()
