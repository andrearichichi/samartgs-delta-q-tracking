from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from plyfile import PlyData
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render
from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from scripts.delta_q_tracking.io_utils import (
    build_colmap_camera,
    default_pipeline,
    ensure_colmap_source,
    load_enriched_ply_metadata,
    load_gaussian_model,
    load_mask_frame,
    load_rgb_frame,
    load_simple_yaml,
    resolve_path,
    resolve_gaussian_ply,
    save_json,
    support_metrics,
    tensor_to_pil_rgb,
)
from scripts.delta_q_tracking.trajectory_io import load_trajectory
from utils.graphics_utils import focal2fov


def read_xyz_ply(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    data = vertex.data
    xyz = np.stack([np.asarray(data[a], dtype=np.float64) for a in "xyz"], axis=1)
    extra = {name: np.asarray(data[name]) for name in data.dtype.names if name not in {"x", "y", "z"}}
    return xyz, extra, [str(c) for c in ply.comments]


def geometry_stats(points: np.ndarray) -> dict[str, Any]:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    size = bbox_max - bbox_min
    return {
        "count": int(points.shape[0]),
        "centroid": points.mean(axis=0).tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_size": size.tolist(),
    }


def nn_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    tree_b = cKDTree(b)
    dist_ab, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    dist_ba, _ = tree_a.query(b, k=1)
    return {
        "a_to_b_mean": float(np.mean(dist_ab)),
        "a_to_b_median": float(np.median(dist_ab)),
        "a_to_b_p95": float(np.percentile(dist_ab, 95)),
        "a_to_b_max": float(np.max(dist_ab)),
        "b_to_a_mean": float(np.mean(dist_ba)),
        "b_to_a_median": float(np.median(dist_ba)),
        "b_to_a_p95": float(np.percentile(dist_ba, 95)),
        "b_to_a_max": float(np.max(dist_ba)),
    }


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
    """Fit dst ~= scale * R @ src + t for paired point sets."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / src.shape[0]
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(s * d) / var_src) if var_src > 0 else float("nan")
    t = dst_mean - scale * (r @ src_mean)
    angle = math.degrees(math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))
    residual = dst - (scale * (src @ r.T) + t)
    return {
        "scale": scale,
        "rotation_angle_deg": angle,
        "translation": t.tolist(),
        "residual_mean": float(np.linalg.norm(residual, axis=1).mean()),
        "residual_median": float(np.median(np.linalg.norm(residual, axis=1))),
    }


def colmap_read(source: Path):
    sparse = source / "sparse" / "0"
    try:
        extr = read_extrinsics_binary(str(sparse / "images.bin"))
        intr = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        extr = read_extrinsics_text(str(sparse / "images.txt"))
        intr = read_intrinsics_text(str(sparse / "cameras.txt"))
    return extr, intr


def active_colmap_camera_params(source: Path, cam_idx: int) -> dict[str, Any]:
    extrinsics, intrinsics = colmap_read(source)
    wanted = f"cam_{cam_idx:03d}/frame_000000.png"
    extr = next(e for e in extrinsics.values() if e.name == wanted)
    intr = intrinsics[extr.camera_id]
    if intr.model == "PINHOLE":
        fx, fy, cx, cy = map(float, intr.params[:4])
    elif intr.model == "SIMPLE_PINHOLE":
        fx = fy = float(intr.params[0])
        cx, cy = map(float, intr.params[1:3])
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")
    r_w2c = qvec2rotmat(extr.qvec)
    t_w2c = np.asarray(extr.tvec, dtype=np.float64)
    center = -r_w2c.T @ t_w2c
    return {
        "name": wanted,
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "R_w2c": r_w2c,
        "t_w2c": t_w2c,
        "center": center,
        "qvec": np.asarray(extr.qvec, dtype=np.float64),
    }


def camera_from_w2c(params: dict[str, Any], rgb: Image.Image, image_name: str, uid: int) -> Camera:
    r_w2c = np.asarray(params["R_w2c"], dtype=np.float64)
    return Camera(
        resolution=(int(params["width"]), int(params["height"])),
        colmap_id=uid,
        R=r_w2c.T,
        T=np.asarray(params["t_w2c"], dtype=np.float64),
        FoVx=focal2fov(float(params["fx"]), int(params["width"])),
        FoVy=focal2fov(float(params["fy"]), int(params["height"])),
        depth_params=None,
        image=rgb,
        invdepthmap=None,
        image_name=image_name,
        uid=uid,
        data_device="cuda",
    )


def project_points(points: np.ndarray, cam: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(cam["R_w2c"], dtype=np.float64)
    t = np.asarray(cam["t_w2c"], dtype=np.float64)
    xyz_cam = points @ r.T + t
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    u = cam["fx"] * xyz_cam[:, 0] / z + cam["cx"]
    v = cam["fy"] * xyz_cam[:, 1] / z + cam["cy"]
    uv = np.stack([u, v], axis=1)
    inside = valid & (u >= 0) & (u < cam["width"]) & (v >= 0) & (v < cam["height"])
    return uv, inside


def point_projection_metrics(points: np.ndarray, cam: dict[str, Any], mask: Image.Image) -> dict[str, Any]:
    uv, inside = project_points(points, cam)
    pts = uv[inside]
    mask_arr = np.asarray(mask.convert("L")) > 0
    if len(pts) == 0:
        return {"projected_inside_count": 0}
    xy = np.round(pts).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, mask_arr.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, mask_arr.shape[0] - 1)
    in_mask = mask_arr[xy[:, 1], xy[:, 0]]
    return {
        "projected_inside_count": int(len(pts)),
        "bbox_xyxy": [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())],
        "centroid_xy": pts.mean(axis=0).tolist(),
        "fraction_inside_full_mask": float(np.mean(in_mask)),
    }


def draw_projection(base: Image.Image, points: np.ndarray, cam: dict[str, Any], path: Path, color=(0, 255, 255), radius=1, max_points=8000) -> None:
    out = base.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    uv, inside = project_points(points, cam)
    pts = uv[inside]
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    rgba = (*color, 190)
    for u, v in pts:
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=rgba)
    out.save(path)


def draw_projection_parts(base: Image.Image, points: np.ndarray, labels: np.ndarray, cam: dict[str, Any], path: Path) -> None:
    out = base.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    uv, inside = project_points(points, cam)
    colors = {0: (0, 255, 255, 180), 1: (255, 0, 0, 210), -1: (255, 255, 0, 150)}
    for (u, v), label, ok in zip(uv, labels, inside):
        if not ok:
            continue
        rgba = colors.get(int(label), (0, 255, 0, 160))
        draw.ellipse((u - 1.5, v - 1.5, u + 1.5, v + 1.5), fill=rgba)
    out.save(path)


def support_mask(render_rgb: Image.Image, threshold=5) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(render_rgb), dtype=np.float32) > threshold


def binary_overlay(mask_img: Image.Image, render_rgb: Image.Image, path: Path) -> None:
    mask = np.asarray(mask_img.convert("L")) > 0
    support = support_mask(render_rgb)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = [90, 90, 90]
    out[np.logical_and(mask, support)] = [255, 255, 255]
    out[np.logical_and(mask, ~support)] = [255, 40, 40]
    out[np.logical_and(~mask, support)] = [0, 220, 255]
    Image.fromarray(out).save(path)


def save_support(render_rgb: Image.Image, path: Path) -> Image.Image:
    img = Image.fromarray((support_mask(render_rgb).astype(np.uint8) * 255))
    img.save(path)
    return img


def contact_sheet(items: list[tuple[str, Image.Image]], path: Path, columns: int = 3) -> None:
    label_h = 28
    w, h = items[0][1].size
    rows = int(math.ceil(len(items) / columns))
    sheet = Image.new("RGB", (w * columns, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(items):
        col = idx % columns
        row = idx // columns
        x = col * w
        y = row * (h + label_h)
        draw.text((x + 8, y + 7), label, fill=(0, 0, 0))
        sheet.paste(image.convert("RGB"), (x, y + label_h))
    sheet.save(path)


class MaskedGaussian:
    def __init__(self, base, mask: torch.Tensor):
        self.base = base
        self.mask = mask
        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree
        self.optimizer = None
        self.spatial_lr_scale = base.spatial_lr_scale

    @property
    def get_xyz(self):
        return self.base.get_xyz[self.mask]

    @property
    def get_features(self):
        return self.base.get_features[self.mask]

    @property
    def get_opacity(self):
        return self.base.get_opacity[self.mask]

    @property
    def get_scaling(self):
        return self.base.get_scaling[self.mask]

    @property
    def get_rotation(self):
        return self.base.get_rotation[self.mask]

    def get_covariance(self, scaling_modifier=1):
        return self.base.get_covariance(scaling_modifier)[self.mask]


def render_and_metric(label: str, camera: Camera, gaussians, mask: Image.Image, out_dir: Path) -> dict[str, Any]:
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        image = tensor_to_pil_rgb(render(camera, gaussians, default_pipeline(), bg)["render"])
    image.save(out_dir / f"{label}.png")
    binary_overlay(mask, image, out_dir / f"{label}_mask_overlay.png")
    return support_metrics(image, mask)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a @ b.T
    return math.degrees(math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def load_json_camera_variants(path: Path, cam_idx: int, intrinsics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "cameras" in data:
        cam = data["cameras"][cam_idx]
        position = np.asarray(cam["position"], dtype=np.float64)
        target = np.asarray(cam["target"], dtype=np.float64)
        up = np.asarray(cam.get("up", [0, 0, 1]), dtype=np.float64)
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        # OpenCV-like camera axes: x right, y down, z forward.
        r_c2w = np.stack([right, -true_up, forward], axis=1)
        name = cam.get("name", f"cam_{cam_idx:03d}")
    else:
        cam = data[cam_idx]
        position = np.asarray(cam["position"], dtype=np.float64)
        r_c2w = np.asarray(cam["rotation"], dtype=np.float64)
        name = cam.get("img_name", f"cam_{cam_idx:03d}")
    r_as_c2w_w2c = r_c2w.T
    t_as_c2w = -r_as_c2w_w2c @ position
    r_as_w2c = r_c2w
    t_as_w2c = -r_as_w2c @ position
    common = {
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
    }
    return {
        f"{path.name}:{name}:as_c2w_inverted": {**common, "R_w2c": r_as_c2w_w2c, "t_w2c": t_as_c2w, "center": position},
        f"{path.name}:{name}:as_w2c_direct": {**common, "R_w2c": r_as_w2c, "t_w2c": t_as_w2c, "center": position},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive initial alignment debug for frame 0.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/alignment_debug/frame_000000")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "camera_variants").mkdir(exist_ok=True)
    (out_dir / "projections").mkdir(exist_ok=True)

    source = ensure_colmap_source(cfg)
    gaussian_source = str(cfg.get("gaussian_source", "point_cloud"))
    model_ply = resolve_gaussian_ply(cfg["model_path"], int(cfg["iteration"]), gaussian_source)
    input_ply = REPO_ROOT / "../dataset/usb_gauss/input.ply"
    points_parts_ply = REPO_ROOT / "../dataset/usb_rgbdm/pointcloud/points3D_parts.ply"

    rgb = load_rgb_frame(cfg["rgb_root"], args.cam, args.frame)
    mask = load_mask_frame(cfg["mask_root"], args.cam, args.frame)
    rgb.save(out_dir / "rgb_frame_000000.png")
    mask.save(out_dir / "mask_frame_000000.png")

    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]), gaussian_source=gaussian_source)
    meta = load_enriched_ply_metadata(model_ply)
    moving_mask = meta["part_ids"] == int(cfg["moving_part_id"])
    static_mask = meta["part_ids"] == int(cfg["static_part_id"])
    ignored_mask = meta["part_ids"] == int(cfg["ignored_part_id"])
    active_camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.frame)
    active_params = active_colmap_camera_params(source, args.cam)

    input_xyz, input_extra, input_comments = read_xyz_ply(input_ply)
    gauss_xyz, gauss_extra, gauss_comments = read_xyz_ply(model_ply)
    moving_np = np.asarray(gauss_extra["joint_part"]) == int(cfg["moving_part_id"]) if "joint_part" in gauss_extra else np.zeros(len(gauss_xyz), dtype=bool)
    static_np = np.asarray(gauss_extra["joint_part"]) == int(cfg["static_part_id"]) if "joint_part" in gauss_extra else np.zeros(len(gauss_xyz), dtype=bool)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        full_render = tensor_to_pil_rgb(render(active_camera, gaussians, default_pipeline(), bg)["render"])
        moving_render = tensor_to_pil_rgb(render(active_camera, MaskedGaussian(gaussians, moving_mask), default_pipeline(), bg)["render"])
        static_render = tensor_to_pil_rgb(render(active_camera, MaskedGaussian(gaussians, static_mask), default_pipeline(), bg)["render"])

    full_render.save(out_dir / "gaussian_render_deltaq0_frame_000000.png")
    moving_render.save(out_dir / "moving_part_gaussian_render_deltaq0_frame_000000.png")
    static_render.save(out_dir / "static_part_gaussian_render_deltaq0_frame_000000.png")
    Image.blend(rgb, full_render, 0.5).save(out_dir / "overlay_rgb_gaussian_deltaq0.png")
    binary_overlay(mask, full_render, out_dir / "overlay_mask_gaussian_support_deltaq0.png")
    binary_overlay(mask, moving_render, out_dir / "overlay_full_mask_moving_gaussian_support_deltaq0.png")
    full_support = save_support(full_render, out_dir / "gaussian_support_mask_deltaq0.png")
    moving_support = save_support(moving_render, out_dir / "moving_part_projected_support_mask_deltaq0.png")

    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    contact_sheet(
        [
            ("RGB frame 0", rgb),
            ("Full mask", mask_rgb),
            ("Gaussian delta_q=0", full_render),
            ("RGB+Gaussian", Image.blend(rgb, full_render, 0.5)),
            ("Mask+Gaussian", Image.open(out_dir / "overlay_mask_gaussian_support_deltaq0.png")),
            ("Moving Gaussian", moving_render),
        ],
        out_dir / "initial_alignment_contact_sheet.png",
        columns=3,
    )

    # Projection overlays.
    draw_projection(rgb, input_xyz, active_params, out_dir / "projections/projected_input_ply_on_rgb.png", color=(0, 255, 0), radius=1.2)
    draw_projection(rgb, gauss_xyz, active_params, out_dir / "projections/projected_gaussian_centers_on_rgb.png", color=(0, 220, 255), radius=0.8)
    draw_projection(rgb, gauss_xyz[moving_np], active_params, out_dir / "projections/projected_moving_gaussian_centers_on_rgb.png", color=(255, 0, 0), radius=1.0)
    draw_projection(rgb, gauss_xyz[static_np], active_params, out_dir / "projections/projected_static_gaussian_centers_on_rgb.png", color=(0, 220, 255), radius=1.0)

    part_projection = {"available": False}
    if points_parts_ply.exists():
        parts_xyz, parts_extra, parts_comments = read_xyz_ply(points_parts_ply)
        labels = np.asarray(parts_extra.get("part", np.zeros(len(parts_xyz))), dtype=int)
        draw_projection_parts(rgb, parts_xyz, labels, active_params, out_dir / "projections/projected_points3D_parts_on_rgb.png")
        draw_projection(rgb, parts_xyz[labels == 1], active_params, out_dir / "projections/projected_points3D_moving_part_on_rgb.png", color=(255, 0, 0), radius=1.5)
        part_projection = {
            "available": True,
            "path": str(points_parts_ply),
            "part_counts": {str(int(v)): int((labels == v).sum()) for v in sorted(set(labels.tolist()))},
            "all_points_projection": point_projection_metrics(parts_xyz, active_params, mask),
            "moving_points_projection": point_projection_metrics(parts_xyz[labels == 1], active_params, mask) if np.any(labels == 1) else None,
            "comments": parts_comments,
        }

    # Geometry comparisons.
    tree = cKDTree(gauss_xyz)
    dist, idx = tree.query(input_xyz, k=1)
    similarity = umeyama_similarity(input_xyz, gauss_xyz[idx])
    geometry_report = {
        "input_ply": {"path": str(input_ply), **geometry_stats(input_xyz), "comments": input_comments[:10]},
        "gaussian_ply": {"path": str(model_ply), **geometry_stats(gauss_xyz), "comments": gauss_comments[:10]},
        "moving_gaussians": geometry_stats(gauss_xyz[moving_np]) if np.any(moving_np) else None,
        "static_gaussians": geometry_stats(gauss_xyz[static_np]) if np.any(static_np) else None,
    }
    geometry_report["centroid_diff_gaussian_minus_input"] = (np.asarray(geometry_report["gaussian_ply"]["centroid"]) - np.asarray(geometry_report["input_ply"]["centroid"])).tolist()
    geometry_report["bbox_size_ratio_gaussian_over_input"] = (
        np.asarray(geometry_report["gaussian_ply"]["bbox_size"]) / np.maximum(np.asarray(geometry_report["input_ply"]["bbox_size"]), 1e-12)
    ).tolist()
    geometry_report["nearest_neighbor_input_vs_gaussian"] = nn_stats(input_xyz, gauss_xyz)
    geometry_report["umeyama_input_to_nearest_gaussian"] = similarity

    # Camera comparisons and variant renders.
    camera_report: dict[str, Any] = {
        "active_colmap": {
            "source_path": str(source),
            "image_name": active_params["name"],
            "width": active_params["width"],
            "height": active_params["height"],
            "fx": active_params["fx"],
            "fy": active_params["fy"],
            "cx": active_params["cx"],
            "cy": active_params["cy"],
            "R_w2c": active_params["R_w2c"].tolist(),
            "t_w2c": active_params["t_w2c"].tolist(),
            "camera_center": active_params["center"].tolist(),
        },
        "variants": {},
    }
    variant_params: dict[str, dict[str, Any]] = {}
    for json_path in [REPO_ROOT / "../dataset/usb_gauss/cameras.json", REPO_ROOT / "../dataset/usb_rgbdm/metadata/cameras.json"]:
        variant_params.update(load_json_camera_variants(json_path, args.cam, active_params))

    axis_flips = {
        "active_colmap_flip_x_axis": np.diag([-1.0, 1.0, 1.0]),
        "active_colmap_flip_y_axis": np.diag([1.0, -1.0, 1.0]),
        "active_colmap_flip_z_axis": np.diag([1.0, 1.0, -1.0]),
        "active_colmap_flip_yz_axes": np.diag([1.0, -1.0, -1.0]),
    }
    for name, flip in axis_flips.items():
        r = flip @ active_params["R_w2c"]
        variant_params[name] = {**{k: active_params[k] for k in ["width", "height", "fx", "fy", "cx", "cy"]}, "R_w2c": r, "t_w2c": active_params["t_w2c"], "center": active_params["center"]}

    for name, params in variant_params.items():
        center = np.asarray(params["center"], dtype=np.float64)
        camera_report["variants"][name] = {
            "camera_center": center.tolist(),
            "center_diff_from_active": float(np.linalg.norm(center - active_params["center"])),
            "rotation_diff_from_active_deg": rotation_angle_deg(np.asarray(params["R_w2c"]), active_params["R_w2c"]),
            "t_w2c": np.asarray(params["t_w2c"]).tolist(),
            "translation_diff_from_active": float(np.linalg.norm(np.asarray(params["t_w2c"]) - active_params["t_w2c"])),
        }
        try:
            cam = camera_from_w2c(params, rgb, name, 100 + len(camera_report["variants"]))
            camera_report["variants"][name]["render_metrics"] = render_and_metric(
                name.replace("/", "_").replace(":", "_"),
                cam,
                gaussians,
                mask,
                out_dir / "camera_variants",
            )
        except Exception as exc:
            camera_report["variants"][name]["render_error"] = repr(exc)

    image_flip_metrics = {}
    for name, image in {
        "active_render_hflip": ImageOps.mirror(full_render),
        "active_render_vflip": ImageOps.flip(full_render),
        "active_render_hvflip": ImageOps.flip(ImageOps.mirror(full_render)),
    }.items():
        image.save(out_dir / "camera_variants" / f"{name}.png")
        binary_overlay(mask, image, out_dir / "camera_variants" / f"{name}_mask_overlay.png")
        image_flip_metrics[name] = support_metrics(image, mask)
    camera_report["image_flip_metrics"] = image_flip_metrics

    projection_report = {
        "input_ply_on_active_camera": point_projection_metrics(input_xyz, active_params, mask),
        "gaussian_centers_on_active_camera": point_projection_metrics(gauss_xyz, active_params, mask),
        "moving_gaussian_centers_on_active_camera": point_projection_metrics(gauss_xyz[moving_np], active_params, mask) if np.any(moving_np) else None,
        "static_gaussian_centers_on_active_camera": point_projection_metrics(gauss_xyz[static_np], active_params, mask) if np.any(static_np) else None,
        "points3D_parts": part_projection,
    }

    overlap_report = {
        "full_render_vs_full_mask": support_metrics(full_render, mask),
        "moving_render_vs_full_mask": support_metrics(moving_render, mask),
        "static_render_vs_full_mask": support_metrics(static_render, mask),
        "moving_2d_mask_available": False,
        "moving_2d_mask_note": "No dense moving-part mask file was found in dataset/usb_rgbdm; only full-object mask is available. points3D_parts.ply is projected as sparse moving-part evidence.",
    }

    q_start = float(cfg.get("q_start", 0.0))
    trajectory_cfg = cfg.get("trajectory", {})
    trajectory = load_trajectory(
        resolve_path(trajectory_cfg["frame_values_path"]),
        str(trajectory_cfg["joint_value_column"]),
        str(trajectory_cfg.get("q_coordinate_mode", "relative_to_first_frame")),
        requested_start_frame=args.frame,
        requested_end_frame=args.frame,
    )
    frame_q = trajectory.q_absolute_by_frame.get(args.frame)
    pose_report = {
        "delta_q_rendered": 0.0,
        "articulation_applied": False,
        "q_start_config": q_start,
        "frame_000000_q_value_from_metadata": frame_q,
        "joint_type_id": int(meta["joint_type_id"]),
        "joint_type_id_map": meta.get("joint_type_id_map"),
        "joint_origin": meta["joint_origin"].detach().cpu().tolist(),
        "joint_axis": meta["joint_axis"].detach().cpu().tolist(),
        "part_counts_in_gaussian": {
            "moving": int(moving_mask.sum().detach().cpu()),
            "static": int(static_mask.sum().detach().cpu()),
            "ignored": int(ignored_mask.sum().detach().cpu()),
        },
        "segmentation_note": "3D part IDs are present in the Gaussian PLY. Dense 2D moving mask is not present; moving-part visual checks use moving Gaussian render and projected points3D_parts labels.",
    }

    # Numeric diagnosis heuristics.
    full_shift = overlap_report["full_render_vs_full_mask"]["centroid_shift_render_minus_mask_px"]
    input_proj = projection_report["input_ply_on_active_camera"]
    gauss_proj = projection_report["gaussian_centers_on_active_camera"]
    diagnosis = {
        "same_initial_image_pose": False,
        "primary_observation": "Gaussian delta_q=0 silhouette is shifted right/down relative to frame 000000 mask.",
        "render_centroid_shift_px": full_shift,
        "likely_causes_ranked": [],
    }
    if abs(full_shift[1]) > 15 and abs(full_shift[0]) > 5:
        diagnosis["likely_causes_ranked"].append("static camera/frame or 2D image-space alignment mismatch")
    if abs(geometry_report["umeyama_input_to_nearest_gaussian"]["rotation_angle_deg"]) > 5:
        diagnosis["likely_causes_ranked"].append("Gaussian centers are not a pure copy of input geometry; fitted input->Gaussian nearest-neighbor rotation is non-trivial")
    if any(abs(v - 1.0) > 0.08 for v in geometry_report["bbox_size_ratio_gaussian_over_input"]):
        diagnosis["likely_causes_ranked"].append("Gaussian geometry scale/bbox differs from input.ply")
    if input_proj.get("fraction_inside_full_mask", 0) < 0.5 and gauss_proj.get("fraction_inside_full_mask", 0) < 0.5:
        diagnosis["likely_causes_ranked"].append("active camera/projection does not align either input geometry or Gaussian centers well with the full mask")
    diagnosis["smallest_concrete_next_fix"] = "Fix/replace the static camera-frame alignment or model-source pose before trusting delta_q; do not solve this by increasing optimizer iterations."

    report = {
        "config": {
            "gaussian_source": gaussian_source,
            "model_ply": str(model_ply),
            "input_ply": str(input_ply),
            "rgb_frame": str(REPO_ROOT / cfg["rgb_root"] / f"cam_{args.cam:03d}" / f"frame_{args.frame:06d}.png"),
            "mask_frame": str(REPO_ROOT / cfg["mask_root"] / f"cam_{args.cam:03d}" / f"frame_{args.frame:06d}.png"),
            "output_dir": str(out_dir),
        },
        "geometry": geometry_report,
        "camera": camera_report,
        "projection": projection_report,
        "overlap": overlap_report,
        "pose_articulation": pose_report,
        "diagnosis": diagnosis,
        "visual_outputs": {
            "contact_sheet": str(out_dir / "initial_alignment_contact_sheet.png"),
            "projection_input": str(out_dir / "projections/projected_input_ply_on_rgb.png"),
            "projection_gaussian": str(out_dir / "projections/projected_gaussian_centers_on_rgb.png"),
            "camera_variants_dir": str(out_dir / "camera_variants"),
        },
    }
    save_json(report, out_dir / "alignment_debug_report.json")
    print(json.dumps(
        {
            "output_dir": str(out_dir),
            "full_render_iou": overlap_report["full_render_vs_full_mask"]["support_mask_iou"],
            "full_render_centroid_shift_px": full_shift,
            "input_projection_fraction_inside_mask": input_proj.get("fraction_inside_full_mask"),
            "gaussian_projection_fraction_inside_mask": gauss_proj.get("fraction_inside_full_mask"),
            "diagnosis": diagnosis,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
