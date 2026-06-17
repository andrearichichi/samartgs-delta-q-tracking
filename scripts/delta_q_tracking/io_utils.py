from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from plyfile import PlyData
from torchvision.utils import save_image

from scene.cameras import Camera
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import focal2fov


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Parse the small scalar YAML config used by these scripts."""
    out: dict[str, Any] = {}
    section: str | None = None
    for raw in resolve_path(path).read_text().splitlines():
        no_comment = raw.split("#", 1)[0].rstrip()
        line = no_comment.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if not raw[: len(raw) - len(raw.lstrip())] and value == "":
            section = key.strip()
            out[section] = {}
            continue
        if raw[: len(raw) - len(raw.lstrip())] and section:
            target = out[section]
        else:
            section = None
            target = out
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip("\"'")
        target[key.strip()] = parsed
    return out


def ensure_output_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_colmap_source(config: dict[str, Any]) -> Path:
    source = resolve_path(config["source_path"])
    sparse = source / "sparse" / "0"
    if (sparse / "cameras.txt").exists() and (sparse / "images.txt").exists():
        return source
    original = resolve_path(config.get("original_source_path", ""))
    if not original.exists():
        raise FileNotFoundError(f"Missing source_path {source} and original_source_path {original}")
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True, exist_ok=True)
    for child in original.iterdir():
        dst = source / child.name
        if child.is_dir():
            shutil.copytree(child, dst)
        else:
            shutil.copy2(child, dst)
    return source


def load_rgb_frame(rgb_root: str | Path, cam_idx: int, frame_idx: int) -> Image.Image:
    return Image.open(resolve_path(rgb_root) / f"cam_{cam_idx:03d}" / f"frame_{frame_idx:06d}.png").convert("RGB")


def load_mask_frame(mask_root: str | Path, cam_idx: int, frame_idx: int) -> Image.Image:
    return Image.open(resolve_path(mask_root) / f"cam_{cam_idx:03d}" / f"frame_{frame_idx:06d}.png").convert("L")


def image_to_cuda_rgb(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().cuda()


def mask_to_cuda(mask: Image.Image) -> torch.Tensor:
    arr = (np.asarray(mask.convert("L"), dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(arr)[None].cuda()


def resolve_gaussian_ply(model_path: str | Path, iteration: int, gaussian_source: str = "point_cloud") -> Path:
    iteration_dir = resolve_path(model_path) / "point_cloud" / f"iteration_{iteration}"
    source = str(gaussian_source)
    if source in {"point_cloud", "iteration", "iteration_30000"}:
        filename = "point_cloud.ply"
    elif source.endswith(".ply"):
        filename = source
    else:
        filename = f"{source}.ply"
    ply = iteration_dir / filename
    if not ply.exists():
        raise FileNotFoundError(f"Missing Gaussian PLY: {ply}")
    return ply


def load_gaussian_model(
    model_path: str | Path,
    iteration: int,
    sh_degree: int = 3,
    gaussian_source: str = "point_cloud",
) -> GaussianModel:
    ply = resolve_gaussian_ply(model_path, iteration, gaussian_source)
    model = GaussianModel(sh_degree)
    model.load_ply(str(ply))
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        tensor = getattr(model, name)
        tensor.requires_grad_(False)
    return model


def load_enriched_ply_metadata(ply_path: str | Path, device: torch.device | str = "cuda") -> dict[str, Any]:
    ply = PlyData.read(resolve_path(ply_path))
    v = ply["vertex"]
    names = set(v.data.dtype.names or [])
    required = {
        "joint_part",
        "joint_type_id",
        "joint_origin_x",
        "joint_origin_y",
        "joint_origin_z",
        "joint_axis_x",
        "joint_axis_y",
        "joint_axis_z",
    }
    missing = sorted(required - names)
    if missing:
        raise KeyError(f"PLY is missing enriched joint fields: {missing}")
    part_ids = torch.as_tensor(np.asarray(v["joint_part"]), dtype=torch.long, device=device)
    joint_type_ids = torch.as_tensor(np.asarray(v["joint_type_id"]), dtype=torch.long, device=device)
    origins = np.stack([np.asarray(v[f"joint_origin_{a}"]) for a in "xyz"], axis=1)
    axes = np.stack([np.asarray(v[f"joint_axis_{a}"]) for a in "xyz"], axis=1)
    origin = torch.as_tensor(origins[0], dtype=torch.float32, device=device)
    axis = torch.as_tensor(axes[0], dtype=torch.float32, device=device)
    comments = [str(c) for c in ply.comments]
    joint_metadata = None
    joint_type_id_map = None
    for c in comments:
        if c.startswith("joint_metadata "):
            joint_metadata = json.loads(c[len("joint_metadata ") :])
        if c.startswith("joint_type_id_map "):
            joint_type_id_map = json.loads(c[len("joint_type_id_map ") :])
    return {
        "part_ids": part_ids,
        "joint_type_ids": joint_type_ids,
        "joint_type_id": int(joint_type_ids[0].item()),
        "joint_origin": origin,
        "joint_axis": axis,
        "joint_metadata": joint_metadata,
        "joint_type_id_map": joint_type_id_map,
    }


def _read_colmap(source: Path):
    sparse = source / "sparse" / "0"
    try:
        extr = read_extrinsics_binary(str(sparse / "images.bin"))
        intr = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except Exception:
        extr = read_extrinsics_text(str(sparse / "images.txt"))
        intr = read_intrinsics_text(str(sparse / "cameras.txt"))
    return extr, intr


def build_colmap_camera(
    source_path: str | Path,
    rgb_root: str | Path,
    cam_idx: int,
    frame_idx: int,
    data_device: str = "cuda",
) -> Camera:
    source = resolve_path(source_path)
    extrinsics, intrinsics = _read_colmap(source)
    wanted = f"cam_{cam_idx:03d}/frame_000000.png"
    extr = next(e for e in extrinsics.values() if e.name == wanted)
    intr = intrinsics[extr.camera_id]
    if intr.model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise ValueError(f"Unsupported camera model {intr.model}")
    if intr.model == "PINHOLE":
        fx, fy = float(intr.params[0]), float(intr.params[1])
    else:
        fx = fy = float(intr.params[0])
    image = load_rgb_frame(rgb_root, cam_idx, frame_idx)
    return Camera(
        resolution=(intr.width, intr.height),
        colmap_id=intr.id,
        R=np.transpose(qvec2rotmat(extr.qvec)),
        T=np.asarray(extr.tvec),
        FoVx=focal2fov(fx, intr.width),
        FoVy=focal2fov(fy, intr.height),
        depth_params=None,
        image=image,
        invdepthmap=None,
        image_name=f"cam_{cam_idx:03d}/frame_{frame_idx:06d}.png",
        uid=cam_idx,
        data_device=data_device,
    )


def default_pipeline() -> SimpleNamespace:
    return SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)


def save_render_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.detach().clamp(0, 1), str(path))


def tensor_to_pil_rgb(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def support_metrics(render_rgb: Image.Image, mask: Image.Image, threshold: int = 5) -> dict[str, Any]:
    support = np.asarray(ImageOps.grayscale(render_rgb), dtype=np.float32) > threshold
    mask_b = np.asarray(mask.convert("L")) > 0
    inter = np.logical_and(support, mask_b).sum()
    union = np.logical_or(support, mask_b).sum()

    def stats(arr: np.ndarray) -> dict[str, Any]:
        ys, xs = np.where(arr)
        if len(xs) == 0:
            return {"pixels": 0, "bbox_xyxy": None, "centroid_xy": None}
        return {
            "pixels": int(arr.sum()),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "centroid_xy": [float(xs.mean()), float(ys.mean())],
        }

    rs = stats(support)
    ms = stats(mask_b)
    out = {
        "support_mask_iou": float(inter / union) if union else 0.0,
        "render_support": rs,
        "target_mask": ms,
    }
    if rs["centroid_xy"] is not None and ms["centroid_xy"] is not None:
        out.update(
            {
                "centroid_shift_render_minus_mask_px": [
                    rs["centroid_xy"][0] - ms["centroid_xy"][0],
                    rs["centroid_xy"][1] - ms["centroid_xy"][1],
                ],
                "bbox_top_left_shift_render_minus_mask_px": [
                    rs["bbox_xyxy"][0] - ms["bbox_xyxy"][0],
                    rs["bbox_xyxy"][1] - ms["bbox_xyxy"][1],
                ],
                "bbox_bottom_right_shift_render_minus_mask_px": [
                    rs["bbox_xyxy"][2] - ms["bbox_xyxy"][2],
                    rs["bbox_xyxy"][3] - ms["bbox_xyxy"][3],
                ],
            }
        )
    return out


def save_overlays(render_rgb: Image.Image, target_rgb: Image.Image, mask: Image.Image, out_dir: str | Path, prefix: str) -> None:
    out = ensure_output_dir(out_dir)
    render_rgb.save(out / f"{prefix}_render.png")
    Image.blend(target_rgb, render_rgb, 0.5).save(out / f"{prefix}_render_rgb_blend_50.png")
    mask_b = np.asarray(mask.convert("L")) > 0
    blend = np.asarray(Image.blend(target_rgb, render_rgb, 0.5)).copy()
    p = np.pad(mask_b, 1, mode="constant")
    eroded = p[1:-1, 1:-1].copy()
    h, w = mask_b.shape
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        eroded &= p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    edge = mask_b & ~eroded
    blend[edge] = [255, 0, 0]
    Image.fromarray(blend).save(out / f"{prefix}_render_rgb_mask_contour.png")
    support = np.asarray(ImageOps.grayscale(render_rgb), dtype=np.float32) > 5
    ov = np.zeros((h, w, 3), dtype=np.uint8)
    ov[np.logical_and(mask_b, support)] = [255, 255, 255]
    ov[np.logical_and(mask_b, ~support)] = [255, 0, 0]
    ov[np.logical_and(~mask_b, support)] = [0, 255, 255]
    Image.fromarray(ov).save(out / f"{prefix}_mask_vs_render_support.png")


def save_json(data: Any, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
