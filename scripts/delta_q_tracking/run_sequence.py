from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render

from scripts.delta_q_tracking.articulation import apply_articulation_transform
from scripts.delta_q_tracking.dataset_manifest import DatasetObject, load_dataset_manifest
from scripts.delta_q_tracking.deformed_gaussian import DeformedGaussian
from scripts.delta_q_tracking.io_utils import (
    build_colmap_camera,
    default_pipeline,
    ensure_colmap_source,
    ensure_output_dir,
    image_to_cuda_rgb,
    load_enriched_ply_metadata,
    load_gaussian_model,
    load_gaussian_model_from_ply,
    load_mask_frame,
    load_rgb_frame,
    load_simple_yaml,
    mask_to_cuda,
    resolve_path,
    resolve_gaussian_ply,
    save_csv,
    save_json,
    support_metrics,
    tensor_to_pil_rgb,
)
from scripts.delta_q_tracking.losses import loss_config, masked_rgb_loss
from scripts.delta_q_tracking.motion_mlp import MotionMLP
from scripts.delta_q_tracking.trajectory_io import load_trajectory


def camera_index(camera_id: str) -> int:
    prefix = "cam_"
    if not camera_id.startswith(prefix) or not camera_id[len(prefix) :].isdigit():
        raise ValueError(f"Invalid camera ID {camera_id!r}; expected cam_NNN")
    return int(camera_id[len(prefix) :])


def apply_manifest_object_to_config(
    cfg: dict,
    dataset_object: DatasetObject,
    gaussian_model_override: str | None,
) -> tuple[dict, Path]:
    gaussian_ply = dataset_object.require_gaussian_model(gaussian_model_override)
    if gaussian_ply.suffix.lower() != ".ply":
        raise ValueError(
            "Manifest-based execution currently expects gaussian_model_path or "
            f"--gaussian-model-override to reference an enriched .ply file, got {gaussian_ply}"
        )
    cfg = dict(cfg)
    cfg.update(
        {
            "rgb_root": str(dataset_object.rgb_dir),
            "mask_root": str(dataset_object.mask_dir),
            "source_path": str(dataset_object.colmap_path),
            "original_source_path": str(dataset_object.colmap_path),
            "output_root": f"outputs/delta_q_tracking/new_dataset/{dataset_object.object_id}",
            "moving_part_ids": list(dataset_object.moving_part_ids),
            "static_part_ids": list(dataset_object.static_part_ids),
            "joint_type_original": dataset_object.joint_type_original,
            "joint_type_normalized": dataset_object.joint_type_normalized,
            "joint_name": dataset_object.joint_name,
            "joint_metadata_path": str(dataset_object.joint_metadata_path),
            "gaussian_ply_override": str(gaussian_ply),
            "q_start": 0.0,
            "trajectory": {
                "frame_values_path": str(dataset_object.trajectory_path),
                "joint_value_column": dataset_object.trajectory_joint_column,
                "q_coordinate_mode": "relative_to_first_frame",
            },
        }
    )
    return cfg, gaussian_ply


def load_external_joint_metadata(cfg: dict, gaussian_meta: dict) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    joint_type = str(cfg.get("joint_type_normalized", "revolute")).lower()
    metadata_path = cfg.get("joint_metadata_path")
    if metadata_path is None:
        return joint_type, gaussian_meta["joint_axis"], gaussian_meta["joint_origin"]

    payload = json.loads(resolve_path(metadata_path).read_text())
    metadata_name = str(payload.get("name", ""))
    if cfg.get("joint_name") and metadata_name != str(cfg["joint_name"]):
        raise ValueError(
            f"Joint metadata name {metadata_name!r} does not match manifest {cfg['joint_name']!r}"
        )
    original_type = str(payload.get("type", "")).lower()
    if cfg.get("joint_type_original") and original_type != str(cfg["joint_type_original"]).lower():
        raise ValueError(
            f"Joint metadata type {original_type!r} does not match manifest "
            f"{cfg['joint_type_original']!r}"
        )
    axis_values = payload.get("axis_vector")
    pivot_values = payload.get("point_of_application")
    if axis_values is None:
        raise ValueError(f"Joint metadata is missing axis_vector: {metadata_path}")
    if joint_type == "revolute" and pivot_values is None:
        raise ValueError(f"Revolute joint metadata is missing point_of_application: {metadata_path}")

    reference_axis = gaussian_meta["joint_axis"]
    axis = torch.as_tensor(axis_values, dtype=reference_axis.dtype, device=reference_axis.device)
    pivot = None
    if pivot_values is not None:
        reference_pivot = gaussian_meta["joint_origin"]
        pivot = torch.as_tensor(
            pivot_values,
            dtype=reference_pivot.dtype,
            device=reference_pivot.device,
        )
    if not torch.allclose(axis, reference_axis, atol=1.0e-6, rtol=1.0e-6):
        raise ValueError(
            f"Manifest joint axis {axis.tolist()} does not match enriched Gaussian axis "
            f"{reference_axis.tolist()}"
        )
    if pivot is not None and not torch.allclose(
        pivot, gaussian_meta["joint_origin"], atol=1.0e-6, rtol=1.0e-6
    ):
        raise ValueError(
            f"Manifest joint pivot {pivot.tolist()} does not match enriched Gaussian pivot "
            f"{gaussian_meta['joint_origin'].tolist()}"
        )
    return joint_type, axis, pivot


def save_sequence_images(
    raw_img: Image.Image,
    target_rgb: Image.Image,
    out_dir: Path,
    target_frame: int,
) -> None:
    raw_img.save(out_dir / f"pred_raw_frame_{target_frame:06d}.png")
    Image.blend(target_rgb, raw_img, 0.5).save(out_dir / f"overlay_raw_frame_{target_frame:06d}.png")


def _state_comments(comments: list[str], frame_index: int, q_ref: float, last_delta_q: float) -> list[str]:
    preserved = [
        comment
        for comment in comments
        if not (
            comment.startswith("articulated_state ")
            or comment.startswith("q_ref ")
            or comment.startswith("last_delta_q ")
            or comment.startswith("frame_index ")
        )
    ]
    state = {
        "frame": int(frame_index),
        "q_ref": float(q_ref),
        "last_delta_q": float(last_delta_q),
    }
    preserved.extend(
        [
            f"articulated_state {json.dumps(state, separators=(',', ':'))}",
            f"q_ref {float(q_ref):.12g}",
            f"last_delta_q {float(last_delta_q):.12g}",
            f"frame_index {int(frame_index)}",
        ]
    )
    return preserved


def export_articulated_state_ply(
    source_ply: Path,
    export_root: Path,
    frame_index: int,
    q_ref: float,
    last_delta_q: float,
    xyz: torch.Tensor,
    rotation: torch.Tensor | None,
) -> dict[str, object]:
    """Export a self-contained articulated Gaussian state without adding per-vertex dynamic fields."""
    ply = PlyData.read(str(source_ply))
    vertex = ply["vertex"].data.copy()
    names = set(vertex.dtype.names or [])
    original_fields = list(vertex.dtype.names or [])
    forbidden_dynamic_fields = ["q_ref", "last_delta_q", "frame_index"]
    forbidden_present = [name for name in forbidden_dynamic_fields if name in names]
    if forbidden_present:
        raise ValueError(f"Source/export vertex properties unexpectedly contain dynamic state fields: {forbidden_present}")

    xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
    if len(vertex) != len(xyz_np):
        raise ValueError(f"PLY vertex count {len(vertex)} does not match xyz count {len(xyz_np)}")
    for idx, key in enumerate(("x", "y", "z")):
        if key not in names:
            raise KeyError(f"PLY is missing required position property {key}")
        vertex[key] = xyz_np[:, idx].astype(vertex[key].dtype, copy=False)

    if rotation is not None:
        rot_np = rotation.detach().cpu().numpy().astype(np.float32)
        for idx, key in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
            if key not in names:
                raise KeyError(f"PLY is missing required rotation property {key}")
            vertex[key] = rot_np[:, idx].astype(vertex[key].dtype, copy=False)

    frame_dir = export_root / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    out_ply = frame_dir / "point_cloud.ply"
    comments = _state_comments([str(comment) for comment in ply.comments], frame_index, q_ref, last_delta_q)
    PlyData(
        [PlyElement.describe(vertex, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=comments,
        obj_info=ply.obj_info,
    ).write(str(out_ply))

    state = {
        "frame_index": int(frame_index),
        "q_ref": float(q_ref),
        "last_delta_q": float(last_delta_q),
        "source_ply": str(source_ply),
        "exported_ply": str(out_ply),
        "articulated_state_comment": {
            "frame": int(frame_index),
            "q_ref": float(q_ref),
            "last_delta_q": float(last_delta_q),
        },
        "dynamic_state_storage": "global_ply_header_comments_and_state_metadata_json",
        "no_per_vertex_dynamic_state_properties": True,
        "original_vertex_properties_preserved": original_fields,
        "structural_articulation_properties_present": {
            name: name in names
            for name in [
                "joint_part",
                "joint_type_id",
                "joint_origin_x",
                "joint_origin_y",
                "joint_origin_z",
                "joint_axis_x",
                "joint_axis_y",
                "joint_axis_z",
                "joint_urdf_origin_x",
                "joint_urdf_origin_y",
                "joint_urdf_origin_z",
                "joint_urdf_rpy_r",
                "joint_urdf_rpy_p",
                "joint_urdf_rpy_y",
                "joint_urdf_axis_x",
                "joint_urdf_axis_y",
                "joint_urdf_axis_z",
                "motion_state",
                "semantics",
            ]
        },
        "preserved_original_comment_count": len(ply.comments),
        "comments_added": {
            "articulated_state": True,
            "q_ref": True,
            "last_delta_q": True,
            "frame_index": True,
        },
    }
    save_json(state, frame_dir / "state_metadata.json")
    return state


def time_tensor(
    frame_index: int,
    start_frame: int,
    num_frames: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if num_frames <= 1:
        value = 0.0
    else:
        value = float(frame_index - start_frame) / float(num_frames - 1)
    return torch.tensor([[value]], dtype=dtype, device=device)


def mlp_regularization_loss(
    motion_mlp: MotionMLP,
    times: torch.Tensor,
    smoothness_weight: float,
    acceleration_weight: float,
    monotonic_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    q_values = motion_mlp(times).reshape(-1)
    zero = q_values.sum() * 0.0
    smoothness = zero
    acceleration = zero
    monotonic = zero
    delta_q_values = q_values[1:] - q_values[:-1] if q_values.numel() >= 2 else q_values[:0]
    if smoothness_weight > 0.0 and q_values.numel() >= 3:
        smoothness = ((delta_q_values[1:] - delta_q_values[:-1]) ** 2).mean()
    if acceleration_weight > 0.0 and q_values.numel() >= 3:
        acceleration = ((q_values[2:] - 2.0 * q_values[1:-1] + q_values[:-2]) ** 2).mean()
    if monotonic_weight > 0.0 and delta_q_values.numel() > 0:
        monotonic = torch.relu(-delta_q_values).pow(2).mean()
    total = smoothness_weight * smoothness + acceleration_weight * acceleration + monotonic_weight * monotonic
    return total, {
        "mlp_smoothness_loss": float(smoothness.detach().cpu()),
        "mlp_acceleration_loss": float(acceleration.detach().cpu()),
        "mlp_monotonic_loss": float(monotonic.detach().cpu()),
        "mlp_regularization_loss": float(total.detach().cpu()),
    }


def parameter_grad_norm(parameters) -> tuple[float, bool, bool]:
    total_sq = 0.0
    finite = True
    nonzero = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all().item())
        value = float(torch.linalg.vector_norm(grad).detach().cpu())
        total_sq += value * value
        nonzero = nonzero or value > 1.0e-10
    return total_sq ** 0.5, finite, nonzero


def initialize_motion_mlp(
    hidden_dim: int,
    num_layers: int,
    q_start: float,
    time_encoding: str,
    fourier_frequencies: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> MotionMLP:
    motion_mlp = MotionMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        time_encoding=time_encoding,
        fourier_frequencies=fourier_frequencies,
    ).to(device=device, dtype=dtype)
    last_linear = next(module for module in reversed(motion_mlp.net) if isinstance(module, torch.nn.Linear))
    torch.nn.init.zeros_(last_linear.weight)
    torch.nn.init.constant_(last_linear.bias, q_start)
    return motion_mlp


def optimize_step(
    cfg: dict,
    gaussians,
    base_xyz: torch.Tensor,
    moving_mask: torch.Tensor,
    pivot: torch.Tensor | None,
    axis: torch.Tensor,
    joint_type: str,
    q_start: float,
    q_ref: float,
    rotation_mode: str,
    camera,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    source_frame: int,
    target_frame: int,
    q_gt_t: float | None,
    q_gt_t1: float | None,
    gt_delta_q: float | None,
    required_delta_to_gt: float | None,
    previous_committed_delta_q: float | None,
) -> dict:
    max_iters = int(cfg["num_iters"])
    use_best_loss_delta_q = bool(cfg.get("use_best_loss_delta_q", False))
    early_cfg = cfg.get("early_stopping", {})
    if not isinstance(early_cfg, dict):
        early_cfg = {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 30))
    early_min_delta = float(early_cfg.get("min_delta", 1.0e-6))
    early_restore_best = bool(early_cfg.get("restore_best", True))
    temporal_cfg = cfg.get("temporal_delta_regularization", {})
    if not isinstance(temporal_cfg, dict):
        temporal_cfg = {}
    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    temporal_mode = str(temporal_cfg.get("mode", "previous_delta_q"))
    temporal_lambda = float(temporal_cfg.get("lambda_temporal_delta", 0.0))
    apply_temporal = temporal_enabled and temporal_lambda > 0.0 and previous_committed_delta_q is not None
    if temporal_enabled and temporal_mode != "previous_delta_q":
        raise ValueError(f"Unsupported temporal_delta_regularization.mode={temporal_mode}")
    delta_q = torch.nn.Parameter(torch.zeros((), dtype=base_xyz.dtype, device=base_xyz.device))
    optimizer = torch.optim.Adam([delta_q], lr=float(cfg["lr"]))
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    loss_curve: list[float] = []
    l1_curve: list[float] = []
    ssim_curve: list[float] = []
    delta_curve: list[float] = []
    grad_curve: list[float] = []
    per_iteration_rows: list[dict[str, object]] = []
    grad_finite = True
    best_loss = float("inf")
    best_iteration = 0
    best_delta_q = 0.0
    best_rgb_loss = 0.0
    best_ssim = 0.0
    no_improve_count = 0
    early_stopping_triggered = False

    for iteration_idx in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        loss_eval_delta_q = float(delta_q.detach().cpu())
        q_pred = torch.as_tensor(q_ref - q_start, dtype=base_xyz.dtype, device=base_xyz.device) + delta_q
        transform = apply_articulation_transform(
            joint_type,
            base_xyz,
            gaussians.get_rotation if rotation_mode == "rigid" else None,
            moving_mask,
            axis,
            pivot,
            q_pred,
        )
        raw_render = render(
            camera,
            DeformedGaussian(gaussians, transform.points, transform.rotations),
            default_pipeline(),
            bg,
        )["render"]
        image_loss, loss_parts = masked_rgb_loss(raw_render, target_rgb, target_mask, cfg)
        if apply_temporal:
            prev_delta = torch.as_tensor(previous_committed_delta_q, dtype=delta_q.dtype, device=delta_q.device)
            temporal_delta_loss = (delta_q - prev_delta) ** 2
            loss = image_loss + temporal_lambda * temporal_delta_loss
            temporal_delta_loss_value = float(temporal_delta_loss.detach().cpu())
        else:
            loss = image_loss
            temporal_delta_loss_value = 0.0
        loss_value = float(loss.detach().cpu())
        image_loss_value = float(image_loss.detach().cpu())
        improved = loss_value < best_loss - early_min_delta
        if improved:
            best_loss = loss_value
            best_iteration = iteration_idx + 1
            best_delta_q = loss_eval_delta_q
            best_rgb_loss = float(loss_parts["loss_l1"])
            best_ssim = float(loss_parts["ssim"])
            no_improve_count = 0
        else:
            no_improve_count += 1
        loss.backward()
        if delta_q.grad is None:
            raise RuntimeError("delta_q.grad is None; deformation is not connected to the loss")
        finite = bool(torch.isfinite(delta_q.grad).all().item())
        grad_finite = grad_finite and finite
        if not finite:
            raise RuntimeError("delta_q.grad is not finite")
        grad = float(delta_q.grad.detach().cpu())
        optimizer.step()
        delta_q_after_step = float(delta_q.detach().cpu())
        delta_error_vs_gt_increment = None if gt_delta_q is None else loss_eval_delta_q - gt_delta_q
        delta_error_vs_required = None if required_delta_to_gt is None else loss_eval_delta_q - required_delta_to_gt
        q_ref_error_after_iteration = None if q_gt_t1 is None else q_ref + loss_eval_delta_q - q_gt_t1
        per_iteration_rows.append(
            {
                "frame_from": source_frame,
                "frame_to": target_frame,
                "iteration": iteration_idx + 1,
                "total_loss": loss_value,
                "image_loss": image_loss_value,
                "temporal_delta_loss": temporal_delta_loss_value,
                "lambda_temporal_delta": temporal_lambda if apply_temporal else 0.0,
                "rgb_loss": loss_parts["loss_l1"],
                "ssim_loss": 1.0 - loss_parts["ssim"],
                "pred_delta_q": loss_eval_delta_q,
                "delta_q_after_step": delta_q_after_step,
                "q_ref_start": q_ref,
                "q_ref_pred": q_ref + loss_eval_delta_q,
                "q_gt_t": "" if q_gt_t is None else q_gt_t,
                "q_gt_t1": "" if q_gt_t1 is None else q_gt_t1,
                "gt_delta_q": "" if gt_delta_q is None else gt_delta_q,
                "required_delta_to_GT": "" if required_delta_to_gt is None else required_delta_to_gt,
                "delta_error_vs_gt_increment": "" if delta_error_vs_gt_increment is None else delta_error_vs_gt_increment,
                "delta_error_vs_required": "" if delta_error_vs_required is None else delta_error_vs_required,
                "q_ref_error_after_iteration": "" if q_ref_error_after_iteration is None else q_ref_error_after_iteration,
                "delta_q_error": "" if delta_error_vs_gt_increment is None else delta_error_vs_gt_increment,
                "abs_delta_q_error": "" if delta_error_vs_gt_increment is None else abs(delta_error_vs_gt_increment),
                "grad_value": grad,
                "grad_norm": abs(grad),
                "grad_finite": finite,
                "grad_nonzero": abs(grad) > 1e-10,
            }
        )

        loss_curve.append(loss_value)
        l1_curve.append(loss_parts["loss_l1"])
        ssim_curve.append(loss_parts["ssim"])
        delta_curve.append(loss_eval_delta_q)
        grad_curve.append(grad)
        if early_enabled and no_improve_count >= early_patience:
            early_stopping_triggered = True
            break

    final_iteration_delta_q = float(delta_q.detach().cpu())
    if use_best_loss_delta_q or (early_stopping_triggered and early_restore_best):
        committed_delta_q = best_delta_q
        commit_source = "best_loss"
        committed_loss = best_loss
    else:
        committed_delta_q = final_iteration_delta_q
        commit_source = "final_iteration"
        committed_loss = loss_curve[-1]
    final_delta_q_error = None if gt_delta_q is None else final_iteration_delta_q - gt_delta_q
    best_delta_q_error = None if gt_delta_q is None else best_delta_q - gt_delta_q
    committed_delta_q_error = None if gt_delta_q is None else committed_delta_q - gt_delta_q
    committed_delta_error_vs_required = None if required_delta_to_gt is None else committed_delta_q - required_delta_to_gt

    with torch.no_grad():
        final_angle = torch.as_tensor(
            q_ref + committed_delta_q - q_start,
            dtype=base_xyz.dtype,
            device=base_xyz.device,
        )
        final_transform = apply_articulation_transform(
            joint_type,
            base_xyz,
            gaussians.get_rotation if rotation_mode == "rigid" else None,
            moving_mask,
            axis,
            pivot,
            final_angle,
        )
        final_raw = render(
            camera,
            DeformedGaussian(gaussians, final_transform.points, final_transform.rotations),
            default_pipeline(),
            bg,
        )["render"]

    return {
        "delta_q": committed_delta_q,
        "committed_delta_q": committed_delta_q,
        "commit_source": commit_source,
        "final_iteration_delta_q": final_iteration_delta_q,
        "final_iteration_loss": loss_curve[-1],
        "best_loss_delta_q": best_delta_q,
        "best_loss": best_loss,
        "best_iteration": best_iteration,
        "best_rgb_loss": best_rgb_loss,
        "best_ssim": best_ssim,
        "final_delta_q_error": "" if final_delta_q_error is None else final_delta_q_error,
        "best_loss_delta_q_error": "" if best_delta_q_error is None else best_delta_q_error,
        "committed_delta_q_error": "" if committed_delta_q_error is None else committed_delta_q_error,
        "committed_delta_error_vs_required": "" if committed_delta_error_vs_required is None else committed_delta_error_vs_required,
        "committed_loss": committed_loss,
        "final_loss": committed_loss,
        "max_iters": max_iters,
        "iterations_run": len(loss_curve),
        "early_stopping": early_stopping_triggered,
        "early_stopping_enabled": early_enabled,
        "early_stopping_patience": early_patience,
        "early_stopping_min_delta": early_min_delta,
        "early_stopping_restore_best": early_restore_best,
        "use_best_loss_delta_q": use_best_loss_delta_q,
        "temporal_delta_regularization": {
            "enabled": temporal_enabled,
            "applied": apply_temporal,
            "lambda_temporal_delta": temporal_lambda,
            "mode": temporal_mode,
            "previous_committed_delta_q": "" if previous_committed_delta_q is None else previous_committed_delta_q,
        },
        "loss_curve": loss_curve,
        "l1_curve": l1_curve,
        "ssim_curve": ssim_curve,
        "delta_curve": delta_curve,
        "grad_curve": grad_curve,
        "per_iteration_rows": per_iteration_rows,
        "grad_nonzero": any(abs(g) > 1e-10 for g in grad_curve),
        "grad_finite": grad_finite,
        "final_raw": final_raw.detach(),
        "final_xyz": final_transform.points.detach(),
        "final_rotation": (
            None if final_transform.rotations is None else final_transform.rotations.detach()
        ),
    }


def optimize_step_mlp(
    cfg: dict,
    motion_mlp: MotionMLP,
    optimizer: torch.optim.Optimizer,
    all_times: torch.Tensor,
    gaussians,
    base_xyz: torch.Tensor,
    moving_mask: torch.Tensor,
    pivot: torch.Tensor | None,
    axis: torch.Tensor,
    joint_type: str,
    q_start: float,
    rotation_mode: str,
    camera,
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    source_frame: int,
    target_frame: int,
    t0: torch.Tensor,
    t1: torch.Tensor,
    q_gt_t: float | None,
    q_gt_t1: float | None,
    gt_delta_q: float | None,
) -> dict:
    max_iters = int(cfg["num_iters"])
    use_best_loss_delta_q = bool(cfg.get("use_best_loss_delta_q", False))
    early_cfg = cfg.get("early_stopping", {})
    if not isinstance(early_cfg, dict):
        early_cfg = {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 30))
    early_min_delta = float(early_cfg.get("min_delta", 1.0e-6))
    early_restore_best = bool(early_cfg.get("restore_best", True))
    smoothness_weight = float(cfg.get("mlp_smoothness_weight", 0.0))
    acceleration_weight = float(cfg.get("mlp_acceleration_weight", 0.0))
    monotonic_weight = float(cfg.get("mlp_monotonic_weight", 0.0))
    apply_mlp_regularization = smoothness_weight > 0.0 or acceleration_weight > 0.0 or monotonic_weight > 0.0

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    loss_curve: list[float] = []
    l1_curve: list[float] = []
    ssim_curve: list[float] = []
    delta_curve: list[float] = []
    grad_curve: list[float] = []
    per_iteration_rows: list[dict[str, object]] = []
    grad_finite = True
    grad_nonzero = False
    best_loss = float("inf")
    best_iteration = 0
    best_delta_q = 0.0
    best_q0 = 0.0
    best_q1 = 0.0
    best_rgb_loss = 0.0
    best_ssim = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve_count = 0
    early_stopping_triggered = False
    final_iteration_delta_q = 0.0
    final_iteration_q0 = 0.0
    final_iteration_q1 = 0.0

    for iteration_idx in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        q0_tensor = motion_mlp(t0).reshape(())
        q1_tensor = motion_mlp(t1).reshape(())
        delta_q = q1_tensor - q0_tensor
        q_pred = q1_tensor - torch.as_tensor(q_start, dtype=base_xyz.dtype, device=base_xyz.device)
        loss_eval_delta_q = float(delta_q.detach().cpu())
        q0_value = float(q0_tensor.detach().cpu())
        q1_value = float(q1_tensor.detach().cpu())
        transform = apply_articulation_transform(
            joint_type,
            base_xyz,
            gaussians.get_rotation if rotation_mode == "rigid" else None,
            moving_mask,
            axis,
            pivot,
            q_pred,
        )
        raw_render = render(
            camera,
            DeformedGaussian(gaussians, transform.points, transform.rotations),
            default_pipeline(),
            bg,
        )["render"]
        image_loss, loss_parts = masked_rgb_loss(raw_render, target_rgb, target_mask, cfg)
        reg_loss, reg_parts = mlp_regularization_loss(
            motion_mlp,
            all_times,
            smoothness_weight,
            acceleration_weight,
            monotonic_weight,
        )
        loss = image_loss + reg_loss
        loss_value = float(loss.detach().cpu())
        image_loss_value = float(image_loss.detach().cpu())
        improved = loss_value < best_loss - early_min_delta
        if improved:
            best_loss = loss_value
            best_iteration = iteration_idx + 1
            best_delta_q = loss_eval_delta_q
            best_q0 = q0_value
            best_q1 = q1_value
            best_rgb_loss = float(loss_parts["loss_l1"])
            best_ssim = float(loss_parts["ssim"])
            best_state = {
                key: value.detach().clone()
                for key, value in motion_mlp.state_dict().items()
            }
            no_improve_count = 0
        else:
            no_improve_count += 1
        loss.backward()
        grad_norm, finite, nonzero = parameter_grad_norm(motion_mlp.parameters())
        grad_finite = grad_finite and finite
        grad_nonzero = grad_nonzero or nonzero
        if not finite:
            raise RuntimeError("MotionMLP parameter gradients are not finite")
        optimizer.step()
        with torch.no_grad():
            delta_q_after_step = float((motion_mlp(t1).reshape(()) - motion_mlp(t0).reshape(())).detach().cpu())
            final_iteration_delta_q = delta_q_after_step
            final_iteration_q0 = float(motion_mlp(t0).reshape(()).detach().cpu())
            final_iteration_q1 = float(motion_mlp(t1).reshape(()).detach().cpu())
        delta_error_vs_gt_increment = None if gt_delta_q is None else loss_eval_delta_q - gt_delta_q
        required_delta_to_gt = None if q_gt_t1 is None else q_gt_t1 - q0_value
        delta_error_vs_required = None if required_delta_to_gt is None else loss_eval_delta_q - required_delta_to_gt
        q_ref_error_after_iteration = None if q_gt_t1 is None else q1_value - q_gt_t1
        per_iteration_rows.append(
            {
                "motion_param": "mlp_q",
                "frame_from": source_frame,
                "frame_to": target_frame,
                "iteration": iteration_idx + 1,
                "total_loss": loss_value,
                "image_loss": image_loss_value,
                "temporal_delta_loss": 0.0,
                "lambda_temporal_delta": 0.0,
                "mlp_smoothness_loss": reg_parts["mlp_smoothness_loss"],
                "mlp_acceleration_loss": reg_parts["mlp_acceleration_loss"],
                "mlp_monotonic_loss": reg_parts["mlp_monotonic_loss"],
                "mlp_regularization_loss": reg_parts["mlp_regularization_loss"],
                "mlp_smoothness_weight": smoothness_weight,
                "mlp_acceleration_weight": acceleration_weight,
                "mlp_monotonic_weight": monotonic_weight,
                "rgb_loss": loss_parts["loss_l1"],
                "ssim_loss": 1.0 - loss_parts["ssim"],
                "pred_delta_q": loss_eval_delta_q,
                "delta_q_pred": loss_eval_delta_q,
                "delta_q_after_step": delta_q_after_step,
                "q_ref_start": q0_value,
                "q_ref_pred": q1_value,
                "q_ref_pred_after_step": final_iteration_q1,
                "q_gt_t": "" if q_gt_t is None else q_gt_t,
                "q_gt_t1": "" if q_gt_t1 is None else q_gt_t1,
                "gt_delta_q": "" if gt_delta_q is None else gt_delta_q,
                "required_delta_to_GT": "" if required_delta_to_gt is None else required_delta_to_gt,
                "delta_error_vs_gt_increment": "" if delta_error_vs_gt_increment is None else delta_error_vs_gt_increment,
                "delta_error_vs_required": "" if delta_error_vs_required is None else delta_error_vs_required,
                "q_ref_error_after_iteration": "" if q_ref_error_after_iteration is None else q_ref_error_after_iteration,
                "delta_q_error": "" if delta_error_vs_gt_increment is None else delta_error_vs_gt_increment,
                "abs_delta_q_error": "" if delta_error_vs_gt_increment is None else abs(delta_error_vs_gt_increment),
                "grad_value": grad_norm,
                "grad_norm": grad_norm,
                "grad_finite": finite,
                "grad_nonzero": nonzero,
            }
        )

        loss_curve.append(loss_value)
        l1_curve.append(loss_parts["loss_l1"])
        ssim_curve.append(loss_parts["ssim"])
        delta_curve.append(loss_eval_delta_q)
        grad_curve.append(grad_norm)
        if early_enabled and no_improve_count >= early_patience:
            early_stopping_triggered = True
            break

    if use_best_loss_delta_q or (early_stopping_triggered and early_restore_best):
        if best_state is not None:
            motion_mlp.load_state_dict(best_state)
        committed_delta_q = best_delta_q
        q_ref_start = best_q0
        q_ref_committed = best_q1
        commit_source = "best_loss"
        committed_loss = best_loss
    else:
        committed_delta_q = final_iteration_delta_q
        q_ref_start = final_iteration_q0
        q_ref_committed = final_iteration_q1
        commit_source = "final_iteration"
        committed_loss = loss_curve[-1]

    final_delta_q_error = None if gt_delta_q is None else final_iteration_delta_q - gt_delta_q
    best_delta_q_error = None if gt_delta_q is None else best_delta_q - gt_delta_q
    committed_delta_q_error = None if gt_delta_q is None else committed_delta_q - gt_delta_q
    required_delta_to_gt = None if q_gt_t1 is None else q_gt_t1 - q_ref_start
    committed_delta_error_vs_required = None if required_delta_to_gt is None else committed_delta_q - required_delta_to_gt

    with torch.no_grad():
        final_angle = torch.as_tensor(
            q_ref_committed - q_start,
            dtype=base_xyz.dtype,
            device=base_xyz.device,
        )
        final_transform = apply_articulation_transform(
            joint_type,
            base_xyz,
            gaussians.get_rotation if rotation_mode == "rigid" else None,
            moving_mask,
            axis,
            pivot,
            final_angle,
        )
        final_raw = render(
            camera,
            DeformedGaussian(gaussians, final_transform.points, final_transform.rotations),
            default_pipeline(),
            bg,
        )["render"]

    return {
        "motion_param": "mlp_q",
        "delta_q": committed_delta_q,
        "delta_q_pred": committed_delta_q,
        "committed_delta_q": committed_delta_q,
        "commit_source": commit_source,
        "final_iteration_delta_q": final_iteration_delta_q,
        "final_iteration_loss": loss_curve[-1],
        "best_loss_delta_q": best_delta_q,
        "best_loss": best_loss,
        "best_iteration": best_iteration,
        "best_rgb_loss": best_rgb_loss,
        "best_ssim": best_ssim,
        "final_delta_q_error": "" if final_delta_q_error is None else final_delta_q_error,
        "best_loss_delta_q_error": "" if best_delta_q_error is None else best_delta_q_error,
        "committed_delta_q_error": "" if committed_delta_q_error is None else committed_delta_q_error,
        "committed_delta_error_vs_required": "" if committed_delta_error_vs_required is None else committed_delta_error_vs_required,
        "required_delta_to_GT": "" if required_delta_to_gt is None else required_delta_to_gt,
        "q_ref_start": q_ref_start,
        "q_ref": q_ref_committed,
        "q_ref_pred": q_ref_committed,
        "q_ref_committed": q_ref_committed,
        "committed_loss": committed_loss,
        "final_loss": committed_loss,
        "max_iters": max_iters,
        "iterations_run": len(loss_curve),
        "early_stopping": early_stopping_triggered,
        "early_stopping_enabled": early_enabled,
        "early_stopping_patience": early_patience,
        "early_stopping_min_delta": early_min_delta,
        "early_stopping_restore_best": early_restore_best,
        "use_best_loss_delta_q": use_best_loss_delta_q,
        "temporal_delta_regularization": {
            "enabled": False,
            "applied": False,
            "lambda_temporal_delta": 0.0,
            "mode": "mlp_q_regularization",
            "previous_committed_delta_q": "",
        },
        "mlp_regularization": {
            "enabled": apply_mlp_regularization,
            "smoothness_weight": smoothness_weight,
            "acceleration_weight": acceleration_weight,
            "monotonic_weight": monotonic_weight,
        },
        "loss_curve": loss_curve,
        "l1_curve": l1_curve,
        "ssim_curve": ssim_curve,
        "delta_curve": delta_curve,
        "grad_curve": grad_curve,
        "per_iteration_rows": per_iteration_rows,
        "grad_nonzero": grad_nonzero,
        "grad_finite": grad_finite,
        "final_raw": final_raw.detach(),
        "final_xyz": final_transform.points.detach(),
        "final_rotation": (
            None if final_transform.rotations is None else final_transform.rotations.detach()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final rigid delta-q tracking sequence.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--manifest", default=None, help="Optional multi-object dataset manifest JSON.")
    parser.add_argument("--object-id", default=None, help="Manifest object_id or object_name.")
    parser.add_argument("--camera-id", default=None, help="Manifest camera ID such as cam_000.")
    parser.add_argument(
        "--gaussian-model-override",
        default=None,
        help="Optional enriched Gaussian PLY overriding the manifest model.",
    )
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=59)
    parser.add_argument("--output-subdir", default=None, help="Override output subfolder under output_root.")
    parser.add_argument("--num-iters", type=int, default=None, help="Override config num_iters.")
    parser.add_argument("--use-best-loss-delta-q", action="store_true", help="Commit the minimum-loss delta_q for each frame.")
    parser.add_argument("--use-final-iteration-delta-q", action="store_true", help="Commit the final optimizer value for each frame.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable optional loss-based early stopping.")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable optional loss-based early stopping.")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--temporal-delta-reg", action="store_true", help="Enable temporal delta_q regularization.")
    parser.add_argument("--no-temporal-delta-reg", action="store_true", help="Disable temporal delta_q regularization.")
    parser.add_argument("--lambda-temporal-delta", type=float, default=None)
    parser.add_argument(
        "--motion-param",
        choices=("direct_delta_q", "mlp_q"),
        default="direct_delta_q",
        help="Motion parameterization to optimize.",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-num-layers", type=int, default=2)
    parser.add_argument("--mlp-lr", type=float, default=1.0e-3)
    parser.add_argument("--mlp-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--mlp-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--mlp-time-encoding", choices=("raw", "fourier"), default="raw")
    parser.add_argument("--mlp-fourier-frequencies", type=int, default=4)
    parser.add_argument("--mlp-monotonic-weight", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    cfg["motion_param"] = args.motion_param
    cfg["mlp_hidden_dim"] = args.mlp_hidden_dim
    cfg["mlp_num_layers"] = args.mlp_num_layers
    cfg["mlp_lr"] = args.mlp_lr
    cfg["mlp_smoothness_weight"] = args.mlp_smoothness_weight
    cfg["mlp_acceleration_weight"] = args.mlp_acceleration_weight
    cfg["mlp_time_encoding"] = args.mlp_time_encoding
    cfg["mlp_fourier_frequencies"] = args.mlp_fourier_frequencies
    cfg["mlp_monotonic_weight"] = args.mlp_monotonic_weight
    dataset_object: DatasetObject | None = None
    manifest_path: Path | None = None
    gaussian_ply_override: Path | None = None
    if args.manifest is not None:
        if not args.object_id:
            parser.error("--object-id is required when --manifest is provided")
        manifest = load_dataset_manifest(args.manifest)
        manifest_path = manifest.path
        dataset_object = manifest.get(args.object_id)
        errors = dataset_object.validate_paths(require_gaussian=False)
        if errors:
            raise FileNotFoundError("Manifest object validation failed:\n- " + "\n- ".join(errors))
        cfg, gaussian_ply_override = apply_manifest_object_to_config(
            cfg,
            dataset_object,
            args.gaussian_model_override,
        )
        selected_camera_id = args.camera_id or f"cam_{args.cam:03d}"
        if selected_camera_id not in dataset_object.available_cameras:
            raise ValueError(
                f"Camera {selected_camera_id!r} is not available for {dataset_object.object_id}; "
                f"available: {dataset_object.available_cameras}"
            )
        args.cam = camera_index(selected_camera_id)
    elif args.camera_id is not None:
        args.cam = camera_index(args.camera_id)
    if args.num_iters is not None:
        cfg["num_iters"] = args.num_iters
    if args.use_best_loss_delta_q:
        cfg["use_best_loss_delta_q"] = True
    if args.use_final_iteration_delta_q:
        cfg["use_best_loss_delta_q"] = False
    early_cfg = cfg.setdefault("early_stopping", {})
    if not isinstance(early_cfg, dict):
        early_cfg = {}
        cfg["early_stopping"] = early_cfg
    if args.early_stopping:
        early_cfg["enabled"] = True
    if args.no_early_stopping:
        early_cfg["enabled"] = False
    if args.early_stopping_patience is not None:
        early_cfg["patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        early_cfg["min_delta"] = args.early_stopping_min_delta
    temporal_cfg = cfg.setdefault("temporal_delta_regularization", {})
    if not isinstance(temporal_cfg, dict):
        temporal_cfg = {}
        cfg["temporal_delta_regularization"] = temporal_cfg
    if args.temporal_delta_reg:
        temporal_cfg["enabled"] = True
    if args.no_temporal_delta_reg:
        temporal_cfg["enabled"] = False
    if args.lambda_temporal_delta is not None:
        temporal_cfg["lambda_temporal_delta"] = args.lambda_temporal_delta
    rotation_mode = str(cfg.get("rotation_mode", "none"))
    if rotation_mode not in {"none", "rigid"}:
        raise ValueError(f"Unsupported rotation_mode={rotation_mode}")
    print("WARNING: sequence results are only trustworthy if static alignment is geometrically valid.")

    source = ensure_colmap_source(cfg)
    output_subdir = args.output_subdir or ("03_sequence_rigid" if rotation_mode == "rigid" else "03_sequence")
    out_dir = ensure_output_dir(Path(cfg["output_root"]) / output_subdir / f"cam_{args.cam:03d}")
    gaussian_source = str(cfg.get("gaussian_source", "point_cloud"))
    model_ply = (
        gaussian_ply_override
        if gaussian_ply_override is not None
        else resolve_gaussian_ply(cfg["model_path"], int(cfg["iteration"]), gaussian_source)
    )

    print(f"gaussian_source={gaussian_source}")
    print(f"gaussian_ply={model_ply}")
    gaussians = (
        load_gaussian_model_from_ply(model_ply)
        if gaussian_ply_override is not None
        else load_gaussian_model(
            cfg["model_path"],
            int(cfg["iteration"]),
            gaussian_source=gaussian_source,
        )
    )
    meta = load_enriched_ply_metadata(model_ply)
    joint_type, joint_axis, joint_pivot = load_external_joint_metadata(cfg, meta)
    moving_part_ids = [
        int(value)
        for value in cfg.get("moving_part_ids", [cfg.get("moving_part_id", 1)])
    ]
    moving_mask = torch.zeros_like(meta["part_ids"], dtype=torch.bool)
    for part_id in moving_part_ids:
        moving_mask |= meta["part_ids"] == part_id
    if not bool(moving_mask.any().item()):
        raise ValueError(
            f"No moving Gaussians found for configured moving_part_ids={moving_part_ids}"
        )
    base_xyz = gaussians.get_xyz.detach().clone()
    q_start = float(cfg.get("q_start", 0.0))
    q_ref = q_start
    motion_param = str(cfg.get("motion_param", "direct_delta_q"))
    if motion_param not in {"direct_delta_q", "mlp_q"}:
        raise ValueError(f"Unsupported motion_param={motion_param!r}")
    num_sequence_frames = args.end_frame - args.start_frame + 1
    motion_mlp: MotionMLP | None = None
    motion_optimizer: torch.optim.Optimizer | None = None
    mlp_time_grid: torch.Tensor | None = None
    mlp_metadata = {
        "motion_param": motion_param,
        "hidden_dim": int(cfg["mlp_hidden_dim"]),
        "num_layers": int(cfg["mlp_num_layers"]),
        "lr": float(cfg["mlp_lr"]),
        "smoothness_weight": float(cfg["mlp_smoothness_weight"]),
        "acceleration_weight": float(cfg["mlp_acceleration_weight"]),
        "time_encoding": str(cfg["mlp_time_encoding"]),
        "fourier_frequencies": int(cfg["mlp_fourier_frequencies"]),
        "monotonic_weight": float(cfg["mlp_monotonic_weight"]),
    }
    if motion_param == "mlp_q":
        motion_mlp = initialize_motion_mlp(
            int(cfg["mlp_hidden_dim"]),
            int(cfg["mlp_num_layers"]),
            q_start,
            str(cfg["mlp_time_encoding"]),
            int(cfg["mlp_fourier_frequencies"]),
            dtype=base_xyz.dtype,
            device=base_xyz.device,
        )
        motion_optimizer = torch.optim.Adam(motion_mlp.parameters(), lr=float(cfg["mlp_lr"]))
        mlp_time_grid = torch.cat(
            [
                time_tensor(
                    frame,
                    args.start_frame,
                    num_sequence_frames,
                    dtype=base_xyz.dtype,
                    device=base_xyz.device,
                )
                for frame in range(args.start_frame, args.end_frame + 1)
            ],
            dim=0,
        )
    trajectory = []
    iteration_logs = []
    trajectory_cfg = cfg.get("trajectory", {})
    if not isinstance(trajectory_cfg, dict):
        raise ValueError("trajectory config must be a mapping")
    trajectory_data = load_trajectory(
        resolve_path(trajectory_cfg["frame_values_path"]),
        str(trajectory_cfg["joint_value_column"]),
        str(trajectory_cfg.get("q_coordinate_mode", "relative_to_first_frame")),
        requested_start_frame=args.start_frame,
        requested_end_frame=args.end_frame,
    )
    gt_q_by_frame = trajectory_data.q_by_frame
    gt_available = True
    print(f"trajectory={trajectory_data.metadata()}")
    print(f"joint_type={joint_type}")
    print(f"joint_axis={joint_axis.detach().cpu().tolist()}")
    print(
        "joint_pivot="
        f"{None if joint_pivot is None else joint_pivot.detach().cpu().tolist()}"
    )
    print(f"gt_delta_q_available={gt_available}")
    print(f"rotation_mode={rotation_mode}")
    print(f"num_iters={int(cfg['num_iters'])}")
    print(f"use_best_loss_delta_q={bool(cfg.get('use_best_loss_delta_q', False))}")
    print(f"early_stopping={cfg.get('early_stopping', {})}")
    print(f"temporal_delta_regularization={cfg.get('temporal_delta_regularization', {})}")
    print(f"motion_param={motion_param}")
    if motion_param == "mlp_q":
        print(f"motion_mlp={mlp_metadata}")
    previous_committed_delta_q: float | None = None
    deformed_export_root = ensure_output_dir(out_dir / "deformed_gaussians")
    deformed_export_entries: list[dict[str, object]] = []

    with torch.no_grad():
        initial_camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.start_frame)
        initial_rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, args.start_frame)
        if motion_param == "mlp_q":
            assert motion_mlp is not None
            initial_t = time_tensor(
                args.start_frame,
                args.start_frame,
                num_sequence_frames,
                dtype=base_xyz.dtype,
                device=base_xyz.device,
            )
            q_ref = float(motion_mlp(initial_t).reshape(()).detach().cpu())
        initial_displacement = torch.as_tensor(
            q_ref - q_start,
            dtype=base_xyz.dtype,
            device=base_xyz.device,
        )
        initial_transform = apply_articulation_transform(
            joint_type,
            base_xyz,
            gaussians.get_rotation if rotation_mode == "rigid" else None,
            moving_mask,
            joint_axis,
            joint_pivot,
            initial_displacement,
        )
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        initial_raw = render(
            initial_camera,
            DeformedGaussian(
                gaussians,
                initial_transform.points,
                initial_transform.rotations,
            ),
            default_pipeline(),
            bg,
        )["render"]
        save_sequence_images(
            tensor_to_pil_rgb(initial_raw),
            initial_rgb_pil,
            out_dir,
            args.start_frame,
        )
        deformed_export_entries.append(
            export_articulated_state_ply(
                model_ply,
                deformed_export_root,
                args.start_frame,
                q_ref,
                0.0,
                initial_transform.points,
                initial_transform.rotations,
            )
        )

    for source_frame in range(args.start_frame, args.end_frame):
        target_frame = source_frame + 1
        print(f"frame {source_frame:06d}->{target_frame:06d}: optimizing {motion_param}")
        camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, target_frame)
        target_rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, target_frame)
        target_mask_pil = load_mask_frame(cfg["mask_root"], args.cam, target_frame)
        target_rgb = image_to_cuda_rgb(target_rgb_pil)
        target_mask = mask_to_cuda(target_mask_pil)
        if motion_param == "mlp_q":
            assert motion_mlp is not None
            t_source = time_tensor(
                source_frame,
                args.start_frame,
                num_sequence_frames,
                dtype=base_xyz.dtype,
                device=base_xyz.device,
            )
            with torch.no_grad():
                q_ref_start = float(motion_mlp(t_source).reshape(()).detach().cpu())
        else:
            q_ref_start = q_ref
        q_gt_t = gt_q_by_frame.get(source_frame)
        q_gt_t1 = gt_q_by_frame.get(target_frame)
        gt_delta_q = None if q_gt_t is None or q_gt_t1 is None else q_gt_t1 - q_gt_t
        required_delta_to_gt = None if q_gt_t1 is None else q_gt_t1 - q_ref_start

        if motion_param == "mlp_q":
            assert motion_mlp is not None
            assert motion_optimizer is not None
            assert mlp_time_grid is not None
            t_target = time_tensor(
                target_frame,
                args.start_frame,
                num_sequence_frames,
                dtype=base_xyz.dtype,
                device=base_xyz.device,
            )
            step = optimize_step_mlp(
                cfg,
                motion_mlp,
                motion_optimizer,
                mlp_time_grid,
                gaussians,
                base_xyz,
                moving_mask,
                joint_pivot,
                joint_axis,
                joint_type,
                q_start,
                rotation_mode,
                camera,
                target_rgb,
                target_mask,
                source_frame,
                target_frame,
                t_source,
                t_target,
                q_gt_t,
                q_gt_t1,
                gt_delta_q,
            )
            q_ref_start = step["q_ref_start"]
            q_ref = step["q_ref_committed"]
            required_delta_to_gt = (
                None if step["required_delta_to_GT"] == "" else step["required_delta_to_GT"]
            )
        else:
            step = optimize_step(
                cfg,
                gaussians,
                base_xyz,
                moving_mask,
                joint_pivot,
                joint_axis,
                joint_type,
                q_start,
                q_ref,
                rotation_mode,
                camera,
                target_rgb,
                target_mask,
                source_frame,
                target_frame,
                q_gt_t,
                q_gt_t1,
                gt_delta_q,
                required_delta_to_gt,
                previous_committed_delta_q,
            )
            q_ref += step["committed_delta_q"]
        previous_committed_delta_q = step["committed_delta_q"]
        deformed_export_entries.append(
            export_articulated_state_ply(
                model_ply,
                deformed_export_root,
                target_frame,
                q_ref,
                step["committed_delta_q"],
                step["final_xyz"],
                step["final_rotation"],
            )
        )
        raw_img = tensor_to_pil_rgb(step["final_raw"])
        save_sequence_images(raw_img, target_rgb_pil, out_dir, target_frame)
        support = support_metrics(raw_img, target_mask_pil)

        loss_payload = {
            "source_frame": source_frame,
            "target_frame": target_frame,
            "max_iters": step["max_iters"],
            "iterations_run": step["iterations_run"],
            "early_stopping": step["early_stopping"],
            "loss_curve": step["loss_curve"],
            "l1_curve": step["l1_curve"],
            "ssim_curve": step["ssim_curve"],
            "delta_curve": step["delta_curve"],
            "grad_curve": step["grad_curve"],
        }
        save_json(loss_payload, out_dir / f"loss_curve_frame_{target_frame:06d}.json")
        per_iteration_dir = ensure_output_dir(out_dir / "per_iteration_logs")
        per_iteration_path = per_iteration_dir / f"{source_frame:06d}_to_{target_frame:06d}_iterations.csv"
        save_csv(step["per_iteration_rows"], per_iteration_path)
        save_json(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "support": support,
            },
            out_dir / f"metrics_frame_{target_frame:06d}.json",
        )

        item = {
            "motion_param": motion_param,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "delta_q": step["delta_q"],
            "delta_q_pred": step.get("delta_q_pred", step["committed_delta_q"]),
            "pred_delta_q": step["committed_delta_q"],
            "committed_delta_q": step["committed_delta_q"],
            "commit_source": step["commit_source"],
            "final_iteration_delta_q": step["final_iteration_delta_q"],
            "final_iteration_loss": step["final_iteration_loss"],
            "best_loss_delta_q": step["best_loss_delta_q"],
            "best_loss": step["best_loss"],
            "best_iteration": step["best_iteration"],
            "final_delta_q_error": step["final_delta_q_error"],
            "best_loss_delta_q_error": step["best_loss_delta_q_error"],
            "committed_delta_q_error": step["committed_delta_q_error"],
            "q_ref_start": q_ref_start,
            "q_ref": q_ref,
            "q_ref_pred": step.get("q_ref_pred", q_ref),
            "q_ref_committed": q_ref,
            "q_gt_t": "" if q_gt_t is None else q_gt_t,
            "q_gt_t1": "" if q_gt_t1 is None else q_gt_t1,
            "required_delta_to_GT": "" if required_delta_to_gt is None else required_delta_to_gt,
            "delta_error_vs_gt_increment": step["committed_delta_q_error"],
            "delta_error_vs_required": step["committed_delta_error_vs_required"],
            "q_ref_error_after_commit": "" if q_gt_t1 is None else q_ref - q_gt_t1,
            "q_start": q_start,
            "joint_displacement_from_base": q_ref - q_start,
            "angle_from_base": q_ref - q_start if joint_type == "revolute" else "",
            "joint_type": joint_type,
            "final_loss": step["final_loss"],
            "grad_nonzero": step["grad_nonzero"],
            "grad_finite": step["grad_finite"],
            "num_iters": step["max_iters"],
            "max_iters": step["max_iters"],
            "iterations_run": step["iterations_run"],
            "early_stopping": step["early_stopping"],
            "early_stopping_enabled": step["early_stopping_enabled"],
            "use_best_loss_delta_q": step["use_best_loss_delta_q"],
            "temporal_delta_regularization": step["temporal_delta_regularization"],
            "mlp_regularization": step.get("mlp_regularization", ""),
            "mlp_hidden_dim": int(cfg["mlp_hidden_dim"]) if motion_param == "mlp_q" else "",
            "mlp_num_layers": int(cfg["mlp_num_layers"]) if motion_param == "mlp_q" else "",
            "mlp_lr": float(cfg["mlp_lr"]) if motion_param == "mlp_q" else "",
            "mlp_smoothness_weight": float(cfg["mlp_smoothness_weight"]) if motion_param == "mlp_q" else "",
            "mlp_acceleration_weight": float(cfg["mlp_acceleration_weight"]) if motion_param == "mlp_q" else "",
            "mlp_time_encoding": str(cfg["mlp_time_encoding"]) if motion_param == "mlp_q" else "",
            "mlp_fourier_frequencies": int(cfg["mlp_fourier_frequencies"]) if motion_param == "mlp_q" else "",
            "mlp_monotonic_weight": float(cfg["mlp_monotonic_weight"]) if motion_param == "mlp_q" else "",
            "rotation_mode": rotation_mode,
            "gaussian_source": gaussian_source,
            "gt_delta_q": "" if gt_delta_q is None else gt_delta_q,
            "support_iou": support["support_mask_iou"],
            "raw_iou": support["support_mask_iou"],
        }
        trajectory.append(item)
        iteration_log = {
            "motion_param": motion_param,
            "frame_index": target_frame,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "transition": f"{source_frame:06d}->{target_frame:06d}",
            "max_iters": step["max_iters"],
            "iterations_run": step["iterations_run"],
            "final_loss": step["final_loss"],
            "delta_q": step["delta_q"],
            "delta_q_pred": step.get("delta_q_pred", step["committed_delta_q"]),
            "pred_delta_q": step["committed_delta_q"],
            "committed_delta_q": step["committed_delta_q"],
            "commit_source": step["commit_source"],
            "final_iteration_delta_q": step["final_iteration_delta_q"],
            "final_iteration_loss": step["final_iteration_loss"],
            "best_loss_delta_q": step["best_loss_delta_q"],
            "best_loss": step["best_loss"],
            "best_iteration": step["best_iteration"],
            "q_ref_start": q_ref_start,
            "q_ref_pred": step.get("q_ref_pred", q_ref),
            "q_ref_committed": q_ref,
            "q_gt_t": "" if q_gt_t is None else q_gt_t,
            "q_gt_t1": "" if q_gt_t1 is None else q_gt_t1,
            "gt_delta_q": "" if gt_delta_q is None else gt_delta_q,
            "required_delta_to_GT": "" if required_delta_to_gt is None else required_delta_to_gt,
            "delta_error_vs_gt_increment": "" if gt_delta_q is None else step["delta_q"] - gt_delta_q,
            "delta_error_vs_required": "" if required_delta_to_gt is None else step["delta_q"] - required_delta_to_gt,
            "q_ref_error_after_commit": "" if q_gt_t1 is None else q_ref - q_gt_t1,
            "delta_q_error": "" if gt_delta_q is None else step["delta_q"] - gt_delta_q,
            "abs_delta_q_error": "" if gt_delta_q is None else abs(step["delta_q"] - gt_delta_q),
            "early_stopping": step["early_stopping"],
            "use_best_loss_delta_q": step["use_best_loss_delta_q"],
            "temporal_delta_regularization_enabled": step["temporal_delta_regularization"]["enabled"],
            "temporal_delta_regularization_applied": step["temporal_delta_regularization"]["applied"],
            "lambda_temporal_delta": step["temporal_delta_regularization"]["lambda_temporal_delta"],
            "mlp_smoothness_weight": float(cfg["mlp_smoothness_weight"]) if motion_param == "mlp_q" else "",
            "mlp_acceleration_weight": float(cfg["mlp_acceleration_weight"]) if motion_param == "mlp_q" else "",
            "mlp_monotonic_weight": float(cfg["mlp_monotonic_weight"]) if motion_param == "mlp_q" else "",
            "mlp_time_encoding": str(cfg["mlp_time_encoding"]) if motion_param == "mlp_q" else "",
            "mlp_fourier_frequencies": int(cfg["mlp_fourier_frequencies"]) if motion_param == "mlp_q" else "",
        }
        iteration_logs.append(iteration_log)
        print(
            f"Frame {source_frame:06d} -> {target_frame:06d} | "
            f"iterations_run = {step['iterations_run']} / max_iters = {step['max_iters']} | "
            f"final_loss = {item['final_loss']:.8f} | "
            f"delta_q = {item['delta_q']:+.8f} ({item['commit_source']}) | "
            f"best_iter = {item['best_iteration']} | early_stopping = {item['early_stopping']}"
        )

    iterations_run = [int(row["iterations_run"]) for row in iteration_logs]
    sequence_summary = {
        "num_transitions": len(iteration_logs),
        "total_optimization_iterations": int(sum(iterations_run)) if iterations_run else 0,
        "average_iterations_per_frame": float(sum(iterations_run) / len(iterations_run)) if iterations_run else 0.0,
        "min_iterations_per_frame": int(min(iterations_run)) if iterations_run else 0,
        "max_iterations_per_frame": int(max(iterations_run)) if iterations_run else 0,
        "gaussian_source": gaussian_source,
        "gaussian_ply": str(model_ply),
        "object_id": None if dataset_object is None else dataset_object.object_id,
        "joint_type_original": cfg.get("joint_type_original", joint_type),
        "joint_type_normalized": joint_type,
        "gt_delta_q_available": gt_available,
        "trajectory": trajectory_data.metadata(),
        "use_best_loss_delta_q": bool(cfg.get("use_best_loss_delta_q", False)),
        "early_stopping": cfg.get("early_stopping", {}),
        "temporal_delta_regularization": cfg.get("temporal_delta_regularization", {}),
        "motion_param": motion_param,
        "motion_mlp": mlp_metadata if motion_param == "mlp_q" else None,
    }
    save_json(
        {
            "warning": "No image-space shift is applied in the final pipeline.",
            "manifest": None if manifest_path is None else str(manifest_path),
            "object_id": None if dataset_object is None else dataset_object.object_id,
            "object_name": None if dataset_object is None else dataset_object.object_name,
            "camera": f"cam_{args.cam:03d}",
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "output_subdir": output_subdir,
            "pose_update": "cumulative_from_base_xyz",
            "q_start": q_start,
            "joint_type_original": cfg.get("joint_type_original", joint_type),
            "joint_type_normalized": joint_type,
            "joint_axis": joint_axis.detach().cpu().tolist(),
            "joint_pivot": (
                None if joint_pivot is None else joint_pivot.detach().cpu().tolist()
            ),
            "loss_config": loss_config(cfg),
            "rotation_mode": rotation_mode,
            "gaussian_source": gaussian_source,
            "gaussian_ply": str(model_ply),
            "gt_delta_q_available": gt_available,
            "trajectory_metadata": trajectory_data.metadata(),
            "use_best_loss_delta_q": bool(cfg.get("use_best_loss_delta_q", False)),
            "early_stopping": cfg.get("early_stopping", {}),
            "temporal_delta_regularization": cfg.get("temporal_delta_regularization", {}),
            "motion_param": motion_param,
            "motion_mlp": mlp_metadata if motion_param == "mlp_q" else None,
            "sequence_summary": sequence_summary,
            "trajectory": trajectory,
        },
        out_dir / "trajectory.json",
    )
    save_csv(trajectory, out_dir / "trajectory.csv")
    save_json(
        {"sequence_summary": sequence_summary, "frames": iteration_logs},
        out_dir / "optimization_iterations.json",
    )
    save_csv(iteration_logs, out_dir / "optimization_iterations.csv")
    save_json(
        {
            "description": "Exported articulated Gaussian states. Structural articulation metadata is preserved per vertex; dynamic state is stored in global PLY header comments and state_metadata.json.",
            "source_ply": str(model_ply),
            "export_root": str(deformed_export_root),
            "num_exported_frames": len(deformed_export_entries),
            "dynamic_state_fields": ["q_ref", "last_delta_q", "frame_index"],
            "dynamic_state_storage": "global_ply_header_comments_and_state_metadata_json",
            "no_per_vertex_dynamic_state_properties": True,
            "entries": deformed_export_entries,
        },
        deformed_export_root / "export_summary.json",
    )
    if motion_param == "mlp_q":
        try:
            from scripts.delta_q_tracking.reporting.plot_tracking_diagnostics import (
                plot_sequence_from_trajectory,
            )

            plot_paths = plot_sequence_from_trajectory(out_dir)
            print("sequence_plots=" + ",".join(str(path) for path in plot_paths))
        except Exception as exc:
            print(f"WARNING: failed to generate MLP sequence plots: {exc}")
    print(
        "Sequence summary | "
        f"transitions = {sequence_summary['num_transitions']} | "
        f"total_iterations = {sequence_summary['total_optimization_iterations']} | "
        f"avg_iterations_per_frame = {sequence_summary['average_iterations_per_frame']:.2f} | "
        f"min/max_iterations_per_frame = {sequence_summary['min_iterations_per_frame']} / "
        f"{sequence_summary['max_iterations_per_frame']}"
    )
    print(f"saved_dir={out_dir}")
    print(f"deformed_gaussian_exports={deformed_export_root}")


if __name__ == "__main__":
    main()
