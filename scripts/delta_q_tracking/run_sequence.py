from __future__ import annotations

import argparse
import csv
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
    support_metrics,
    tensor_to_pil_rgb,
)
from scripts.delta_q_tracking.losses import loss_config, masked_rgb_loss


def load_gt_relative_q_by_frame() -> dict[int, float]:
    path = REPO_ROOT / "../dataset/usb_rgbdm/metadata/frame_values.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    value_key = next((key for key in rows[0].keys() if key != "frame_index"), None)
    if value_key is None:
        return {}
    q_abs = {int(row["frame_index"]): float(row[value_key]) for row in rows}
    if 0 not in q_abs:
        return {}
    q0 = q_abs[0]
    return {frame: value - q0 for frame, value in q_abs.items()}


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


def optimize_step(
    cfg: dict,
    gaussians,
    base_xyz: torch.Tensor,
    moving_mask: torch.Tensor,
    origin: torch.Tensor,
    axis: torch.Tensor,
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
        xyz = deform_xyz_relative_continuous(base_xyz, moving_mask, origin, axis, q_pred)
        rotation = None
        if rotation_mode == "rigid":
            rotation = deform_rotation_relative_continuous(gaussians.get_rotation, moving_mask, axis, q_pred)
        raw_render = render(camera, DeformedGaussian(gaussians, xyz, rotation), default_pipeline(), bg)["render"]
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
        final_xyz = deform_xyz_relative_continuous(base_xyz, moving_mask, origin, axis, final_angle)
        final_rotation = None
        if rotation_mode == "rigid":
            final_rotation = deform_rotation_relative_continuous(gaussians.get_rotation, moving_mask, axis, final_angle)
        final_raw = render(camera, DeformedGaussian(gaussians, final_xyz, final_rotation), default_pipeline(), bg)["render"]

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
        "final_xyz": final_xyz.detach(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final rigid delta-q tracking sequence.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
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
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
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
    model_ply = resolve_gaussian_ply(cfg["model_path"], int(cfg["iteration"]), gaussian_source)

    print(f"gaussian_source={gaussian_source}")
    print(f"gaussian_ply={model_ply}")
    gaussians = load_gaussian_model(cfg["model_path"], int(cfg["iteration"]), gaussian_source=gaussian_source)
    meta = load_enriched_ply_metadata(model_ply)
    if int(meta["joint_type_id"]) != 3:
        print(f"WARNING: expected continuous joint_type_id=3, got {meta['joint_type_id']}")
    id_map = meta.get("joint_type_id_map")
    if id_map is not None:
        continuous_id = str(id_map.get("continuous", id_map.get("CONTINUOUS", "")))
        if continuous_id != "3":
            print(f"WARNING: joint_type_id_map does not confirm continuous=3: {id_map}")
        else:
            print("joint_type_id_map confirms continuous=3")
    else:
        print("WARNING: PLY comment joint_type_id_map is missing; assuming joint_type_id=3 means continuous.")
    moving_mask = meta["part_ids"] == int(cfg["moving_part_id"])
    base_xyz = gaussians.get_xyz.detach().clone()
    q_start = float(cfg.get("q_start", 0.0))
    q_ref = q_start
    trajectory = []
    iteration_logs = []
    gt_q_by_frame = load_gt_relative_q_by_frame()
    gt_available = bool(gt_q_by_frame)
    print(f"gt_delta_q_available={gt_available}")
    print(f"rotation_mode={rotation_mode}")
    print(f"num_iters={int(cfg['num_iters'])}")
    print(f"use_best_loss_delta_q={bool(cfg.get('use_best_loss_delta_q', False))}")
    print(f"early_stopping={cfg.get('early_stopping', {})}")
    print(f"temporal_delta_regularization={cfg.get('temporal_delta_regularization', {})}")
    previous_committed_delta_q: float | None = None
    deformed_export_root = ensure_output_dir(out_dir / "deformed_gaussians")
    deformed_export_entries: list[dict[str, object]] = []

    with torch.no_grad():
        initial_camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, args.start_frame)
        initial_rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, args.start_frame)
        initial_angle = torch.as_tensor(q_ref - q_start, dtype=base_xyz.dtype, device=base_xyz.device)
        initial_xyz = deform_xyz_relative_continuous(
            base_xyz,
            moving_mask,
            meta["joint_origin"],
            meta["joint_axis"],
            initial_angle,
        )
        initial_rotation = None
        if rotation_mode == "rigid":
            initial_rotation = deform_rotation_relative_continuous(
                gaussians.get_rotation,
                moving_mask,
                meta["joint_axis"],
                initial_angle,
            )
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        initial_raw = render(
            initial_camera,
            DeformedGaussian(gaussians, initial_xyz, initial_rotation),
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
                initial_xyz,
                initial_rotation,
            )
        )

    for source_frame in range(args.start_frame, args.end_frame):
        target_frame = source_frame + 1
        print(f"frame {source_frame:06d}->{target_frame:06d}: optimizing delta_q")
        camera = build_colmap_camera(source, cfg["rgb_root"], args.cam, target_frame)
        target_rgb_pil = load_rgb_frame(cfg["rgb_root"], args.cam, target_frame)
        target_mask_pil = load_mask_frame(cfg["mask_root"], args.cam, target_frame)
        target_rgb = image_to_cuda_rgb(target_rgb_pil)
        target_mask = mask_to_cuda(target_mask_pil)
        q_ref_start = q_ref
        q_gt_t = gt_q_by_frame.get(source_frame)
        q_gt_t1 = gt_q_by_frame.get(target_frame)
        gt_delta_q = None if q_gt_t is None or q_gt_t1 is None else q_gt_t1 - q_gt_t
        required_delta_to_gt = None if q_gt_t1 is None else q_gt_t1 - q_ref_start

        step = optimize_step(
            cfg,
            gaussians,
            base_xyz,
            moving_mask,
            meta["joint_origin"],
            meta["joint_axis"],
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
                None if rotation_mode != "rigid" else deform_rotation_relative_continuous(
                    gaussians.get_rotation,
                    moving_mask,
                    meta["joint_axis"],
                    torch.as_tensor(q_ref - q_start, dtype=base_xyz.dtype, device=base_xyz.device),
                ),
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
            "source_frame": source_frame,
            "target_frame": target_frame,
            "delta_q": step["delta_q"],
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
            "q_ref_committed": q_ref,
            "q_gt_t": "" if q_gt_t is None else q_gt_t,
            "q_gt_t1": "" if q_gt_t1 is None else q_gt_t1,
            "required_delta_to_GT": "" if required_delta_to_gt is None else required_delta_to_gt,
            "delta_error_vs_gt_increment": step["committed_delta_q_error"],
            "delta_error_vs_required": step["committed_delta_error_vs_required"],
            "q_ref_error_after_commit": "" if q_gt_t1 is None else q_ref - q_gt_t1,
            "q_start": q_start,
            "angle_from_base": q_ref - q_start,
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
            "rotation_mode": rotation_mode,
            "gaussian_source": gaussian_source,
            "gt_delta_q": "" if gt_delta_q is None else gt_delta_q,
            "support_iou": support["support_mask_iou"],
            "raw_iou": support["support_mask_iou"],
        }
        trajectory.append(item)
        iteration_log = {
            "frame_index": target_frame,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "transition": f"{source_frame:06d}->{target_frame:06d}",
            "max_iters": step["max_iters"],
            "iterations_run": step["iterations_run"],
            "final_loss": step["final_loss"],
            "delta_q": step["delta_q"],
            "committed_delta_q": step["committed_delta_q"],
            "commit_source": step["commit_source"],
            "final_iteration_delta_q": step["final_iteration_delta_q"],
            "final_iteration_loss": step["final_iteration_loss"],
            "best_loss_delta_q": step["best_loss_delta_q"],
            "best_loss": step["best_loss"],
            "best_iteration": step["best_iteration"],
            "q_ref_start": q_ref_start,
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
        "gt_delta_q_available": gt_available,
        "use_best_loss_delta_q": bool(cfg.get("use_best_loss_delta_q", False)),
        "early_stopping": cfg.get("early_stopping", {}),
        "temporal_delta_regularization": cfg.get("temporal_delta_regularization", {}),
    }
    save_json(
        {
            "warning": "No image-space shift is applied in the final pipeline.",
            "camera": f"cam_{args.cam:03d}",
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "output_subdir": output_subdir,
            "pose_update": "cumulative_from_base_xyz",
            "q_start": q_start,
            "loss_config": loss_config(cfg),
            "rotation_mode": rotation_mode,
            "gaussian_source": gaussian_source,
            "gaussian_ply": str(model_ply),
            "gt_delta_q_available": gt_available,
            "use_best_loss_delta_q": bool(cfg.get("use_best_loss_delta_q", False)),
            "early_stopping": cfg.get("early_stopping", {}),
            "temporal_delta_regularization": cfg.get("temporal_delta_regularization", {}),
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
