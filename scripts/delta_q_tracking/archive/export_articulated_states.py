from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.io_utils import load_simple_yaml, resolve_gaussian_ply, save_json


def normalize(v: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


def rodrigues_np(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis.reshape(3))
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def normalize_quat(q: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)


def axis_angle_quat_wxyz(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis.reshape(3))
    half = 0.5 * angle
    return normalize_quat(np.array([math.cos(half), *(axis * math.sin(half))], dtype=np.float64))


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def state_comments(comments: list[str], frame_index: int, q_ref: float, last_delta_q: float) -> list[str]:
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
    state = {"frame": int(frame_index), "q_ref": float(q_ref), "last_delta_q": float(last_delta_q)}
    preserved.extend(
        [
            f"articulated_state {json.dumps(state, separators=(',', ':'))}",
            f"q_ref {float(q_ref):.12g}",
            f"last_delta_q {float(last_delta_q):.12g}",
            f"frame_index {int(frame_index)}",
        ]
    )
    return preserved


def export_state(
    source_ply: Path,
    base_ply: PlyData,
    export_root: Path,
    frame_index: int,
    q_ref: float,
    last_delta_q: float,
    q_start: float,
    moving_part_id: int,
    rotation_mode: str,
) -> dict[str, Any]:
    vertex = base_ply["vertex"].data.copy()
    fields = list(vertex.dtype.names or [])
    names = set(fields)
    forbidden = [name for name in ("q_ref", "last_delta_q", "frame_index") if name in names]
    if forbidden:
        raise ValueError(f"Dynamic state fields must not be per-vertex properties, found: {forbidden}")

    required = {
        "x",
        "y",
        "z",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "joint_part",
        "joint_origin_x",
        "joint_origin_y",
        "joint_origin_z",
        "joint_axis_x",
        "joint_axis_y",
        "joint_axis_z",
    }
    missing = sorted(required - names)
    if missing:
        raise KeyError(f"Source PLY is missing required fields: {missing}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    base_rot = np.stack([vertex[f"rot_{idx}"] for idx in range(4)], axis=1).astype(np.float64)
    moving_mask = np.asarray(vertex["joint_part"]) == moving_part_id
    origin = np.array([vertex["joint_origin_x"][0], vertex["joint_origin_y"][0], vertex["joint_origin_z"][0]], dtype=np.float64)
    axis = np.array([vertex["joint_axis_x"][0], vertex["joint_axis_y"][0], vertex["joint_axis_z"][0]], dtype=np.float64)
    angle = float(q_ref) - float(q_start)
    r_delta = rodrigues_np(axis, angle)
    moved_xyz = origin.reshape(1, 3) + (xyz - origin.reshape(1, 3)) @ r_delta.T
    xyz_out = np.where(moving_mask[:, None], moved_xyz, xyz)
    for idx, key in enumerate(("x", "y", "z")):
        vertex[key] = xyz_out[:, idx].astype(vertex[key].dtype, copy=False)

    if rotation_mode == "rigid":
        q_delta = axis_angle_quat_wxyz(axis, angle).reshape(1, 4)
        moved_rot = normalize_quat(quat_multiply_wxyz(np.repeat(q_delta, len(base_rot), axis=0), base_rot))
        rot_out = np.where(moving_mask[:, None], moved_rot, base_rot)
        for idx in range(4):
            key = f"rot_{idx}"
            vertex[key] = rot_out[:, idx].astype(vertex[key].dtype, copy=False)

    frame_dir = export_root / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    out_ply = frame_dir / "point_cloud.ply"
    comments = state_comments([str(comment) for comment in base_ply.comments], frame_index, q_ref, last_delta_q)
    PlyData(
        [PlyElement.describe(vertex, "vertex")],
        text=base_ply.text,
        byte_order=base_ply.byte_order,
        comments=comments,
        obj_info=base_ply.obj_info,
    ).write(str(out_ply))

    structural_fields = [
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
    state = {
        "frame_index": int(frame_index),
        "q_ref": float(q_ref),
        "last_delta_q": float(last_delta_q),
        "source_ply": str(source_ply),
        "exported_ply": str(out_ply),
        "state_metadata_json": str(frame_dir / "state_metadata.json"),
        "articulated_state": {"frame": int(frame_index), "q_ref": float(q_ref), "last_delta_q": float(last_delta_q)},
        "dynamic_state_storage": "global_ply_header_comments_and_state_metadata_json",
        "no_per_vertex_dynamic_state_properties": True,
        "rotation_mode": rotation_mode,
        "original_vertex_properties_preserved": fields,
        "structural_articulation_properties_present": {name: name in names for name in structural_fields},
        "preserved_original_comments": [str(comment) for comment in base_ply.comments],
        "comments_added": {
            "articulated_state": True,
            "q_ref": True,
            "last_delta_q": True,
            "frame_index": True,
        },
    }
    save_json(state, frame_dir / "state_metadata.json")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Export self-contained articulated Gaussian PLY states from a trajectory.")
    parser.add_argument("--config", default="scripts/delta_q_tracking/config_usb.yaml")
    parser.add_argument("--sequence-dir", default="outputs/delta_q_tracking/usb/03_sequence_rigid_final/cam_000")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    sequence_dir = Path(args.sequence_dir)
    if not sequence_dir.is_absolute():
        sequence_dir = REPO_ROOT / sequence_dir
    trajectory_path = sequence_dir / "trajectory.json"
    trajectory_data = json.loads(trajectory_path.read_text())
    trajectory = trajectory_data.get("trajectory", [])
    gaussian_source = str(cfg.get("gaussian_source", trajectory_data.get("gaussian_source", "point_cloud")))
    source_ply = resolve_gaussian_ply(cfg["model_path"], int(cfg["iteration"]), gaussian_source)
    base_ply = PlyData.read(str(source_ply))
    export_root = sequence_dir / "deformed_gaussians"
    export_root.mkdir(parents=True, exist_ok=True)
    q_start = float(trajectory_data.get("q_start", cfg.get("q_start", 0.0)))
    rotation_mode = str(trajectory_data.get("rotation_mode", cfg.get("rotation_mode", "rigid")))
    moving_part_id = int(cfg["moving_part_id"])

    entries: list[dict[str, Any]] = [
        export_state(source_ply, base_ply, export_root, int(trajectory_data.get("start_frame", 0)), q_start, 0.0, q_start, moving_part_id, rotation_mode)
    ]
    for item in trajectory:
        entries.append(
            export_state(
                source_ply,
                base_ply,
                export_root,
                int(item["target_frame"]),
                float(item["q_ref"]),
                float(item["delta_q"]),
                q_start,
                moving_part_id,
                rotation_mode,
            )
        )

    summary = {
        "description": "Exported articulated Gaussian states. Structural articulation metadata is preserved per vertex; dynamic state is stored in global PLY header comments and state_metadata.json.",
        "source_ply": str(source_ply),
        "export_root": str(export_root),
        "trajectory_json": str(trajectory_path),
        "num_exported_frames": len(entries),
        "dynamic_state_fields": ["q_ref", "last_delta_q", "frame_index"],
        "dynamic_state_storage": "global_ply_header_comments_and_state_metadata_json",
        "no_per_vertex_dynamic_state_properties": True,
        "entries": entries,
    }
    save_json(summary, export_root / "export_summary.json")
    print(f"export_root={export_root}")
    print(f"export_summary={export_root / 'export_summary.json'}")
    print(f"num_exported_frames={len(entries)}")
    if entries:
        print(f"first_ply={entries[0]['exported_ply']}")
        print(f"last_ply={entries[-1]['exported_ply']}")


if __name__ == "__main__":
    main()
