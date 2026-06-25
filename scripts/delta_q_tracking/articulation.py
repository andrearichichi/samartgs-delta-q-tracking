from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def normalize_axis(axis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = axis.norm(dim=-1, keepdim=True)
    if bool((norm < eps).any().item()):
        raise ValueError("Joint axis must be non-zero")
    return axis / norm


SUPPORTED_JOINT_TYPES = {"prismatic", "revolute", "continuous"}
REVOLUTE_JOINT_TYPES = {"revolute", "continuous"}


@dataclass(frozen=True)
class JointMetadata:
    joint_name: str
    joint_type: str
    axis_world: torch.Tensor
    pivot_world: torch.Tensor | None
    limits: tuple[float, float] | None
    moving_part_ids: tuple[int, ...]
    q_units: str

    @property
    def branch(self) -> str:
        return "prismatic" if self.joint_type == "prismatic" else "revolute"

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "joint_type": self.joint_type,
            "branch": self.branch,
            "axis_world": self.axis_world.detach().cpu().tolist(),
            "pivot_world": None if self.pivot_world is None else self.pivot_world.detach().cpu().tolist(),
            "limits": None if self.limits is None else list(self.limits),
            "moving_part_ids": list(self.moving_part_ids),
            "q_units": self.q_units,
        }


@dataclass(frozen=True)
class ArticulationTransform:
    points: torch.Tensor
    rotations: torch.Tensor | None
    branch: str


def canonical_joint_type(joint_type: str) -> str:
    value = str(joint_type).strip().lower()
    if value not in SUPPORTED_JOINT_TYPES:
        raise NotImplementedError(
            f"Unsupported joint type {joint_type!r}; supported types are "
            f"{sorted(SUPPORTED_JOINT_TYPES)}. Screw joints remain unsupported."
        )
    return value


def build_joint_metadata(
    raw_metadata: dict[str, Any],
    *,
    joint_type_override: str = "auto",
    expected_joint_type: str | None = None,
    joint_name: str | None = None,
    moving_part_ids: tuple[int, ...] = (1,),
) -> JointMetadata:
    embedded = raw_metadata.get("joint_metadata") or {}
    id_map = raw_metadata.get("joint_type_id_map") or {}
    metadata_type = embedded.get("type")
    if metadata_type is None:
        joint_type_id = int(raw_metadata["joint_type_id"])
        reverse_map = {int(value): str(key).lower() for key, value in id_map.items()}
        metadata_type = reverse_map.get(joint_type_id)
    if metadata_type is None:
        raise ValueError("Joint type is missing from both PLY joint_metadata and joint_type_id_map")

    detected_type = canonical_joint_type(str(metadata_type))
    override = str(joint_type_override).strip().lower()
    selected_type = detected_type if override == "auto" else canonical_joint_type(override)
    if override != "auto" and selected_type != detected_type:
        raise ValueError(
            f"Configured articulation.joint_type={selected_type!r} does not match "
            f"loaded metadata joint type {detected_type!r}"
        )
    if expected_joint_type not in {None, ""}:
        expected = canonical_joint_type(str(expected_joint_type))
        if expected != detected_type:
            raise ValueError(
                f"Configured articulation.expected_joint_type={expected!r} does not match "
                f"loaded metadata joint type {detected_type!r}"
            )

    detected_name = str(embedded.get("name", ""))
    if joint_name not in {None, ""} and detected_name and str(joint_name) != detected_name:
        raise ValueError(
            f"Configured articulation.joint_name={joint_name!r} does not match "
            f"loaded metadata joint name {detected_name!r}"
        )
    axis = normalize_axis(raw_metadata["joint_axis"].reshape(3))
    pivot = raw_metadata.get("joint_origin")
    if selected_type in REVOLUTE_JOINT_TYPES and pivot is None:
        raise ValueError(f"Joint type {selected_type!r} requires pivot_world")
    if pivot is not None:
        pivot = pivot.reshape(3)

    limits_value = embedded.get("limits")
    limits = None
    if selected_type != "continuous" and isinstance(limits_value, (list, tuple)) and len(limits_value) == 2:
        limits = (float(limits_value[0]), float(limits_value[1]))
    return JointMetadata(
        joint_name=str(joint_name or detected_name or "unnamed_joint"),
        joint_type=selected_type,
        axis_world=axis,
        pivot_world=pivot,
        limits=limits,
        moving_part_ids=tuple(int(value) for value in moving_part_ids),
        q_units="dataset_units" if selected_type == "prismatic" else "radians",
    )


def rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Return a 3x3 rotation matrix for an axis-angle rotation."""
    axis = normalize_axis(axis.reshape(3))
    angle = angle.reshape(())
    x, y, z = axis[0], axis[1], axis[2]
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Return a wxyz quaternion for an axis-angle rotation."""
    axis = normalize_axis(axis.reshape(3))
    angle = angle.reshape(())
    half = 0.5 * angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return normalize_quaternion(torch.cat([w.reshape(1), xyz], dim=0))


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions, broadcast over leading dims."""
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def apply_prismatic_transform(
    points: torch.Tensor,
    rotations: torch.Tensor | None,
    moving_mask: torch.Tensor,
    axis_world: torch.Tensor,
    delta_q: torch.Tensor,
) -> ArticulationTransform:
    """Translate moving Gaussians by delta_q along the normalized joint axis."""
    axis = normalize_axis(axis_world.to(device=points.device, dtype=points.dtype).reshape(3))
    displacement = delta_q.to(device=points.device, dtype=points.dtype).reshape(()) * axis
    mask = moving_mask.to(device=points.device, dtype=torch.bool).reshape(-1)
    moved = points + displacement.reshape(1, 3)
    deformed_points = torch.where(mask[:, None], moved, points)
    return ArticulationTransform(deformed_points, rotations, "prismatic")


def apply_revolute_transform(
    points: torch.Tensor,
    rotations: torch.Tensor | None,
    moving_mask: torch.Tensor,
    axis_world: torch.Tensor,
    pivot_world: torch.Tensor,
    delta_q: torch.Tensor,
) -> ArticulationTransform:
    """Rotate moving Gaussians around the world-space axis through the pivot."""
    pivot = pivot_world.to(device=points.device, dtype=points.dtype).reshape(1, 3)
    axis = axis_world.to(device=points.device, dtype=points.dtype).reshape(3)
    angle = delta_q.to(device=points.device, dtype=points.dtype).reshape(())
    mask = moving_mask.to(device=points.device, dtype=torch.bool).reshape(-1)
    rotation_matrix = rodrigues(axis, angle)
    moved = pivot + (points - pivot) @ rotation_matrix.T
    deformed_points = torch.where(mask[:, None], moved, points)
    deformed_rotations = rotations
    if rotations is not None:
        deformed_rotations = deform_rotation_relative_continuous(rotations, mask, axis, angle)
    return ArticulationTransform(deformed_points, deformed_rotations, "revolute")


def apply_articulation_transform(
    joint_type: str,
    points: torch.Tensor,
    rotations: torch.Tensor | None,
    moving_mask: torch.Tensor,
    axis_world: torch.Tensor,
    pivot_world: torch.Tensor | None,
    delta_q: torch.Tensor,
) -> ArticulationTransform:
    """Dispatch Gaussian deformation explicitly by normalized joint type."""
    normalized_type = canonical_joint_type(joint_type)
    if normalized_type == "prismatic":
        return apply_prismatic_transform(points, rotations, moving_mask, axis_world, delta_q)
    if normalized_type in REVOLUTE_JOINT_TYPES:
        if pivot_world is None:
            raise ValueError(f"Joint type {normalized_type!r} requires pivot_world")
        return apply_revolute_transform(
            points,
            rotations,
            moving_mask,
            axis_world,
            pivot_world,
            delta_q,
        )
    raise AssertionError(f"Unhandled supported joint type {normalized_type!r}")


def deform_rotation_relative_continuous(
    base_rot: torch.Tensor,
    moving_mask: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Apply q_delta * q_gaussian to moving Gaussian wxyz rotations only."""
    base_rot = normalize_quaternion(base_rot)
    axis = axis.to(device=base_rot.device, dtype=base_rot.dtype)
    angle = angle.to(device=base_rot.device, dtype=base_rot.dtype)
    moving_mask = moving_mask.to(device=base_rot.device, dtype=torch.bool).reshape(-1)
    q_delta = axis_angle_to_quaternion(axis, angle).reshape(1, 4)
    moved = normalize_quaternion(quaternion_multiply(q_delta.expand_as(base_rot), base_rot))
    return torch.where(moving_mask[:, None], moved, base_rot)


def deform_xyz_relative_continuous(
    xyz: torch.Tensor,
    moving_mask: torch.Tensor,
    origin: torch.Tensor,
    axis: torch.Tensor,
    delta_q: torch.Tensor,
) -> torch.Tensor:
    """Apply a relative continuous/revolute delta to moving points only."""
    origin = origin.to(device=xyz.device, dtype=xyz.dtype).reshape(1, 3)
    axis = axis.to(device=xyz.device, dtype=xyz.dtype)
    delta_q = delta_q.to(device=xyz.device, dtype=xyz.dtype)
    moving_mask = moving_mask.to(device=xyz.device, dtype=torch.bool).reshape(-1)
    r_delta = rodrigues(axis, delta_q)
    moved = origin + (xyz - origin) @ r_delta.T
    return torch.where(moving_mask[:, None], moved, xyz)
