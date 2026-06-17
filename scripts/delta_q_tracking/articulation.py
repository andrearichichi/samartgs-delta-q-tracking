from __future__ import annotations

import torch


def normalize_axis(axis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return axis / axis.norm(dim=-1, keepdim=True).clamp_min(eps)


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
