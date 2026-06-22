from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryProfile:
    name: str
    frame_index: np.ndarray
    q: np.ndarray
    delta_q_from_prev: np.ndarray
    normalized_velocity: np.ndarray

    def rows(self, joint_column: str = "q") -> list[dict[str, float | int]]:
        return [
            {
                "frame_index": int(frame),
                joint_column: float(q),
                "delta_q_from_prev": float(delta),
                "normalized_velocity": float(velocity),
            }
            for frame, q, delta, velocity in zip(
                self.frame_index,
                self.q,
                self.delta_q_from_prev,
                self.normalized_velocity,
            )
        ]


def _validate_num_frames(num_frames: int) -> None:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")


def constant_linear(q_start: float, q_end: float, num_frames: int) -> TrajectoryProfile:
    _validate_num_frames(num_frames)
    q = np.linspace(float(q_start), float(q_end), num_frames, dtype=np.float64)
    delta = np.zeros(num_frames, dtype=np.float64)
    delta[1:] = np.diff(q)
    velocity = np.zeros(num_frames, dtype=np.float64)
    if q_end != q_start:
        velocity[1:] = delta[1:] / np.max(np.abs(delta[1:]))
    return TrajectoryProfile(
        name="constant_linear",
        frame_index=np.arange(num_frames, dtype=np.int64),
        q=q,
        delta_q_from_prev=delta,
        normalized_velocity=velocity,
    )


def trapezoidal_velocity(
    q_start: float,
    q_end: float,
    num_frames: int,
    accel_fraction: float,
    plateau_fraction: float,
    decel_fraction: float,
) -> TrajectoryProfile:
    _validate_num_frames(num_frames)
    fractions = np.asarray([accel_fraction, plateau_fraction, decel_fraction], dtype=np.float64)
    if np.any(fractions < 0):
        raise ValueError("accel, plateau, and decel fractions must be non-negative")
    if not np.isclose(float(fractions.sum()), 1.0, atol=1e-9):
        raise ValueError("accel_fraction + plateau_fraction + decel_fraction must equal 1")
    if accel_fraction == 0 and decel_fraction == 0:
        return constant_linear(q_start, q_end, num_frames)

    transition_count = num_frames - 1
    phase_position = (np.arange(transition_count, dtype=np.float64) + 0.5) / transition_count
    weights = np.empty(transition_count, dtype=np.float64)
    accel_end = accel_fraction
    plateau_end = accel_fraction + plateau_fraction
    for idx, position in enumerate(phase_position):
        if position < accel_end and accel_fraction > 0:
            weights[idx] = position / accel_fraction
        elif position < plateau_end:
            weights[idx] = 1.0
        elif decel_fraction > 0:
            weights[idx] = max(0.0, (1.0 - position) / decel_fraction)
        else:
            weights[idx] = 1.0
    if not np.any(weights > 0):
        raise ValueError("trajectory profile produced zero velocity at every transition")

    displacement = float(q_end) - float(q_start)
    transition_delta = displacement * weights / weights.sum()
    q = np.empty(num_frames, dtype=np.float64)
    q[0] = float(q_start)
    q[1:] = float(q_start) + np.cumsum(transition_delta)
    q[-1] = float(q_end)
    delta = np.zeros(num_frames, dtype=np.float64)
    delta[1:] = np.diff(q)
    velocity = np.zeros(num_frames, dtype=np.float64)
    if np.max(weights) > 0:
        velocity[1:] = np.sign(displacement) * weights / np.max(weights)
    return TrajectoryProfile(
        name="trapezoidal_velocity",
        frame_index=np.arange(num_frames, dtype=np.int64),
        q=q,
        delta_q_from_prev=delta,
        normalized_velocity=velocity,
    )
