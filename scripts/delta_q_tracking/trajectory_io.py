from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


Q_COORDINATE_MODES = {"relative_to_first_frame", "absolute"}


@dataclass(frozen=True)
class TrajectoryData:
    source_path: Path
    joint_value_column: str
    q_coordinate_mode: str
    frame_indices: tuple[int, ...]
    q_absolute_by_frame: dict[int, float]
    q_relative_to_first_by_frame: dict[int, float]
    q_by_frame: dict[int, float]
    gt_delta_q_by_source_frame: dict[int, float]

    def q(self, frame_index: int) -> float:
        return self.q_by_frame[frame_index]

    def delta(self, source_frame: int) -> float:
        return self.gt_delta_q_by_source_frame[source_frame]

    def metadata(self) -> dict[str, object]:
        return {
            "frame_values_path": str(self.source_path),
            "joint_value_column": self.joint_value_column,
            "q_coordinate_mode": self.q_coordinate_mode,
            "first_frame": self.frame_indices[0],
            "last_frame": self.frame_indices[-1],
            "num_frames": len(self.frame_indices),
        }


def load_trajectory(
    frame_values_path: str | Path,
    joint_value_column: str,
    q_coordinate_mode: str = "relative_to_first_frame",
    requested_start_frame: int | None = None,
    requested_end_frame: int | None = None,
) -> TrajectoryData:
    path = Path(frame_values_path).expanduser().resolve()
    if q_coordinate_mode not in Q_COORDINATE_MODES:
        raise ValueError(
            f"Unsupported q_coordinate_mode={q_coordinate_mode!r}; "
            f"expected one of {sorted(Q_COORDINATE_MODES)}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Missing frame values CSV: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "frame_index" not in fieldnames:
            raise ValueError(f"{path} must contain a frame_index column")
        if joint_value_column not in fieldnames:
            raise ValueError(
                f"{path} does not contain joint column {joint_value_column!r}; "
                f"available columns: {fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no trajectory rows")

    q_absolute_by_frame: dict[int, float] = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            frame_value = row["frame_index"]
            frame_index = int(frame_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{row_number}: invalid frame_index {row.get('frame_index')!r}") from exc
        if str(frame_index) != str(frame_value).strip():
            raise ValueError(f"{path}:{row_number}: frame_index must be an integer, got {frame_value!r}")
        if frame_index in q_absolute_by_frame:
            raise ValueError(f"{path}:{row_number}: duplicate frame_index {frame_index}")
        try:
            q_value = float(row[joint_value_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{row_number}: invalid {joint_value_column} value {row.get(joint_value_column)!r}"
            ) from exc
        if not math.isfinite(q_value):
            raise ValueError(f"{path}:{row_number}: non-finite q value {q_value}")
        q_absolute_by_frame[frame_index] = q_value

    frame_indices = tuple(sorted(q_absolute_by_frame))
    start = frame_indices[0] if requested_start_frame is None else requested_start_frame
    end = frame_indices[-1] if requested_end_frame is None else requested_end_frame
    if start > end:
        raise ValueError(f"requested_start_frame={start} exceeds requested_end_frame={end}")
    missing = [frame for frame in range(start, end + 1) if frame not in q_absolute_by_frame]
    if missing:
        preview = ", ".join(str(frame) for frame in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"{path} is missing requested frame indices: {preview}{suffix}")

    first_q = q_absolute_by_frame[frame_indices[0]]
    q_relative_to_first_by_frame = {
        frame: value - first_q for frame, value in q_absolute_by_frame.items()
    }
    if q_coordinate_mode == "relative_to_first_frame":
        q_by_frame = q_relative_to_first_by_frame
    else:
        q_by_frame = dict(q_absolute_by_frame)

    gt_delta_q_by_source_frame = {
        frame: q_by_frame[frame + 1] - q_by_frame[frame]
        for frame in range(start, end)
    }
    return TrajectoryData(
        source_path=path,
        joint_value_column=joint_value_column,
        q_coordinate_mode=q_coordinate_mode,
        frame_indices=frame_indices,
        q_absolute_by_frame=q_absolute_by_frame,
        q_relative_to_first_by_frame=q_relative_to_first_by_frame,
        q_by_frame=q_by_frame,
        gt_delta_q_by_source_frame=gt_delta_q_by_source_frame,
    )
