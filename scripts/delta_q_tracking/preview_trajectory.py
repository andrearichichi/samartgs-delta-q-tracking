from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.trajectory_io import Q_COORDINATE_MODES, load_trajectory
from scripts.delta_q_tracking.trajectory_profiles import constant_linear, trapezoidal_velocity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or synthesize a joint trajectory without loading CUDA or images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frame-values", type=Path, help="Existing frame_values.csv to validate and preview.")
    source.add_argument("--profile", choices=["constant_linear", "trapezoidal_velocity"])
    parser.add_argument("--joint-column", default=None)
    parser.add_argument("--q-coordinate-mode", choices=sorted(Q_COORDINATE_MODES), default="relative_to_first_frame")
    parser.add_argument("--q-start", type=float, default=0.0)
    parser.add_argument("--q-end", type=float, default=1.0)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--accel-fraction", type=float, default=0.25)
    parser.add_argument("--plateau-fraction", type=float, default=0.50)
    parser.add_argument("--decel-fraction", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_profile(frames: np.ndarray, q: np.ndarray, delta: np.ndarray, velocity: np.ndarray, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150)
        plt.close()

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(frames, q, linewidth=2.2)
    plt.xlabel("frame index")
    plt.ylabel("q")
    plt.title("Ground-truth q trajectory")
    plt.grid(True, alpha=0.3)
    save("gt_q_profile.png")

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(frames[1:], delta[1:], linewidth=2.2)
    plt.xlabel("transition target frame")
    plt.ylabel("q[t] - q[t-1]")
    plt.title("Ground-truth delta_q profile")
    plt.grid(True, alpha=0.3)
    save("gt_delta_q_profile.png")

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(frames[1:], velocity[1:], linewidth=2.2)
    plt.xlabel("transition target frame")
    plt.ylabel("normalized velocity")
    plt.title("Normalized velocity / delta profile")
    plt.grid(True, alpha=0.3)
    save("gt_velocity_or_delta_profile.png")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_values is not None:
        if not args.joint_column:
            raise ValueError("--joint-column is required with --frame-values")
        path = args.frame_values if args.frame_values.is_absolute() else REPO_ROOT / args.frame_values
        trajectory = load_trajectory(path, args.joint_column, args.q_coordinate_mode)
        frames = np.asarray(trajectory.frame_indices, dtype=np.int64)
        q = np.asarray([trajectory.q_by_frame[frame] for frame in frames], dtype=np.float64)
        delta = np.zeros(len(frames), dtype=np.float64)
        delta[1:] = np.diff(q)
        velocity = np.zeros(len(frames), dtype=np.float64)
        max_abs_delta = float(np.max(np.abs(delta[1:]))) if len(delta) > 1 else 0.0
        if max_abs_delta > 0:
            velocity[1:] = delta[1:] / max_abs_delta
        profile_name = "csv"
        metadata = trajectory.metadata()
        joint_column = args.joint_column
    else:
        if args.profile == "constant_linear":
            profile = constant_linear(args.q_start, args.q_end, args.num_frames)
        else:
            profile = trapezoidal_velocity(
                args.q_start,
                args.q_end,
                args.num_frames,
                args.accel_fraction,
                args.plateau_fraction,
                args.decel_fraction,
            )
        frames = profile.frame_index
        q = profile.q
        delta = profile.delta_q_from_prev
        velocity = profile.normalized_velocity
        profile_name = profile.name
        joint_column = args.joint_column or "joint_q"
        metadata = {
            "profile": profile.name,
            "q_start": args.q_start,
            "q_end": args.q_end,
            "num_frames": args.num_frames,
            "accel_fraction": args.accel_fraction,
            "plateau_fraction": args.plateau_fraction,
            "decel_fraction": args.decel_fraction,
            "q_coordinate_mode": "absolute",
        }

    csv_path = output_dir / "frame_values_preview.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", joint_column, "delta_q_from_prev", "normalized_velocity"],
        )
        writer.writeheader()
        for frame, q_value, delta_value, velocity_value in zip(frames, q, delta, velocity):
            writer.writerow(
                {
                    "frame_index": int(frame),
                    joint_column: float(q_value),
                    "delta_q_from_prev": float(delta_value),
                    "normalized_velocity": float(velocity_value),
                }
            )

    deltas = delta[1:]
    metadata.update(
        {
            "profile_name": profile_name,
            "joint_value_column": joint_column,
            "first_q": float(q[0]),
            "last_q": float(q[-1]),
            "delta_q_min": float(np.min(deltas)),
            "delta_q_max": float(np.max(deltas)),
            "delta_q_mean": float(np.mean(deltas)),
            "constant_delta_q": bool(np.allclose(deltas, deltas[0], rtol=0.0, atol=1e-10)),
        }
    )
    (output_dir / "trajectory_profile.json").write_text(json.dumps(metadata, indent=2))
    plot_profile(frames, q, delta, velocity, output_dir)
    print(f"output_dir={output_dir}")
    print(f"frames={len(frames)}")
    print(f"q_start={q[0]:.12g}")
    print(f"q_end={q[-1]:.12g}")
    print(f"delta_q_min={np.min(deltas):.12g}")
    print(f"delta_q_max={np.max(deltas):.12g}")
    print(f"constant_delta_q={metadata['constant_delta_q']}")


if __name__ == "__main__":
    main()
