from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.trajectory_io import load_trajectory


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def has_gt(rows: list[dict[str, str]]) -> bool:
    return any(row.get("gt_delta_q") not in {None, ""} for row in rows)


def plot_frame(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in rows]
    total_loss = [float(row["total_loss"]) for row in rows]
    pred_delta_q = [float(row["pred_delta_q"]) for row in rows]
    gt_value = as_float(rows[-1].get("gt_delta_q")) if rows else None
    paths: list[Path] = []

    def finish(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(7.5, 4.0))
    plt.plot(iterations, total_loss, linewidth=1.8)
    plt.xlabel("optimization iteration")
    plt.ylabel("total loss")
    plt.title("Loss vs optimization iteration")
    plt.grid(True, alpha=0.3)
    finish("loss_vs_iteration.png")

    plt.figure(figsize=(7.5, 4.0))
    plt.plot(iterations, pred_delta_q, linewidth=1.8, label="pred delta_q")
    if gt_value is not None:
        plt.axhline(gt_value, color="#b91c1c", linestyle="--", linewidth=1.4, label="GT delta_q")
        plt.legend()
    plt.xlabel("optimization iteration")
    plt.ylabel("delta_q")
    plt.title("Predicted delta_q vs optimization iteration")
    plt.grid(True, alpha=0.3)
    finish("delta_q_vs_iteration.png")

    if gt_value is not None:
        errors = [float(row["delta_q_error"]) for row in rows]
        plt.figure(figsize=(7.5, 4.0))
        plt.plot(iterations, errors, linewidth=1.8)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("optimization iteration")
        plt.ylabel("delta_q error")
        plt.title("delta_q error vs optimization iteration")
        plt.grid(True, alpha=0.3)
        finish("delta_q_error_vs_iteration.png")

    return paths


def plot_sequence(final_rows: list[dict[str, object]], out_dir: Path, gt_available: bool) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [int(row["frame_to"]) for row in final_rows]
    paths: list[Path] = []

    def finish(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(8.0, 4.2))
    plt.plot(frames, [float(row["total_loss"]) for row in final_rows], linewidth=2)
    plt.xlabel("target frame")
    plt.ylabel("final loss")
    plt.title("Final loss by frame transition")
    plt.grid(True, alpha=0.3)
    finish("final_loss_by_frame.png")

    plt.figure(figsize=(8.0, 4.2))
    plt.plot(frames, [float(row["pred_delta_q"]) for row in final_rows], linewidth=2, label="pred delta_q")
    if gt_available:
        plt.plot(frames, [float(row["gt_delta_q"]) for row in final_rows], linewidth=2, label="GT delta_q")
        plt.legend()
    plt.xlabel("target frame")
    plt.ylabel("delta_q")
    plt.title("Final predicted delta_q by frame")
    plt.grid(True, alpha=0.3)
    finish("final_delta_q_by_frame.png")

    if gt_available:
        plt.figure(figsize=(8.0, 4.2))
        plt.plot(frames, [float(row["delta_q_error"]) for row in final_rows], linewidth=2)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("target frame")
        plt.ylabel("final delta_q error")
        plt.title("Final delta_q error by frame")
        plt.grid(True, alpha=0.3)
        finish("final_delta_q_error_by_frame.png")

        plt.figure(figsize=(8.0, 4.2))
        plt.plot(frames, [float(row["abs_delta_q_error"]) for row in final_rows], linewidth=2)
        plt.xlabel("target frame")
        plt.ylabel("absolute final delta_q error")
        plt.title("Absolute final delta_q error by frame")
        plt.grid(True, alpha=0.3)
        finish("abs_final_delta_q_error_by_frame.png")

    return paths


def load_gt_fallback(sequence_dir: Path) -> dict[int, float]:
    payload_path = sequence_dir / "trajectory.json"
    if not payload_path.exists():
        return {}
    payload = json.loads(payload_path.read_text())
    metadata = payload.get("trajectory_metadata") or payload.get("sequence_summary", {}).get("trajectory")
    if not isinstance(metadata, dict):
        return {}
    try:
        trajectory = load_trajectory(
            metadata["frame_values_path"],
            metadata["joint_value_column"],
            metadata.get("q_coordinate_mode", "relative_to_first_frame"),
        )
    except (FileNotFoundError, KeyError, ValueError):
        return {}
    return trajectory.q_by_frame


def plot_sequence_from_trajectory(sequence_dir: Path) -> list[Path]:
    trajectory_path = sequence_dir / "trajectory.csv"
    if not trajectory_path.exists():
        return []
    rows = read_rows(trajectory_path)
    if not rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = sequence_dir / "sequence_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_fallback = load_gt_fallback(sequence_dir)
    frames = [int(row["target_frame"]) for row in rows]
    transitions = [int(row["source_frame"]) for row in rows]
    paths: list[Path] = []

    def finish(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    q_pred = [float(row.get("q_ref_committed") or row["q_ref"]) for row in rows]
    dq_pred = [float(row.get("pred_delta_q") or row.get("committed_delta_q") or row["delta_q"]) for row in rows]
    gt_q = [
        as_float(row.get("q_gt_t1")) if as_float(row.get("q_gt_t1")) is not None else gt_fallback.get(frame)
        for row, frame in zip(rows, frames)
    ]
    gt_dq = [
        as_float(row.get("gt_delta_q"))
        if as_float(row.get("gt_delta_q")) is not None
        else None
        if int(row["source_frame"]) not in gt_fallback or int(row["target_frame"]) not in gt_fallback
        else gt_fallback[int(row["target_frame"])] - gt_fallback[int(row["source_frame"])]
        for row in rows
    ]
    q_ref_starts: list[float] = []
    previous_q_ref = 0.0
    for row in rows:
        if row.get("q_ref_start") not in {None, ""}:
            q_ref_starts.append(float(row["q_ref_start"]))
        else:
            q_ref_starts.append(previous_q_ref)
        previous_q_ref = float(row.get("q_ref_committed") or row["q_ref"])
    required_delta = [
        float(row["required_delta_to_GT"]) if row.get("required_delta_to_GT") not in {None, ""}
        else None if gt_q_value is None
        else gt_q_value - q_ref_start
        for row, q_ref_start, gt_q_value in zip(rows, q_ref_starts, gt_q)
    ]
    has_gt_q = all(value is not None for value in gt_q)
    has_gt_dq = all(value is not None for value in gt_dq)
    has_required_delta = all(value is not None for value in required_delta)

    if has_gt_q:
        source_frame = int(rows[0]["source_frame"])
        source_gt = as_float(rows[0].get("q_gt_t"))
        if source_gt is None:
            source_gt = gt_fallback.get(source_frame)
        profile_frames = ([source_frame] if source_gt is not None else []) + frames
        profile_q = ([source_gt] if source_gt is not None else []) + gt_q

        plt.figure(figsize=(8.4, 4.4))
        plt.plot(profile_frames, profile_q, linewidth=2.2)
        plt.xlabel("frame index")
        plt.ylabel("q_gt")
        plt.title("Ground-truth q trajectory")
        plt.grid(True, alpha=0.3)
        finish("gt_q_profile.png")

        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, q_pred, linewidth=2.2, label="predicted q_ref")
        plt.plot(frames, gt_q, linewidth=2.2, label="ground truth q_ref")
        plt.xlabel("frame index")
        plt.ylabel("cumulative joint state q")
        plt.title("q_ref predicted vs ground truth")
        plt.grid(True, alpha=0.3)
        plt.legend()
        finish("q_ref_vs_gt_by_frame.png")

        q_errors = [pred - gt for pred, gt in zip(q_pred, gt_q)]
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, q_errors, linewidth=2.0)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("frame index")
        plt.ylabel("predicted q_ref - GT q_ref")
        plt.title("q_ref error by frame")
        plt.grid(True, alpha=0.3)
        finish("q_ref_error_by_frame.png")

        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, [abs(value) for value in q_errors], linewidth=2.0)
        plt.xlabel("frame index")
        plt.ylabel("absolute q_ref error")
        plt.title("Absolute q_ref error by frame")
        plt.grid(True, alpha=0.3)
        finish("abs_q_ref_error_by_frame.png")

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(transitions, dq_pred, linewidth=2.2, label="predicted delta_q")
    if has_gt_dq:
        plt.plot(transitions, gt_dq, linewidth=2.2, label="ground truth delta_q")
        plt.legend()
    plt.xlabel("transition start frame")
    plt.ylabel("frame-to-frame delta_q")
    plt.title("delta_q predicted vs ground truth")
    plt.grid(True, alpha=0.3)
    finish("final_delta_q_by_frame.png")

    if has_gt_dq:
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(transitions, gt_dq, linewidth=2.2)
        plt.xlabel("transition start frame")
        plt.ylabel("q_gt[t+1] - q_gt[t]")
        plt.title("Ground-truth delta_q profile")
        plt.grid(True, alpha=0.3)
        finish("gt_delta_q_profile.png")

    if has_required_delta and has_gt_dq:
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(transitions, dq_pred, linewidth=2.2, label="predicted delta_q")
        plt.plot(transitions, gt_dq, linewidth=2.2, label="GT increment")
        plt.plot(transitions, required_delta, linewidth=2.2, label="required delta to GT")
        plt.xlabel("transition start frame")
        plt.ylabel("delta_q")
        plt.title("Predicted delta_q vs GT increment vs required delta")
        plt.grid(True, alpha=0.3)
        plt.legend()
        finish("delta_q_pred_vs_gt_vs_required.png")

    if has_required_delta:
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(transitions, dq_pred, linewidth=2.2, label="predicted delta_q")
        plt.plot(transitions, required_delta, linewidth=2.2, label="required delta to reach GT")
        plt.xlabel("transition start frame")
        plt.ylabel("delta_q")
        plt.title("delta_q predicted vs required delta to reach GT")
        plt.grid(True, alpha=0.3)
        plt.legend()
        finish("delta_q_vs_required_delta_by_frame.png")

    if has_gt_dq:
        errors = [pred - gt for pred, gt in zip(dq_pred, gt_dq)]
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, errors, linewidth=2)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("transition target frame")
        plt.ylabel("predicted delta_q - GT increment")
        plt.title("delta_q error vs GT increment")
        plt.grid(True, alpha=0.3)
        finish("final_delta_q_error_by_frame.png")

        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, [abs(v) for v in errors], linewidth=2)
        plt.xlabel("transition target frame")
        plt.ylabel("absolute delta_q error vs GT increment")
        plt.title("Absolute delta_q error by frame")
        plt.grid(True, alpha=0.3)
        finish("abs_final_delta_q_error_by_frame.png")

    if has_required_delta:
        required_errors = [pred - req for pred, req in zip(dq_pred, required_delta)]
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, required_errors, linewidth=2)
        plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        plt.xlabel("transition target frame")
        plt.ylabel("predicted delta_q - required delta")
        plt.title("delta_q error vs required delta")
        plt.grid(True, alpha=0.3)
        finish("delta_q_error_vs_required_delta_by_frame.png")
        alias_path = out_dir / "delta_error_vs_required.png"
        shutil.copy2(out_dir / "delta_q_error_vs_required_delta_by_frame.png", alias_path)
        paths.append(alias_path)

        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, [abs(value) for value in required_errors], linewidth=2)
        plt.xlabel("transition target frame")
        plt.ylabel("absolute predicted delta_q - required delta")
        plt.title("Absolute delta_q error vs required delta")
        plt.grid(True, alpha=0.3)
        finish("abs_delta_error_vs_required.png")

    iou_key = "support_iou" if "support_iou" in rows[0] else "raw_iou"
    plt.figure(figsize=(8.4, 4.4))
    plt.plot(frames, [float(row[iou_key]) for row in rows], linewidth=2)
    plt.xlabel("frame index")
    plt.ylabel("support IoU")
    plt.title("Render-mask support IoU by frame")
    plt.grid(True, alpha=0.3)
    finish("support_iou_by_frame.png")

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(frames, [int(row["iterations_run"]) for row in rows], linewidth=2)
    plt.xlabel("transition target frame")
    plt.ylabel("optimization iterations actually run")
    plt.title("Iterations per frame")
    plt.grid(True, alpha=0.3)
    finish("iterations_per_frame.png")

    if "best_iteration" in rows[0]:
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, [int(row["best_iteration"]) for row in rows], linewidth=2)
        plt.xlabel("transition target frame")
        plt.ylabel("best-loss iteration")
        plt.title("Best iteration by frame")
        plt.grid(True, alpha=0.3)
        finish("best_iteration_by_frame.png")

    # Internal optimization diagnostics: objective/loss components are useful for debugging
    # optimizer behavior, but they are not direct joint-motion quality metrics.
    diagnostic_dir = sequence_dir / "sequence_plots" / "internal_optimization"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    def finish_diag(name: str) -> None:
        path = diagnostic_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(8.4, 4.4))
    plt.plot(frames, [float(row["final_loss"]) for row in rows], linewidth=2)
    plt.xlabel("transition target frame")
    plt.ylabel("total optimization objective")
    plt.title("Optimization objective by frame")
    plt.grid(True, alpha=0.3)
    finish_diag("optimization_objective_by_frame.png")

    if "best_rgb_loss" in rows[0] and "best_ssim" in rows[0]:
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, [float(row["best_rgb_loss"]) for row in rows], linewidth=2, label="L1/RGB loss")
        plt.plot(frames, [1.0 - float(row["best_ssim"]) for row in rows], linewidth=2, label="1 - SSIM")
        plt.xlabel("transition target frame")
        plt.ylabel("loss component value")
        plt.title("Image objective components by frame")
        plt.grid(True, alpha=0.3)
        plt.legend()
        finish_diag("image_loss_components_by_frame.png")

    # Temporal regularization is logged per iteration, so use the committed/best iteration row when available.
    temporal_values: list[float] = []
    image_values: list[float] = []
    l1_values: list[float] = []
    ssim_loss_values: list[float] = []
    logs_dir = sequence_dir / "per_iteration_logs"
    for row in rows:
        log_path = logs_dir / f"{int(row['source_frame']):06d}_to_{int(row['target_frame']):06d}_iterations.csv"
        log_rows = read_rows(log_path) if log_path.exists() else []
        best_iter = int(row.get("best_iteration", row.get("iterations_run", 1)))
        selected = log_rows[max(0, min(best_iter - 1, len(log_rows) - 1))] if log_rows else {}
        temporal_values.append(float(selected.get("temporal_delta_loss") or 0.0))
        image_values.append(float(selected.get("image_loss") or row["final_loss"]))
        l1_values.append(float(selected.get("rgb_loss") or 0.0))
        ssim_loss_values.append(float(selected.get("ssim_loss") or 0.0))
    if any(value != 0.0 for value in l1_values) or any(value != 0.0 for value in ssim_loss_values):
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, l1_values, linewidth=2, label="masked RGB/L1 loss")
        plt.plot(frames, ssim_loss_values, linewidth=2, label="1 - SSIM")
        plt.xlabel("transition target frame")
        plt.ylabel("image loss component")
        plt.title("Image objective components by frame")
        plt.grid(True, alpha=0.3)
        plt.legend()
        finish_diag("image_loss_components_by_frame.png")
    if any(value != 0.0 for value in temporal_values):
        plt.figure(figsize=(8.4, 4.4))
        plt.plot(frames, image_values, linewidth=2, label="image objective")
        plt.plot(frames, temporal_values, linewidth=2, label="temporal_delta_loss")
        plt.plot(frames, [float(row["final_loss"]) for row in rows], linewidth=2, label="total objective")
        plt.xlabel("transition target frame")
        plt.ylabel("objective value")
        plt.title("Optimization objective components by frame")
        plt.grid(True, alpha=0.3)
        plt.legend()
        finish_diag("optimization_objective_components_by_frame.png")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-iteration delta-q tracking diagnostics.")
    parser.add_argument(
        "--sequence-dir",
        default="outputs/delta_q_tracking/usb/03_sequence_rigid_600iters/cam_000",
        help="Sequence output directory containing per_iteration_logs/.",
    )
    args = parser.parse_args()

    sequence_dir = Path(args.sequence_dir)
    if not sequence_dir.is_absolute():
        sequence_dir = REPO_ROOT / sequence_dir
    if not sequence_dir.exists() and "final_rigid_usb_gauss_new" in str(sequence_dir):
        fallback = REPO_ROOT / "outputs" / "delta_q_tracking" / "usb" / "03_sequence_rigid_final" / "cam_000"
        if fallback.exists():
            sequence_dir = fallback
    logs_dir = sequence_dir / "per_iteration_logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Missing per-iteration log directory: {logs_dir}")

    csv_paths = sorted(logs_dir.glob("*_iterations.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per-iteration CSV files found in {logs_dir}")

    final_rows: list[dict[str, object]] = []
    frame_plot_count = 0
    gt_available = False
    for csv_path in csv_paths:
        rows = read_rows(csv_path)
        if not rows:
            continue
        transition = csv_path.name.removesuffix("_iterations.csv")
        frame_plot_dir = sequence_dir / "plots_per_frame" / transition
        frame_plot_count += len(plot_frame(rows, frame_plot_dir))
        frame_has_gt = has_gt(rows)
        gt_available = gt_available or frame_has_gt
        final = dict(rows[-1])
        final_rows.append(final)

    sequence_paths = plot_sequence_from_trajectory(sequence_dir)
    if not sequence_paths:
        sequence_paths = plot_sequence(final_rows, sequence_dir / "sequence_plots", gt_available)
    print(f"sequence_dir={sequence_dir}")
    print(f"per_iteration_logs={logs_dir}")
    print(f"transitions={len(final_rows)}")
    print(f"gt_delta_q_available={gt_available}")
    print(f"per_frame_plot_images={frame_plot_count}")
    print("sequence_plots=" + ",".join(str(path) for path in sequence_paths))


if __name__ == "__main__":
    main()
