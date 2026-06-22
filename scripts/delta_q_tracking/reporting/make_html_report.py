from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.trajectory_io import load_trajectory


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, ndigits: int = 6) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{ndigits}f}"


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def rmse(values: list[float]) -> float | None:
    return math.sqrt(mean([value * value for value in values])) if values else None


def stat(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def load_run(sequence_dir: Path) -> dict[str, Any]:
    trajectory_path = sequence_dir / "trajectory.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory output: {trajectory_path}")
    data = json.loads(trajectory_path.read_text())
    rows = data.get("trajectory", [])
    metadata = data.get("trajectory_metadata") or data.get("sequence_summary", {}).get("trajectory")
    fallback_q: dict[int, float] = {}
    if isinstance(metadata, dict):
        try:
            fallback_q = load_trajectory(
                metadata["frame_values_path"],
                metadata["joint_value_column"],
                metadata.get("q_coordinate_mode", "relative_to_first_frame"),
            ).q_by_frame
        except (FileNotFoundError, KeyError, ValueError):
            fallback_q = {}

    q_errors: list[float] = []
    increment_errors: list[float] = []
    required_errors: list[float] = []
    support_values: list[float] = []
    iterations: list[int] = []
    detail_rows: list[dict[str, Any]] = []
    for row in rows:
        source = int(row["source_frame"])
        target = int(row["target_frame"])
        pred_delta = float(row.get("pred_delta_q", row.get("committed_delta_q", row["delta_q"])))
        q_ref_start = float(row.get("q_ref_start", 0.0))
        q_ref_committed = float(row.get("q_ref_committed", row.get("q_ref", q_ref_start + pred_delta)))
        q_gt_t = optional_float(row.get("q_gt_t"))
        q_gt_t1 = optional_float(row.get("q_gt_t1"))
        if q_gt_t is None:
            q_gt_t = fallback_q.get(source)
        if q_gt_t1 is None:
            q_gt_t1 = fallback_q.get(target)
        gt_delta = optional_float(row.get("gt_delta_q"))
        if gt_delta is None and q_gt_t is not None and q_gt_t1 is not None:
            gt_delta = q_gt_t1 - q_gt_t
        required_delta = optional_float(row.get("required_delta_to_GT"))
        if required_delta is None and q_gt_t1 is not None:
            required_delta = q_gt_t1 - q_ref_start
        increment_error = None if gt_delta is None else pred_delta - gt_delta
        required_error = None if required_delta is None else pred_delta - required_delta
        q_error = None if q_gt_t1 is None else q_ref_committed - q_gt_t1
        if increment_error is not None:
            increment_errors.append(increment_error)
        if required_error is not None:
            required_errors.append(required_error)
        if q_error is not None:
            q_errors.append(q_error)
        support = float(row.get("support_iou", row.get("raw_iou", 0.0)))
        support_values.append(support)
        iterations.append(int(row.get("iterations_run", 0)))
        detail_rows.append(
            {
                "transition": f"{source:06d}->{target:06d}",
                "pred_delta": pred_delta,
                "gt_delta": gt_delta,
                "required_delta": required_delta,
                "increment_error": increment_error,
                "required_error": required_error,
                "q_error": q_error,
                "support_iou": support,
                "iterations": iterations[-1],
            }
        )

    summary = data.get("sequence_summary", {})
    return {
        "data": data,
        "rows": rows,
        "metadata": metadata or {},
        "completed_transitions": len(rows),
        "total_iterations": summary.get("total_optimization_iterations", sum(iterations)),
        "q_ref_mae": mean([abs(value) for value in q_errors]) if q_errors else None,
        "q_ref_rmse": rmse(q_errors),
        "final_q_ref_error": q_errors[-1] if q_errors else None,
        "increment_mae": mean([abs(value) for value in increment_errors]) if increment_errors else None,
        "increment_rmse": rmse(increment_errors),
        "required_mae": mean([abs(value) for value in required_errors]) if required_errors else None,
        "required_rmse": rmse(required_errors),
        "support_iou": stat(support_values),
        "worst_increment": sorted(
            [row for row in detail_rows if row["increment_error"] is not None],
            key=lambda row: abs(row["increment_error"]),
            reverse=True,
        )[:10],
        "worst_required": sorted(
            [row for row in detail_rows if row["required_error"] is not None],
            key=lambda row: abs(row["required_error"]),
            reverse=True,
        )[:10],
    }


def gallery(sequence_dir: Path, report_dir: Path, items: list[tuple[str, str]]) -> str:
    figures = []
    for label, relative_path in items:
        path = sequence_dir / relative_path
        if path.exists():
            src = esc(os.path.relpath(path, report_dir))
            figures.append(f"<figure><img src='{src}' alt='{esc(label)}'><figcaption>{esc(label)}</figcaption></figure>")
    return "<div class='gallery'>" + "\n".join(figures) + "</div>" if figures else "<p>No plots found.</p>"


def worst_table(rows: list[dict[str, Any]], error_key: str) -> str:
    return table(
        ["transition", "pred delta", "GT increment", "required delta", "abs error", "q_ref error", "support IoU", "iters"],
        [
            [
                row["transition"],
                fmt(row["pred_delta"]),
                fmt(row["gt_delta"]),
                fmt(row["required_delta"]),
                fmt(abs(row[error_key])),
                fmt(row["q_error"]),
                fmt(row["support_iou"]),
                row["iterations"],
            ]
            for row in rows
        ],
    )


def write_report(sequence_dir: Path, report_dir: Path) -> tuple[Path, dict[str, Any]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "index.html"
    stats = load_run(sequence_dir)
    data = stats["data"]
    support = stats["support_iou"]
    metric_table = table(
        ["Metric", "Value"],
        [
            ["transitions", stats["completed_transitions"]],
            ["total optimization iterations", stats["total_iterations"]],
            ["q_ref MAE", fmt(stats["q_ref_mae"])],
            ["q_ref RMSE", fmt(stats["q_ref_rmse"])],
            ["delta_q MAE vs GT increment", fmt(stats["increment_mae"])],
            ["delta_q RMSE vs GT increment", fmt(stats["increment_rmse"])],
            ["delta_q MAE vs required delta", fmt(stats["required_mae"])],
            ["delta_q RMSE vs required delta", fmt(stats["required_rmse"])],
            ["final q_ref error", fmt(stats["final_q_ref_error"])],
            ["support IoU mean", fmt(support["mean"])],
            ["support IoU std", fmt(support["std"])],
            ["support IoU min", fmt(support["min"])],
        ],
    )
    main_plots = gallery(
        sequence_dir,
        report_dir,
        [
            ("GT q profile", "sequence_plots/gt_q_profile.png"),
            ("GT delta_q profile", "sequence_plots/gt_delta_q_profile.png"),
            ("q_ref vs GT", "sequence_plots/q_ref_vs_gt_by_frame.png"),
            ("q_ref error", "sequence_plots/q_ref_error_by_frame.png"),
            ("predicted vs GT vs required delta", "sequence_plots/delta_q_pred_vs_gt_vs_required.png"),
            ("delta error vs GT increment", "sequence_plots/final_delta_q_error_by_frame.png"),
            ("delta error vs required delta", "sequence_plots/delta_error_vs_required.png"),
            ("absolute delta error vs required", "sequence_plots/abs_delta_error_vs_required.png"),
            ("support IoU", "sequence_plots/support_iou_by_frame.png"),
            ("iterations per frame", "sequence_plots/iterations_per_frame.png"),
        ],
    )
    diagnostics = gallery(
        sequence_dir,
        report_dir,
        [
            ("Optimization objective", "sequence_plots/internal_optimization/optimization_objective_by_frame.png"),
            ("Image objective components", "sequence_plots/internal_optimization/image_loss_components_by_frame.png"),
            ("Objective components", "sequence_plots/internal_optimization/optimization_objective_components_by_frame.png"),
        ],
    )
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delta-q tracking report</title>
<style>
body{{margin:0;background:#f7f8fb;color:#172033;font:15px/1.55 system-ui,sans-serif}}header{{background:#111827;color:white;padding:32px}}
main{{max-width:1180px;margin:24px auto;padding:0 20px}}section{{background:white;border:1px solid #d8e0ea;border-radius:8px;padding:20px;margin:18px 0}}
table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d8e0ea;padding:8px;text-align:left}}th{{background:#f1f5f9}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}figure{{margin:0;border:1px solid #d8e0ea}}img{{width:100%;display:block}}figcaption{{padding:8px}}
code{{background:#e8edf4;padding:2px 5px;border-radius:4px}}.callout{{background:#eff6ff;border-left:5px solid #2563eb;padding:12px}}
</style></head><body>
<header><h1>Delta-q tracking report</h1><p>{esc(sequence_dir)}</p></header><main>
<section><h2>Tracking formulation</h2>
<p>For every transition, <code>GT increment delta_q = q_gt[t+1] - q_gt[t]</code>.</p>
<p><code>required_delta_to_GT = q_gt[t+1] - q_ref[t]</code>. These differ after accumulated tracking error because the optimizer starts from predicted <code>q_ref(t)</code>, not from <code>q_gt(t)</code>.</p>
<p>The tracker still optimizes only one scalar <code>delta_q</code> and updates <code>q_ref(t+1) = q_ref(t) + delta_q</code>. Non-constant velocity is supported whenever the configured <code>frame_values.csv</code> contains the correct per-frame q values.</p>
<div class="callout">Loss is the internal image-fitting objective, not the final tracking metric. Use q_ref error, both delta-error definitions, and support IoU to assess tracking.</div>
</section>
<section><h2>Configuration</h2>{table(["Field", "Value"], [
    ["trajectory", stats["metadata"]],
    ["gaussian source", data.get("gaussian_source")],
    ["rotation mode", data.get("rotation_mode")],
    ["temporal regularization", data.get("temporal_delta_regularization")],
])}</section>
<section><h2>Summary metrics</h2>{metric_table}</section>
<section><h2>Main tracking plots</h2>{main_plots}</section>
<section><h2>Worst transitions vs GT increment</h2>{worst_table(stats["worst_increment"], "increment_error")}</section>
<section><h2>Worst transitions vs required delta</h2>{worst_table(stats["worst_required"], "required_error")}</section>
<section><h2>Internal optimization diagnostics</h2>{diagnostics}</section>
</main></body></html>"""
    report_path.write_text(html_text)
    return report_path, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report for a delta-q tracking run.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path("outputs/delta_q_tracking/usb/final_rigid_usb_gauss_new/cam_000"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/delta_q_tracking/usb/report"))
    args = parser.parse_args()
    sequence_dir = args.sequence_dir if args.sequence_dir.is_absolute() else REPO_ROOT / args.sequence_dir
    report_dir = args.report_dir if args.report_dir.is_absolute() else REPO_ROOT / args.report_dir
    path, stats = write_report(sequence_dir, report_dir)
    print(f"report={path}")
    print(f"sequence_dir={sequence_dir}")
    print(f"transitions={stats['completed_transitions']}")
    print(f"delta_q_mae_vs_gt_increment={fmt(stats['increment_mae'], 8)}")
    print(f"delta_q_mae_vs_required={fmt(stats['required_mae'], 8)}")


if __name__ == "__main__":
    main()
