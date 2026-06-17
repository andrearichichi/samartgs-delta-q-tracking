from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "outputs" / "delta_q_tracking" / "usb" / "report"
REPORT_PATH = REPORT_DIR / "index.html"
GT_PATH = REPO_ROOT / "../dataset/usb_rgbdm/metadata/frame_values.csv"
FINAL_SEQUENCE_DIR = REPO_ROOT / "outputs" / "delta_q_tracking" / "usb" / "final_rigid_usb_gauss_new" / "cam_000"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, ndigits: int = 6) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if math.isnan(v):
        return "n/a"
    return f"{v:.{ndigits}f}"


def rel(path: Path) -> str:
    return esc(os.path.relpath(path, REPORT_DIR))


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def badge(text: str, kind: str = "done") -> str:
    return f'<span class="badge {esc(kind)}">{esc(text)}</span>'


def load_gt_relative() -> tuple[dict[int, float], str | None]:
    rows = read_csv(GT_PATH)
    if not rows:
        return {}, None
    value_key = next((key for key in rows[0] if key != "frame_index"), None)
    if value_key is None:
        return {}, None
    q_abs = {int(row["frame_index"]): float(row[value_key]) for row in rows}
    if 0 not in q_abs:
        return {}, value_key
    q0 = q_abs[0]
    return {frame: value - q0 for frame, value in q_abs.items()}, value_key


def rmse(values: list[float]) -> float | None:
    return math.sqrt(mean([v * v for v in values])) if values else None


def stat(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def metric_rows(stats: dict[str, Any]) -> list[list[Any]]:
    support = stats["support_iou"]
    return [
        ["transitions", stats["completed_transitions"]],
        ["total iterations", stats["total_iterations"]],
        ["average iterations/frame", fmt(stats["avg_iterations"], 2)],
        ["min/max iterations/frame", f"{stats['min_iterations']} / {stats['max_iterations']}"],
        ["q_ref MAE", fmt(stats["q_ref_mae"])],
        ["q_ref RMSE", fmt(stats["q_ref_rmse"])],
        ["delta_q MAE", fmt(stats["delta_q_mae"])],
        ["delta_q RMSE", fmt(stats["delta_q_rmse"])],
        ["final q_ref error", fmt(stats["final_q_ref_error"])],
        ["support IoU mean/min/max", f"{fmt(support['mean'])} / {fmt(support['min'])} / {fmt(support['max'])}"],
        ["finite gradients", f"{stats['grad_finite_count']} / {stats['completed_transitions']}"],
        ["non-zero gradients", f"{stats['grad_nonzero_count']} / {stats['completed_transitions']}"],
    ]


def load_final_run() -> dict[str, Any]:
    data = read_json(FINAL_SEQUENCE_DIR / "trajectory.json") or {}
    traj = data.get("trajectory", [])
    gt_rel, gt_column = load_gt_relative()
    q_errors: list[float] = []
    dq_errors: list[float] = []
    support_iou: list[float] = []
    final_loss: list[float] = []
    iterations: list[int] = []
    worst_rows: list[dict[str, Any]] = []

    for item in traj:
        source = int(item["source_frame"])
        target = int(item["target_frame"])
        pred_q = float(item["q_ref"])
        pred_dq = float(item["delta_q"])
        gt_q = gt_rel.get(target)
        gt_dq = None if source not in gt_rel or target not in gt_rel else gt_rel[target] - gt_rel[source]
        q_err = None if gt_q is None else pred_q - gt_q
        dq_err = None if gt_dq is None else pred_dq - gt_dq
        if q_err is not None:
            q_errors.append(q_err)
        if dq_err is not None:
            dq_errors.append(dq_err)
        iou = float(item.get("support_iou", item.get("raw_iou", 0.0)))
        loss = float(item["final_loss"])
        support_iou.append(iou)
        final_loss.append(loss)
        iterations.append(int(item["iterations_run"]))
        worst_rows.append(
            {
                "transition": f"{source:06d}->{target:06d}",
                "pred_delta_q": pred_dq,
                "gt_delta_q": gt_dq,
                "abs_delta_q_error": None if dq_err is None else abs(dq_err),
                "q_ref_error": q_err,
                "loss": loss,
                "support_iou": iou,
                "iterations": int(item["iterations_run"]),
            }
        )

    summary = data.get("sequence_summary", {})
    return {
        "found": bool(traj),
        "data": data,
        "traj": traj,
        "gt_column": gt_column,
        "completed_transitions": len(traj),
        "total_iterations": summary.get("total_optimization_iterations", sum(iterations)),
        "avg_iterations": summary.get("average_iterations_per_frame", mean(iterations) if iterations else None),
        "min_iterations": summary.get("min_iterations_per_frame", min(iterations) if iterations else None),
        "max_iterations": summary.get("max_iterations_per_frame", max(iterations) if iterations else None),
        "q_ref_mae": mean([abs(x) for x in q_errors]) if q_errors else None,
        "q_ref_rmse": rmse(q_errors),
        "delta_q_mae": mean([abs(x) for x in dq_errors]) if dq_errors else None,
        "delta_q_rmse": rmse(dq_errors),
        "final_q_ref_error": q_errors[-1] if q_errors else None,
        "support_iou": stat(support_iou),
        "final_loss": stat(final_loss),
        "grad_finite_count": sum(1 for item in traj if bool(item.get("grad_finite"))),
        "grad_nonzero_count": sum(1 for item in traj if bool(item.get("grad_nonzero"))),
        "worst": sorted(
            [row for row in worst_rows if row["abs_delta_q_error"] is not None],
            key=lambda row: row["abs_delta_q_error"],
            reverse=True,
        )[:10],
    }


def gallery(items: list[tuple[str, Path]]) -> str:
    found = [(label, path) for label, path in items if path.exists()]
    if not found:
        return "<p class='muted'>No images found.</p>"
    figs = []
    for label, path in found:
        figs.append(
            f"<figure><img src='{rel(path)}' alt='{esc(label)}' loading='lazy'><figcaption>{esc(label)}</figcaption></figure>"
        )
    return "<div class='gallery'>" + "\n".join(figs) + "</div>"


def plot_gallery() -> str:
    plot_dir = FINAL_SEQUENCE_DIR / "sequence_plots"
    return gallery(
        [
            ("q_ref vs GT", plot_dir / "q_ref_vs_gt_by_frame.png"),
            ("q_ref error by frame", plot_dir / "q_ref_error_by_frame.png"),
            ("absolute q_ref error by frame", plot_dir / "abs_q_ref_error_by_frame.png"),
            ("delta_q vs GT increment", plot_dir / "final_delta_q_by_frame.png"),
            ("delta_q vs required delta to GT", plot_dir / "delta_q_vs_required_delta_by_frame.png"),
            ("delta_q error vs GT increment", plot_dir / "final_delta_q_error_by_frame.png"),
            ("delta_q error vs required delta", plot_dir / "delta_q_error_vs_required_delta_by_frame.png"),
            ("absolute delta_q error by frame", plot_dir / "abs_final_delta_q_error_by_frame.png"),
            ("support IoU by frame", plot_dir / "support_iou_by_frame.png"),
            ("iterations per frame", plot_dir / "iterations_per_frame.png"),
            ("best iteration by frame", plot_dir / "best_iteration_by_frame.png"),
        ]
    )


def internal_diagnostics_gallery() -> str:
    plot_dir = FINAL_SEQUENCE_DIR / "sequence_plots" / "internal_optimization"
    plots = [
        ("Optimization objective by frame", plot_dir / "optimization_objective_by_frame.png"),
        ("Image objective components by frame", plot_dir / "image_loss_components_by_frame.png"),
        ("Objective components by frame", plot_dir / "optimization_objective_components_by_frame.png"),
    ]
    per_frame = FINAL_SEQUENCE_DIR / "plots_per_frame"
    for transition in ["000000_to_000001", "000002_to_000003", "000058_to_000059"]:
        plots.extend(
            [
                (f"{transition} loss vs iteration", per_frame / transition / "loss_vs_iteration.png"),
                (f"{transition} delta_q vs iteration", per_frame / transition / "delta_q_vs_iteration.png"),
                (f"{transition} delta_q error vs iteration", per_frame / transition / "delta_q_error_vs_iteration.png"),
            ]
        )
    return gallery(plots)


def overlay_gallery() -> str:
    return gallery(
        [
            ("frame 001 raw overlay", FINAL_SEQUENCE_DIR / "overlay_raw_frame_000001.png"),
            ("frame 030 raw overlay", FINAL_SEQUENCE_DIR / "overlay_raw_frame_000030.png"),
            ("frame 059 raw overlay", FINAL_SEQUENCE_DIR / "overlay_raw_frame_000059.png"),
            ("frame 001 predicted render", FINAL_SEQUENCE_DIR / "pred_raw_frame_000001.png"),
            ("frame 030 predicted render", FINAL_SEQUENCE_DIR / "pred_raw_frame_000030.png"),
            ("frame 059 predicted render", FINAL_SEQUENCE_DIR / "pred_raw_frame_000059.png"),
        ]
    )


def worst_table(stats: dict[str, Any]) -> str:
    rows = [
        [
            row["transition"],
            fmt(row["pred_delta_q"]),
            fmt(row["gt_delta_q"]),
            fmt(row["abs_delta_q_error"]),
            fmt(row["q_ref_error"]),
            fmt(row["loss"]),
            fmt(row["support_iou"]),
            row["iterations"],
        ]
        for row in stats["worst"]
    ]
    return table(["transition", "pred delta_q", "GT delta_q", "abs dq err", "q_ref err", "loss", "support IoU", "iters"], rows)


def write_report() -> tuple[Path, dict[str, Any]]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stats = load_final_run()
    data = stats["data"]
    gaussian_ply = data.get(
        "gaussian_ply",
        "../dataset/usb_gauss_new/point_cloud/iteration_30000/scene_mask_filtered_renderer.ply",
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>USB final rigid delta-q tracking report</title>
  <style>
    :root {{ --bg:#f7f8fb; --ink:#172033; --muted:#64748b; --card:#fff; --line:#d8e0ea; --green:#15803d; --blue:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; }}
    header {{ background:#111827; color:white; padding:36px 28px; }}
    header h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }}
    nav {{ position:sticky; top:0; background:white; border-bottom:1px solid var(--line); padding:10px 28px; z-index:2; }}
    nav a {{ margin-right:16px; color:#1f3b73; text-decoration:none; font-weight:700; }}
    main {{ max-width:1180px; margin:24px auto 56px; padding:0 20px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:20px; margin:20px 0; box-shadow:0 8px 24px rgba(16,24,40,.05); }}
    .grid {{ display:grid; gap:16px; }}
    .two {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .three {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .badge {{ display:inline-block; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:800; margin-right:5px; background:#dcfce7; color:var(--green); }}
    code,pre {{ background:#0f172a; color:#e5e7eb; border-radius:6px; }}
    code {{ padding:2px 5px; }}
    pre {{ padding:12px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; margin:10px 0 16px; background:white; }}
    th,td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    th {{ background:#f1f5f9; }}
    .muted {{ color:var(--muted); }}
    .callout {{ border-left:5px solid var(--blue); background:#eff6ff; padding:13px 15px; border-radius:8px; }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
    figure {{ margin:0; background:white; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    figure img {{ display:block; width:100%; background:#e5e7eb; }}
    figcaption {{ padding:9px 11px; color:var(--muted); font-weight:650; }}
    @media (max-width:780px) {{ .two,.three {{ grid-template-columns:1fr; }} nav {{ position:static; }} }}
  </style>
</head>
<body>
<header>
  <h1>USB final rigid delta-q tracking</h1>
  <p>No-shift tracking with the corrected <code>usb_gauss_new</code> Gaussian source.</p>
</header>
<nav>
  <a href="#method">Method</a><a href="#config">Config</a><a href="#results">Metrics</a><a href="#plots">Main Plots</a><a href="#worst">Worst Frames</a><a href="#overlays">Overlays</a><a href="#diagnostics">Diagnostics</a><a href="#artifacts">Commands</a>
</nav>
<main>
  <section id="method" class="card">
    <h2>Method Summary</h2>
    <p>{badge('RIGID')} {badge('NO IMAGE SHIFT')} {badge('USB_GAUSS_NEW')} {badge('DELTA_Q ONLY')}</p>
    <div class="callout"><strong>Main result.</strong> This report shows only the final no-shift rigid delta-q tracking run at <code>{esc(FINAL_SEQUENCE_DIR.relative_to(REPO_ROOT))}</code>. Alignment diagnostics and old image-shift experiments are intentionally excluded from the main result.</div>
    <h2>Tracking Formulation</h2>
    <p>For each transition, the optimizer starts from the current reference pose at frame <code>t</code>, uses the RGB/mask/camera target at frame <code>t+1</code>, and optimizes only a scalar <code>delta_q</code>. Moving Gaussians are selected by <code>joint_part == 1</code>; static and ignored Gaussians remain fixed. The prediction renders frame <code>t+1</code>, then updates <code>q_ref(t+1) = q_ref(t) + delta_q</code>.</p>
    <pre>x_pred = origin + R(axis, q_ref + delta_q - q_start) @ (base_xyz - origin)
q_pred = q_delta ⊗ q_gaussian   # quaternion order: wxyz</pre>
    <p>Gaussian parameters, camera, joint axis/pivot/type, and part ids are frozen.</p>
  </section>

  <section id="config" class="card">
    <h2>Final Configuration</h2>
    {table(['Field', 'Value'], [
        ['model_path', '../dataset/usb_gauss_new'],
        ['gaussian_source', data.get('gaussian_source', 'scene_mask_filtered_renderer')],
        ['gaussian_ply', gaussian_ply],
        ['rotation_mode', data.get('rotation_mode', 'rigid')],
        ['use_best_loss_delta_q', data.get('use_best_loss_delta_q', True)],
        ['early_stopping', data.get('early_stopping', {})],
        ['temporal_delta_regularization', data.get('temporal_delta_regularization', {})],
        ['loss', data.get('loss_config', {})],
        ['GT column', stats.get('gt_column')],
    ])}
  </section>

  <section id="results" class="card">
    <h2>Main Tracking Metrics</h2>
    {table(['Metric', 'Value'], metric_rows(stats))}
  </section>

  <section id="plots" class="card">
    <h2>Main Tracking Plots</h2>
    <p>These plots are the primary quality checks: cumulative <code>q_ref</code> error, frame-to-frame <code>delta_q</code> error, support IoU, and optimizer iteration counts under early stopping.</p>
    {plot_gallery()}
  </section>

  <section id="worst" class="card">
    <h2>Worst Frames / Outliers</h2>
    {worst_table(stats)}
  </section>

  <section id="overlays" class="card">
    <h2>Representative Overlays</h2>
    {overlay_gallery()}
  </section>

  <section id="diagnostics" class="card">
    <h2>Internal Optimization Diagnostics</h2>
    <div class="callout">This loss is the internal objective minimized to estimate <code>delta_q</code>. It is not a direct joint-motion error metric. Tracking quality should be evaluated mainly with <code>q_ref</code> error, <code>delta_q</code> error, and support IoU.</div>
    {internal_diagnostics_gallery()}
  </section>

  <section id="artifacts" class="card">
    <h2>Reproduce Commands</h2>
    <p>The final run folder contains <code>trajectory.json/csv</code>, <code>optimization_iterations.json/csv</code>, per-frame images, per-iteration CSV logs, exported articulated Gaussian states, and sequence plots.</p>
    <pre>python scripts/delta_q_tracking/run_sequence.py \
  --config scripts/delta_q_tracking/config_usb.yaml \
  --cam 0 \
  --start-frame 0 \
  --end-frame 59 \
  --output-subdir final_rigid_usb_gauss_new

python scripts/delta_q_tracking/reporting/plot_tracking_diagnostics.py \
  --sequence-dir outputs/delta_q_tracking/usb/final_rigid_usb_gauss_new/cam_000

python scripts/delta_q_tracking/reporting/make_html_report.py</pre>
  </section>
</main>
</body>
</html>
"""
    REPORT_PATH.write_text(html_text)
    return REPORT_PATH, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final USB delta-q tracking HTML report.")
    parser.parse_args()
    path, stats = write_report()
    print(f"report={path}")
    print(f"sequence_dir={FINAL_SEQUENCE_DIR}")
    print(f"trajectory_found={stats['found']}")
    print(f"transitions={stats['completed_transitions']}")
    print(f"total_iterations={stats['total_iterations']}")
    print(f"q_ref_mae={fmt(stats['q_ref_mae'], 8)}")
    print(f"q_ref_rmse={fmt(stats['q_ref_rmse'], 8)}")
    print(f"delta_q_mae={fmt(stats['delta_q_mae'], 8)}")
    print(f"delta_q_rmse={fmt(stats['delta_q_rmse'], 8)}")
    print(f"final_q_ref_error={fmt(stats['final_q_ref_error'], 8)}")


if __name__ == "__main__":
    main()
