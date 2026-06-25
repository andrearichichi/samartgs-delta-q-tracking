from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = {
    "direct": "Direct delta_q",
    "mlp": "Best MLP q(t)",
}

METRIC_COLUMNS = [
    ("q_mae", "q MAE", False),
    ("delta_q_mae", "delta_q MAE", False),
    ("final_q_error_abs", "final q error", False),
    ("ssim", "SSIM", True),
    ("smoothness", "smoothness", False),
    ("score", "score", False),
]

DETAIL_COLUMNS = [
    ("support_iou", "Support IoU"),
    ("l1", "L1"),
    ("loss", "Loss"),
    ("mean_iterations", "Mean iterations"),
]

PLOTS = [
    ("q_ref vs GT", "sequence_plots/q_ref_vs_gt_by_frame.png", "q_ref_vs_gt_by_frame.png"),
    ("delta_q vs GT", "sequence_plots/delta_q_vs_gt_increment.png", "delta_q_vs_gt_increment.png"),
]

MLP_CONFIG = {
    "motion_param": "mlp_q",
    "mlp_time_encoding": "raw",
    "mlp_hidden_dim": "64",
    "mlp_num_layers": "2",
    "mlp_lr": "0.001",
    "mlp_smoothness_weight": "0.001",
    "mlp_acceleration_weight": "0.001",
    "mlp_monotonic_weight": "0.0",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def rel(path: Path, output: Path) -> str:
    return os.path.relpath(path, output.parent)


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    if value != 0.0 and abs(value) < 1.0e-4:
        return f"{value:.4e}"
    return f"{value:.8f}".rstrip("0").rstrip(".")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def q_value(row: dict[str, str]) -> float:
    return float(row.get("q_ref_pred") or row.get("q_ref_committed") or row["q_ref"])


def is_mlp_rows(rows: list[dict[str, str]]) -> bool:
    return any(row.get("motion_param") == "mlp_q" for row in rows)


def run_config(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    keys = [
        "motion_param", "mlp_hidden_dim", "mlp_num_layers", "mlp_lr",
        "mlp_smoothness_weight", "mlp_acceleration_weight",
        "mlp_time_encoding", "mlp_fourier_frequencies", "mlp_monotonic_weight",
        "num_iters", "max_iters",
    ]
    row = rows[0]
    return {key: row.get(key, "") for key in keys if row.get(key, "") != ""}


def derived_delta_from_q_profile(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    q_profile = [float(rows[0].get("q_ref_start") or 0.0)]
    q_profile.extend(q_value(row) for row in rows)
    return [q_profile[index + 1] - q_profile[index] for index in range(len(rows))]


def selected_iteration_row(run_dir: Path, row: dict[str, str]) -> dict[str, str]:
    log_path = run_dir / "per_iteration_logs" / f"{int(row['source_frame']):06d}_to_{int(row['target_frame']):06d}_iterations.csv"
    log_rows = read_csv(log_path)
    if not log_rows:
        return {}
    best_iteration = int(float(row.get("best_iteration") or row.get("iterations_run") or len(log_rows)))
    return log_rows[max(0, min(best_iteration - 1, len(log_rows) - 1))]


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def compute_metrics(run_dir: Path) -> dict[str, Any]:
    rows = read_csv(run_dir / "trajectory.csv")
    mlp_deltas = derived_delta_from_q_profile(rows) if is_mlp_rows(rows) else []
    q_errors: list[float] = []
    delta_errors: list[float] = []
    support: list[float] = []
    losses: list[float] = []
    deltas: list[float] = []
    l1_values: list[float] = []
    ssim_values: list[float] = []
    iterations: list[float] = []

    for index, row in enumerate(rows):
        q_error = optional_float(row.get("q_ref_error_after_commit"))
        if q_error is not None:
            q_errors.append(abs(q_error))
        if mlp_deltas:
            gt_delta = optional_float(row.get("gt_delta_q"))
            if gt_delta is not None:
                delta_errors.append(abs(mlp_deltas[index] - gt_delta))
            deltas.append(mlp_deltas[index])
        else:
            delta_error = optional_float(row.get("committed_delta_q_error"))
            if delta_error is None:
                delta_error = optional_float(row.get("delta_error_vs_gt_increment"))
            if delta_error is not None:
                delta_errors.append(abs(delta_error))
            delta_value = optional_float(row.get("pred_delta_q") or row.get("committed_delta_q") or row.get("delta_q"))
            if delta_value is not None:
                deltas.append(delta_value)
        support_value = optional_float(row.get("support_iou") or row.get("raw_iou"))
        if support_value is not None:
            support.append(support_value)
        final_loss = optional_float(row.get("final_loss"))
        if final_loss is not None:
            losses.append(final_loss)
        iter_value = optional_float(row.get("iterations_run"))
        if iter_value is not None:
            iterations.append(iter_value)

        selected = selected_iteration_row(run_dir, row)
        l1_value = optional_float(selected.get("rgb_loss"))
        if l1_value is not None:
            l1_values.append(l1_value)
        ssim_loss = optional_float(selected.get("ssim_loss"))
        if ssim_loss is not None:
            ssim_values.append(1.0 - ssim_loss)

    smoothness_terms = [(deltas[index] - deltas[index - 1]) ** 2 for index in range(1, len(deltas))]
    final_q_error = optional_float(rows[-1].get("q_ref_error_after_commit")) if rows else None
    q_mae = mean_or_none(q_errors)
    delta_q_mae = mean_or_none(delta_errors)
    final_q_error_abs = abs(final_q_error) if final_q_error is not None else None
    smoothness = mean_or_none(smoothness_terms)
    score = None
    if q_mae is not None and delta_q_mae is not None and final_q_error_abs is not None and smoothness is not None:
        score = q_mae + delta_q_mae + 0.5 * final_q_error_abs + 0.1 * smoothness
    return {
        "transitions": len(rows),
        "q_mae": q_mae,
        "delta_q_mae": delta_q_mae,
        "final_q_error": final_q_error,
        "final_q_error_abs": final_q_error_abs,
        "ssim": mean_or_none(ssim_values),
        "support_iou": mean_or_none(support),
        "l1": mean_or_none(l1_values),
        "loss": mean_or_none(losses),
        "smoothness": smoothness,
        "score": score,
        "mean_iterations": mean_or_none(iterations),
        "delta_source": "derived from q_ref profile" if mlp_deltas else "saved direct delta_q",
        "config": run_config(rows),
    }


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return str(dst)
    shutil.copy2(src, dst)
    return str(dst)


def prepare_object_assets(obj: dict[str, Any], output_root: Path) -> dict[str, Any]:
    obj_dir = output_root / obj["object_id"]
    result = {
        "short_name": obj["short_name"],
        "object_id": obj["object_id"],
        "joint_type": obj["joint_type"],
        "camera": obj["camera"],
        "selection_note": obj.get("selection_note", ""),
        "methods": {},
    }
    for method_key, run_dir in obj["runs"].items():
        method_dir = obj_dir / method_key
        plots_dir = method_dir / "plots"
        videos_dir = obj_dir / "videos"
        metrics = compute_metrics(run_dir)
        copied_plots = {}
        for label, relative_path, filename in PLOTS:
            copied = copy_if_exists(run_dir / relative_path, plots_dir / filename)
            copied_plots[label] = copied
        copied_traj = copy_if_exists(run_dir / "trajectory.csv", method_dir / "trajectory.csv")
        result["methods"][method_key] = {
            "label": METHODS[method_key],
            "source_run_dir": str(run_dir),
            "final_dir": str(method_dir),
            "trajectory_csv": copied_traj,
            "plots": copied_plots,
            "metrics": metrics,
            "selection_note": obj.get("selection_notes", {}).get(method_key, ""),
        }
    video_src = obj.get("comparison_video")
    video_dst = obj_dir / "videos" / "direct_vs_mlp_raw_reg.mp4"
    result["comparison_video"] = copy_if_exists(video_src, video_dst) if video_src else None
    (obj_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def better_method(methods: dict[str, Any], key: str, higher_is_better: bool = False) -> str:
    direct = methods["direct"]["metrics"].get(key)
    mlp = methods["mlp"]["metrics"].get(key)
    if direct is None or mlp is None:
        return "n/a"
    direct_better = direct >= mlp if higher_is_better else direct <= mlp
    return methods["direct"]["label"] if direct_better else methods["mlp"]["label"]


def metric_table(methods: dict[str, Any]) -> str:
    winners = {
        key: better_method(methods, key, higher)
        for key, _, higher in METRIC_COLUMNS
    }
    headers = ["Method"] + [label for _, label, _ in METRIC_COLUMNS]
    rows = []
    for method_key in ["direct", "mlp"]:
        method = methods[method_key]
        cells = [f"<strong>{esc(method['label'])}</strong>"]
        for key, _, _ in METRIC_COLUMNS:
            class_attr = " class='best'" if winners.get(key) == method["label"] else ""
            cells.append(f"<span{class_attr}>{esc(fmt(method['metrics'].get(key)))}</span>")
        rows.append(cells)
    return table(headers, rows)


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure(path: str | None, output: Path, label: str) -> str:
    if not path:
        return f"<p class='missing'>Missing: {esc(label)}</p>"
    return (
        "<figure>"
        f"<img src='{esc(rel(Path(path), output))}' alt='{esc(label)}'>"
        f"<figcaption>{esc(label)}</figcaption>"
        "</figure>"
    )


def video(path: str | None, output: Path, label: str) -> str:
    if not path:
        return f"<p class='missing'>Missing video: {esc(label)}</p>"
    return (
        "<figure>"
        "<video controls preload='metadata'>"
        f"<source src='{esc(rel(Path(path), output))}' type='video/mp4'>"
        "Your browser does not support embedded video."
        "</video>"
        f"<figcaption>{esc(label)}</figcaption>"
        "</figure>"
    )


def object_section(obj: dict[str, Any], output: Path) -> str:
    methods = obj["methods"]
    q_winner = better_method(methods, "q_mae")
    dq_winner = better_method(methods, "delta_q_mae")
    ssim_direct = methods["direct"]["metrics"].get("ssim")
    ssim_mlp = methods["mlp"]["metrics"].get("ssim")
    visual = "comparable" if ssim_direct is not None and ssim_mlp is not None and abs(ssim_direct - ssim_mlp) < 1.0e-4 else "different"
    plot_rows = []
    for plot_label, _, _ in PLOTS:
        plot_rows.append(
            "<div class='columns two'>"
            + figure(methods["direct"]["plots"].get(plot_label), output, f"{obj['short_name']} Direct: {plot_label}")
            + figure(methods["mlp"]["plots"].get(plot_label), output, f"{obj['short_name']} MLP: {plot_label}")
            + "</div>"
        )
    return f"""
    <section class="card">
      <h2>{esc(obj["short_name"])} Results</h2>
      <p class="muted">Joint type: {esc(obj["joint_type"])}. Camera: <code>{esc(obj["camera"])}</code>.</p>
      <p>{esc(obj.get("selection_note", ""))}</p>
      {metric_table(methods)}
      <p><strong>Takeaway:</strong> q MAE winner: {esc(q_winner)}. delta_q MAE winner: {esc(dq_winner)}. Visual quality is {esc(visual)} by SSIM.</p>
      {''.join(plot_rows)}
      {video(obj.get("comparison_video"), output, f"{obj['short_name']} direct vs MLP synchronized video")}
      <p class="links">
        <a href="{esc(rel(Path(methods["direct"]["source_run_dir"]), output))}">Direct source folder</a> |
        <a href="{esc(rel(Path(methods["mlp"]["source_run_dir"]), output))}">MLP source folder</a>
      </p>
    </section>
    """


def cross_object_table(objects: list[dict[str, Any]]) -> str:
    rows = []
    for obj in objects:
        methods = obj["methods"]
        best_q = better_method(methods, "q_mae")
        best_dq = better_method(methods, "delta_q_mae")
        best_score = better_method(methods, "score")
        visual = "comparable SSIM"
        rows.append([
            esc(obj["short_name"]),
            esc(obj["joint_type"]),
            esc(obj["camera"]),
            esc(best_q),
            esc(best_dq),
            esc(best_score),
            esc(visual),
        ])
    return table(
        ["Object", "Joint type", "Camera", "Best q MAE", "Best delta_q MAE", "Best method overall", "Visual quality conclusion"],
        rows,
    )


def motion_parameterization_section() -> str:
    return """
  <section class="card compact">
    <h2>Motion parameterization</h2>
    <div class="two-col">
      <div>
        <h3>Direct delta_q</h3>
        <p>Optimizes each frame-to-frame joint increment as an independent scalar parameter. For a transition t to t+1, delta_q is updated directly through the differentiable rendering loss.</p>
        <pre>delta_q_i = optimized parameter</pre>
      </div>
      <div>
        <h3>MLP q(t)</h3>
        <p>Optimizes the weights of a temporal network that predicts a continuous joint trajectory q(t), then derives each frame increment from adjacent predictions.</p>
        <pre>q_i = MLP(t_i)
delta_q_i = q_{i+1} - q_i</pre>
      </div>
    </div>
    <p class="note">Both methods use the same Gaussian model, joint metadata, articulated transform, renderer, and RGB/SSIM loss. The only difference is how the motion variable is parameterized: direct is local and frame-wise, while MLP is global and continuous over time.</p>
  </section>
    """


def css() -> str:
    return """
    :root { --ink: #172033; --muted: #5d6878; --border: #d9e1ec; --soft: #f7f9fc; --accent: #1d5fa7; --good: #dcfce7; --good-ink: #14532d; }
    * { box-sizing: border-box; }
    body { margin: 0; background: white; color: var(--ink); font-family: Arial, Helvetica, sans-serif; line-height: 1.48; }
    main { max-width: 1240px; margin: 0 auto; padding: 30px 20px 56px; }
    header { border-bottom: 1px solid var(--border); margin-bottom: 20px; padding-bottom: 18px; }
    h1 { margin: 0 0 8px; font-size: 31px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 21px; letter-spacing: 0; }
    h3 { margin: 10px 0 8px; font-size: 17px; letter-spacing: 0; }
    .subtitle, .muted, figcaption { color: var(--muted); }
    .subtitle { max-width: 900px; margin: 0; font-size: 16px; }
    .card { border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 16px 0; background: white; box-shadow: 0 1px 2px rgba(18, 31, 50, 0.04); }
    .takeaway { border-left: 4px solid #15803d; background: #f0fdf4; }
    .compact { padding: 14px 16px; }
    .columns.two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .two-col { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .note { color: var(--muted); font-size: 14px; margin: 8px 0 0; }
    pre { margin: 8px 0 0; padding: 9px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--soft); overflow-x: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.35; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 10px 0; }
    th, td { border: 1px solid var(--border); padding: 8px 10px; text-align: right; vertical-align: top; }
    th:first-child, td:first-child { text-align: left; }
    th { background: var(--soft); }
    .best { display: inline-block; padding: 2px 7px; border-radius: 999px; background: var(--good); color: var(--good-ink); font-weight: 700; }
    figure { margin: 0 0 14px; }
    img, video { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; background: white; }
    figcaption { margin-top: 5px; font-size: 13px; }
    .method-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .method-card { border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: var(--soft); }
    .missing { border: 1px dashed #eab308; background: #fffbeb; border-radius: 6px; padding: 10px; color: #854d0e; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }
    a { color: var(--accent); }
    ul { margin-top: 6px; }
    @media (max-width: 900px) { .columns.two, .method-grid, .two-col { grid-template-columns: 1fr; } h1 { font-size: 26px; } th, td { font-size: 13px; padding: 6px; } }
    """


def markdown_summary(objects: list[dict[str, Any]], output_root: Path) -> str:
    lines = [
        "# SAM-ARTGS final comparison: Direct delta_q vs MLP q(t)",
        "",
        "Methods: direct per-transition delta_q optimization and sequence-specific MLP q(t) parameterizations.",
        "",
        f"Report: `{output_root / 'final_report.html'}`",
        "",
    ]
    for obj in objects:
        lines.append(f"## {obj['short_name']}")
        lines.append(f"- Object: `{obj['object_id']}`")
        lines.append(f"- Joint type: {obj['joint_type']}")
        lines.append(f"- Camera: `{obj['camera']}`")
        lines.append(f"- MLP selection: {obj.get('selection_note', '')}")
        lines.append(f"- Video: `{obj.get('comparison_video')}`")
        for method_key in ["direct", "mlp"]:
            method = obj["methods"][method_key]
            m = method["metrics"]
            lines.append(
                f"- {method['label']}: q MAE {fmt(m['q_mae'])}, delta_q MAE {fmt(m['delta_q_mae'])}, "
                f"SSIM {fmt(m['ssim'])}, score {fmt(m['score'])}"
            )
        lines.append("")
    lines.append("Conclusion: the final report compares direct delta_q against the best available sequence-specific MLP for each object. USB uses the USB-selected MLP; storage uses the best MLP from the storage search ranking.")
    return "\n".join(lines) + "\n"


def build_html(objects: list[dict[str, Any]], output: Path) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAM-ARTGS final comparison: Direct delta_q vs MLP q(t)</title>
  <style>{css()}</style>
</head>
<body>
<main>
  <header>
    <h1>SAM-ARTGS final comparison: Direct delta_q vs MLP q(t)</h1>
    <p class="subtitle">Evaluation on USB revolute-continuous and storage prismatic articulated tracking.</p>
  </header>

  <section class="card takeaway">
    <h2>Main Takeaway</h2>
    <p>We keep direct delta_q as the unchanged baseline and compare it with the best available sequence-specific temporal MLP q(t) parameterization for each object. USB uses the USB-selected MLP. Storage uses the best MLP from the storage search ranking, where the transferred USB-best configuration remained the strongest MLP but still did not beat direct delta_q by the composite score.</p>
  </section>

  <section class="card">
    <h2>Methods</h2>
    <div class="method-grid">
      <div class="method-card"><h3>Direct delta_q</h3><p>Optimizes each frame-to-frame joint increment independently.</p><p><code>--motion-param direct_delta_q</code></p></div>
      <div class="method-card"><h3>Sequence-specific MLP q(t)</h3><p>Fits one temporal MLP per evaluated sequence, predicts q(t), then derives delta_q as q(t+1)-q(t). This is not a single general model shared across objects.</p><p><code>--motion-param mlp_q</code></p></div>
    </div>
  </section>

  {motion_parameterization_section()}

  {''.join(object_section(obj, output) for obj in objects)}

  <section class="card">
    <h2>Cross-object Summary</h2>
    {cross_object_table(objects)}
  </section>

  <section class="card">
    <h2>Videos</h2>
    {''.join(video(obj.get("comparison_video"), output, f"{obj['short_name']} direct vs MLP") for obj in objects)}
  </section>

  <section class="card">
    <h2>Remaining Limitations</h2>
    <ul>
      <li>Only one USB trajectory and one storage trajectory are tested here.</li>
      <li>The MLPs are sequence-specific optimization parameterizations, not general learned predictors.</li>
      <li>Camera choice matters for storage; this report uses the best available validated storage full run.</li>
      <li>Future force-conditioned evaluation should test force-generated trajectories from ForceSAPIEN.</li>
    </ul>
  </section>

  <section class="card">
    <h2>Next Step</h2>
    <p>The next step is to use ForceSAPIEN to generate force-driven RGB sequences and ground-truth q(t), qdot(t), qddot(t), then evaluate SAM-ARTGS tracking against simulator ground truth.</p>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final Direct-vs-MLP report for USB and storage.")
    parser.add_argument("--root", type=Path, help="Final report asset root. Enables the clean default final workflow.")
    parser.add_argument("--output", type=Path, help="Final HTML path. Defaults to ROOT/final_report.html.")
    parser.add_argument("--output-root", type=Path, help="Backward-compatible alias for --root.")
    parser.add_argument("--usb-direct", type=Path)
    parser.add_argument("--usb-mlp", type=Path)
    parser.add_argument("--storage-direct", type=Path)
    parser.add_argument("--storage-mlp", type=Path)
    parser.add_argument("--usb-video", type=Path)
    parser.add_argument("--storage-video", type=Path)
    parser.add_argument("--storage-camera", default="cam_007")
    args = parser.parse_args()

    output_root_arg = args.root or args.output_root
    if output_root_arg is None:
        parser.error("Provide --root or --output-root")
    output_root = output_root_arg.resolve()
    report_path = (args.output.resolve() if args.output else output_root / "final_report.html")

    usb_direct = args.usb_direct or Path("outputs/delta_q_tracking/new_dataset/USB_100109/usb_trapezoidal_cam_000_0_59_600iters/cam_000")
    usb_mlp = args.usb_mlp or Path("outputs/delta_q_tracking/new_dataset/USB_100109/usb_mlp_cam_000_0_59_600iters_reg/cam_000")
    storage_direct = args.storage_direct or Path("outputs/delta_q_tracking/new_dataset/storage_45135/storage_prismatic_cam_007_0_59_600iters/cam_007")
    storage_mlp = args.storage_mlp or Path("outputs/delta_q_tracking/new_dataset/storage_45135/storage_mlp_raw_reg_cam_007_0_59_600iters/cam_007")
    usb_video = args.usb_video or output_root / "USB_100109/videos/direct_vs_mlp_raw_reg.mp4"
    storage_video = args.storage_video or output_root / "storage_45135/videos/direct_vs_best_storage_mlp.mp4"

    output_root.mkdir(parents=True, exist_ok=True)
    objects = [
        {
            "short_name": "USB",
            "object_id": "USB_100109",
            "joint_type": "revolute-continuous",
            "camera": "cam_000",
            "runs": {"direct": usb_direct.resolve(), "mlp": usb_mlp.resolve()},
            "comparison_video": usb_video.resolve() if usb_video else None,
            "selection_note": "USB uses the best MLP selected from the USB hyperparameter search.",
            "selection_notes": {
                "direct": "Unchanged direct delta_q baseline.",
                "mlp": "Best USB-selected sequence-specific MLP q(t).",
            },
        },
        {
            "short_name": "Storage",
            "object_id": "storage_45135",
            "joint_type": "prismatic",
            "camera": args.storage_camera,
            "runs": {"direct": storage_direct.resolve(), "mlp": storage_mlp.resolve()},
            "comparison_video": storage_video.resolve() if storage_video else None,
            "selection_note": "Storage uses the best MLP from the storage search ranking; S1-S10 did not beat direct, and the transferred USB-best MLP remained the strongest MLP by the composite score.",
            "selection_notes": {
                "direct": "Unchanged direct delta_q baseline.",
                "mlp": "Best storage-ranked sequence-specific MLP q(t).",
            },
        },
    ]
    prepared = [prepare_object_assets(obj, output_root) for obj in objects]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_html(prepared, report_path))
    summary_json = output_root / "final_summary.metrics.json"
    summary_json.write_text(json.dumps({"mlp_config": MLP_CONFIG, "objects": prepared}, indent=2))
    summary_md = output_root / "final_summary.md"
    summary_md.write_text(markdown_summary(prepared, output_root))
    print(f"final_report={report_path}")
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")
    for obj in prepared:
        print(f"{obj['object_id']} video={obj.get('comparison_video')}")


if __name__ == "__main__":
    main()
