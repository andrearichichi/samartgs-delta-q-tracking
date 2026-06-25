from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.make_tracking_videos import (
    even_size,
    parse_frame_index,
    resolve_repo_path,
    rgb_dir_from_run,
    write_video,
)


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    image = image.convert("RGB")
    if image.height == height:
        return image
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 18) -> None:
    draw.text(xy, text, fill=(20, 30, 45), font=font(size))


def pair_frame(rgb: Image.Image, pred: Image.Image, method_label: str, frame_index: int, max_panel_height: int) -> Image.Image:
    panel_height = min(max_panel_height, rgb.height, pred.height)
    rgb = resize_to_height(rgb, panel_height)
    pred = resize_to_height(pred, panel_height)
    gap = 8
    label_height = 58
    width = rgb.width + gap + pred.width
    out = Image.new("RGB", (width, label_height + panel_height), "white")
    draw = ImageDraw.Draw(out)
    draw_label(draw, (10, 7), method_label, 18)
    draw_label(draw, (10, 34), f"RGB target | frame {frame_index:06d}", 14)
    draw_label(draw, (rgb.width + gap + 10, 34), "Gaussian render", 14)
    out.paste(rgb, (0, label_height))
    out.paste(pred, (rgb.width + gap, label_height))
    return even_size(out)


def make_pair_video(
    run_dir: Path,
    rgb_dir: Path,
    output: Path,
    label: str,
    fps: int,
    max_panel_height: int,
) -> dict[str, object]:
    pred_frames = {
        parse_frame_index(path, "pred_raw_frame_"): path
        for path in run_dir.glob("pred_raw_frame_*.png")
    }
    rgb_frames = {
        parse_frame_index(path, "frame_"): path
        for path in rgb_dir.glob("frame_*.png")
    }
    frame_indices = sorted(set(pred_frames) & set(rgb_frames))
    if not frame_indices:
        raise RuntimeError(f"No matching RGB/render frames found for {run_dir}")

    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for out_index, frame_index in enumerate(frame_indices):
        rgb = Image.open(rgb_frames[frame_index]).convert("RGB")
        pred = Image.open(pred_frames[frame_index]).convert("RGB")
        pair_frame(rgb, pred, label, frame_index, max_panel_height).save(frame_dir / f"frame_{out_index:06d}.png")
    writer = write_video(frame_dir, output, fps)
    return {
        "output": str(output),
        "writer": writer,
        "frame_count": len(frame_indices),
        "first_frame": frame_indices[0],
        "last_frame": frame_indices[-1],
        "frame_dir": str(frame_dir),
    }


def triptych_frame(
    columns: list[tuple[str, Image.Image, Image.Image]],
    frame_index: int,
    max_panel_height: int,
) -> Image.Image:
    rendered_columns = [
        pair_frame(rgb, pred, label, frame_index, max_panel_height)
        for label, rgb, pred in columns
    ]
    target_height = min(image.height for image in rendered_columns)
    rendered_columns = [resize_to_height(image, target_height) for image in rendered_columns]
    gap = 10
    width = sum(image.width for image in rendered_columns) + gap * (len(rendered_columns) - 1)
    out = Image.new("RGB", (width, target_height), "white")
    x = 0
    for image in rendered_columns:
        out.paste(image, (x, 0))
        x += image.width + gap
    return even_size(out)


def make_triptych_video(
    runs: list[tuple[str, Path]],
    rgb_dir: Path,
    output: Path,
    fps: int,
    max_panel_height: int,
) -> dict[str, object]:
    rgb_frames = {
        parse_frame_index(path, "frame_"): path
        for path in rgb_dir.glob("frame_*.png")
    }
    pred_by_run = []
    for label, run_dir in runs:
        pred_frames = {
            parse_frame_index(path, "pred_raw_frame_"): path
            for path in run_dir.glob("pred_raw_frame_*.png")
        }
        pred_by_run.append((label, run_dir, pred_frames))

    frame_sets = [set(rgb_frames)] + [set(pred_frames) for _, _, pred_frames in pred_by_run]
    frame_indices = sorted(set.intersection(*frame_sets))
    if not frame_indices:
        raise RuntimeError("No synchronized frames found for triptych video")

    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for out_index, frame_index in enumerate(frame_indices):
        rgb = Image.open(rgb_frames[frame_index]).convert("RGB")
        columns = []
        for label, _, pred_frames in pred_by_run:
            pred = Image.open(pred_frames[frame_index]).convert("RGB")
            columns.append((label, rgb, pred))
        triptych_frame(columns, frame_index, max_panel_height).save(frame_dir / f"frame_{out_index:06d}.png")
    writer = write_video(frame_dir, output, fps)
    return {
        "output": str(output),
        "writer": writer,
        "frame_count": len(frame_indices),
        "first_frame": frame_indices[0],
        "last_frame": frame_indices[-1],
        "frame_dir": str(frame_dir),
    }


def infer_rgb_dir(run_dir: Path, override_rgb_dir: Path | None) -> Path:
    if override_rgb_dir is not None:
        return resolve_repo_path(override_rgb_dir)
    payload_path = run_dir / "trajectory.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing trajectory metadata: {payload_path}")
    return rgb_dir_from_run(json.loads(payload_path.read_text()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RGB-target vs Gaussian-render videos from existing tracking frames.")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="Run folder. Repeat for triptych.")
    parser.add_argument("--label", action="append", default=[], help="Method label. Repeat with --run-dir.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-panel-height", type=int, default=360)
    parser.add_argument("--triptych", action="store_true")
    args = parser.parse_args()

    run_dirs = [resolve_repo_path(path) for path in args.run_dir]
    if len(run_dirs) != len(args.label):
        raise ValueError("--run-dir and --label must be provided the same number of times")
    if not run_dirs:
        raise ValueError("At least one --run-dir is required")
    output = resolve_repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb_dir = infer_rgb_dir(run_dirs[0], args.rgb_dir)

    if args.triptych:
        result = make_triptych_video(
            list(zip(args.label, run_dirs)),
            rgb_dir,
            output,
            args.fps,
            args.max_panel_height,
        )
    else:
        if len(run_dirs) != 1:
            raise ValueError("Non-triptych mode expects exactly one --run-dir")
        result = make_pair_video(run_dirs[0], rgb_dir, output, args.label[0], args.fps, args.max_panel_height)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
