from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCE_DIR = REPO_ROOT / "outputs/delta_q_tracking/usb/final_rigid_usb_gauss_new/cam_000"
DEFAULT_RGB_DIR = REPO_ROOT / "../dataset/usb_rgbdm/rgb/cam_000"
DEFAULT_ASSET_DIR = REPO_ROOT / "docs/assets/readme"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight README visual assets from a final delta-q run.")
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-panel-height", type=int, default=360)
    return parser.parse_args()


def copy_plot(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing plot: {src}")
    shutil.copy2(src, dst)


def frame_index(path: Path, prefix: str) -> int:
    return int(path.stem.replace(prefix, ""))


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_labeled_pair(rgb: Image.Image, pred: Image.Image, max_panel_height: int) -> Image.Image:
    panel_height = min(max_panel_height, rgb.height, pred.height)
    rgb = resize_to_height(rgb.convert("RGB"), panel_height)
    pred = resize_to_height(pred.convert("RGB"), panel_height)

    label_h = 32
    gap = 8
    width = rgb.width + gap + pred.width
    out = Image.new("RGB", (width, label_h + panel_height), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    draw.text((10, 8), "Target RGB", fill=(20, 30, 45), font=font)
    draw.text((rgb.width + gap + 10, 8), "Predicted Gaussian", fill=(20, 30, 45), font=font)
    out.paste(rgb, (0, label_h))
    out.paste(pred, (rgb.width + gap, label_h))
    return out


def write_mp4(frames: list[Image.Image], path: Path, fps: int) -> bool:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        print(f"Skipping MP4 because OpenCV/numpy import failed: {exc}")
        return False

    first = frames[0]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.width, first.height),
    )
    if not writer.isOpened():
        print(f"Skipping MP4 because VideoWriter could not open: {path}")
        return False
    for frame in frames:
        arr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        writer.write(arr)
    writer.release()
    return True


def build_animation(sequence_dir: Path, rgb_dir: Path, asset_dir: Path, fps: int, sample_every: int, max_panel_height: int) -> tuple[Path, Path | None, int]:
    rgb_frames = {frame_index(path, "frame_"): path for path in rgb_dir.glob("frame_*.png")}
    pred_frames = {frame_index(path, "pred_raw_frame_"): path for path in sequence_dir.glob("pred_raw_frame_*.png")}
    indices = sorted(set(rgb_frames) & set(pred_frames))
    if not indices:
        raise RuntimeError(f"No matching RGB/predicted frames found in {rgb_dir} and {sequence_dir}")
    indices = indices[:: max(1, sample_every)]

    frames: list[Image.Image] = []
    for idx in indices:
        rgb = Image.open(rgb_frames[idx])
        pred = Image.open(pred_frames[idx])
        frames.append(make_labeled_pair(rgb, pred, max_panel_height))

    gif_path = asset_dir / "rgb_vs_predicted_gauss.gif"
    duration_ms = round(1000 / fps)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    mp4_path = asset_dir / "rgb_vs_predicted_gauss.mp4"
    if not write_mp4(frames, mp4_path, fps):
        mp4_path = None
    return gif_path, mp4_path, len(frames)


def main() -> None:
    args = parse_args()
    if not args.sequence_dir.exists():
        raise FileNotFoundError(f"Final sequence directory does not exist: {args.sequence_dir}")
    if not args.rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {args.rgb_dir}")
    args.asset_dir.mkdir(parents=True, exist_ok=True)

    sequence_plots = args.sequence_dir / "sequence_plots"
    copy_plot(sequence_plots / "q_ref_vs_gt_by_frame.png", args.asset_dir / "q_ref_vs_gt.png")
    copy_plot(sequence_plots / "delta_q_vs_required_delta_by_frame.png", args.asset_dir / "delta_q_vs_required_delta.png")
    gif_path, mp4_path, frame_count = build_animation(
        args.sequence_dir,
        args.rgb_dir,
        args.asset_dir,
        args.fps,
        args.sample_every,
        args.max_panel_height,
    )

    print(f"Copied plots to: {args.asset_dir}")
    print(f"Created GIF: {gif_path}")
    if mp4_path is not None:
        print(f"Created MP4: {mp4_path}")
    print(f"Animation frames: {frame_count}")


if __name__ == "__main__":
    main()
