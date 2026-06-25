from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.delta_q_tracking.dataset_manifest import load_dataset_manifest


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_frame_index(path: Path, prefix: str) -> int:
    return int(path.stem.replace(prefix, ""))


def load_run_metadata(run_dir: Path) -> dict:
    trajectory_path = run_dir / "trajectory.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory metadata: {trajectory_path}")
    return json.loads(trajectory_path.read_text())


def rgb_dir_from_run(run_payload: dict, override_rgb_dir: Path | None) -> Path:
    if override_rgb_dir is not None:
        return resolve_repo_path(override_rgb_dir)

    manifest_path = run_payload.get("manifest")
    object_id = run_payload.get("object_id")
    camera = run_payload.get("camera")
    if not manifest_path or not object_id or not camera:
        raise ValueError(
            "Cannot infer RGB directory. Provide --rgb-dir, or run from a manifest-based output "
            "with manifest/object_id/camera recorded in trajectory.json."
        )
    manifest = load_dataset_manifest(resolve_repo_path(Path(manifest_path)))
    dataset_object = manifest.get(str(object_id))
    return dataset_object.rgb_dir / str(camera)


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    image = image.convert("RGB")
    if image.height == height:
        return image
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def even_size(image: Image.Image) -> Image.Image:
    width = image.width if image.width % 2 == 0 else image.width + 1
    height = image.height if image.height % 2 == 0 else image.height + 1
    if (width, height) == image.size:
        return image
    out = Image.new("RGB", (width, height), "black")
    out.paste(image, (0, 0))
    return out


def labeled_pair(left: Image.Image, right: Image.Image, max_panel_height: int) -> Image.Image:
    panel_height = min(max_panel_height, left.height, right.height)
    left = resize_to_height(left, panel_height)
    right = resize_to_height(right, panel_height)

    label_height = 32
    gap = 8
    width = left.width + gap + right.width
    out = Image.new("RGB", (width, label_height + panel_height), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), "RGB target", fill=(20, 30, 45), font=font)
    draw.text((left.width + gap + 10, 8), "Gaussian render", fill=(20, 30, 45), font=font)
    out.paste(left, (0, label_height))
    out.paste(right, (left.width + gap, label_height))
    return even_size(out)


def copy_or_resize_frame(src: Path, dst: Path, size: tuple[int, int] | None = None) -> None:
    image = Image.open(src).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    even_size(image).save(dst)


def write_video_ffmpeg(frame_dir: Path, output_path: Path, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def write_video_cv2(frame_dir: Path, output_path: Path, fps: int, fourcc_values: list[str]) -> str:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        raise RuntimeError(
            "Could not write MP4: ffmpeg is unavailable and cv2 import failed"
        ) from exc

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No video frames found in {frame_dir}")
    first = Image.open(frames[0]).convert("RGB")
    writer = None
    selected_fourcc = None
    for fourcc in fourcc_values:
        candidate = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            float(fps),
            first.size,
        )
        if candidate.isOpened():
            writer = candidate
            selected_fourcc = fourcc
            break
        candidate.release()
    if writer is None or selected_fourcc is None:
        raise RuntimeError(f"OpenCV VideoWriter could not open {output_path}")
    for frame_path in frames:
        image = Image.open(frame_path).convert("RGB")
        if image.size != first.size:
            image = image.resize(first.size, Image.Resampling.LANCZOS)
        writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    writer.release()
    return selected_fourcc


def write_video(frame_dir: Path, output_path: Path, fps: int) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if write_video_ffmpeg(frame_dir, output_path, fps):
        return "ffmpeg/libx264/yuv420p"
    fourcc = write_video_cv2(frame_dir, output_path, fps, ["mp4v"])
    return f"opencv/{fourcc}"


def write_webm(frame_dir: Path, output_path: Path, fps: int) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = write_video_cv2(frame_dir, output_path, fps, ["VP90", "VP80"])
    return f"opencv/{fourcc}"


def write_poster(frame_dir: Path, output_path: Path) -> None:
    first_frame = frame_dir / "frame_000000.png"
    if not first_frame.exists():
        raise FileNotFoundError(f"Cannot create poster; missing first frame: {first_frame}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(first_frame).convert("RGB").save(output_path)


def build_videos(run_dir: Path, rgb_dir: Path, fps: int, max_panel_height: int) -> dict[str, object]:
    pred_frames = {
        parse_frame_index(path, "pred_raw_frame_"): path
        for path in run_dir.glob("pred_raw_frame_*.png")
    }
    overlay_frames = {
        parse_frame_index(path, "overlay_raw_frame_"): path
        for path in run_dir.glob("overlay_raw_frame_*.png")
    }
    rgb_frames = {
        parse_frame_index(path, "frame_"): path
        for path in rgb_dir.glob("frame_*.png")
    }
    frame_indices = sorted(set(pred_frames) & set(rgb_frames))
    if not frame_indices:
        raise RuntimeError(f"No matching RGB/predicted frames found in {rgb_dir} and {run_dir}")

    frame_root = run_dir / "frames"
    rgb_out = frame_root / "rgb"
    gaussian_out = frame_root / "gaussian"
    overlay_out = frame_root / "overlay"
    pair_out = frame_root / "side_by_side"
    for directory in [rgb_out, gaussian_out, overlay_out, pair_out]:
        directory.mkdir(parents=True, exist_ok=True)

    missing_overlay: list[int] = []
    for out_index, frame_index in enumerate(frame_indices):
        rgb = Image.open(rgb_frames[frame_index]).convert("RGB")
        pred = Image.open(pred_frames[frame_index]).convert("RGB")
        target_name = f"frame_{out_index:06d}.png"

        even_size(rgb).save(rgb_out / target_name)
        even_size(pred).save(gaussian_out / target_name)
        labeled_pair(rgb, pred, max_panel_height).save(pair_out / target_name)

        overlay_path = overlay_frames.get(frame_index)
        if overlay_path is None:
            missing_overlay.append(frame_index)
        else:
            copy_or_resize_frame(overlay_path, overlay_out / target_name)

    videos_dir = run_dir / "videos"
    results: dict[str, object] = {
        "frame_count": len(frame_indices),
        "first_frame": frame_indices[0],
        "last_frame": frame_indices[-1],
        "rgb_dir": str(rgb_dir),
        "missing_overlay_frames": missing_overlay,
        "videos": {},
    }
    video_specs = [
        ("rgb_target", rgb_out),
        ("gaussian_render", gaussian_out),
        ("rgb_vs_gaussian", pair_out),
    ]
    if not missing_overlay:
        video_specs.append(("overlay", overlay_out))

    for name, frame_dir in video_specs:
        mp4_path = videos_dir / f"{name}.mp4"
        webm_path = videos_dir / f"{name}.webm"
        poster_path = videos_dir / f"{name}_poster.png"
        mp4_method = write_video(frame_dir, mp4_path, fps)
        webm_method = write_webm(frame_dir, webm_path, fps)
        write_poster(frame_dir, poster_path)
        results["videos"][name] = {
            "mp4_path": str(mp4_path),
            "mp4_writer": mp4_method,
            "webm_path": str(webm_path),
            "webm_writer": webm_method,
            "poster_path": str(poster_path),
            "fps": fps,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MP4 videos from a delta-q tracking run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-panel-height", type=int, default=480)
    args = parser.parse_args()

    run_dir = resolve_repo_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    payload = load_run_metadata(run_dir)
    rgb_dir = rgb_dir_from_run(payload, args.rgb_dir)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")

    results = build_videos(run_dir, rgb_dir, args.fps, args.max_panel_height)
    summary_path = run_dir / "videos" / "video_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"run_dir={run_dir}")
    print(f"rgb_dir={rgb_dir}")
    print(f"frames={results['frame_count']} first={results['first_frame']} last={results['last_frame']}")
    for name, info in results["videos"].items():
        print(f"{name}_mp4={info['mp4_path']} writer={info['mp4_writer']}")
        print(f"{name}_webm={info['webm_path']} writer={info['webm_writer']}")
        print(f"{name}_poster={info['poster_path']}")
    if results["missing_overlay_frames"]:
        print(f"missing_overlay_frames={results['missing_overlay_frames']}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
