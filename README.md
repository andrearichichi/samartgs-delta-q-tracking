# SAM-ARTGS Delta-Q Tracking

This repository builds on the GraphDECO 3D Gaussian Splatting codebase with a compact articulated tracking pipeline for SAM-ARTGS.

## SAM-ARTGS delta-q tracking

### What this does

The delta-q tracker estimates articulated object motion while keeping the reconstructed Gaussian model fixed. The Gaussian parameters are kept frozen. The tracker estimates the articulated motion by rendering the deformed Gaussian model and minimizing an RGB/SSIM reconstruction loss against the target frame.

The pipeline uses object joint metadata and part labels to transform only the moving Gaussians, render the predicted frame, and backpropagate into a low-dimensional motion parameter.

### Supported motion parameterizations

**Direct delta_q:** Optimizes each frame-to-frame joint increment directly as an independent scalar parameter.

**MLP q(t):** Optimizes a small temporal MLP that predicts a continuous joint trajectory `q(t)`. The frame increment is derived as `delta_q(t) = q(t+1) - q(t)`.

Both modes use the same Gaussian model, joint metadata, articulated transform, renderer, and RGB/SSIM loss. Only the motion parameterization changes.

### Important default

The default mode is `direct_delta_q`.

### Example direct command

```bash
python scripts/delta_q_tracking/run_sequence.py \
  --manifest configs/delta_q_tracking/dataset_manifest.json \
  --object-id USB_100109 \
  --camera-id cam_000 \
  --start-frame 0 \
  --end-frame 59 \
  --num-iters 600 \
  --motion-param direct_delta_q \
  --output-subdir usb_direct_example
```

### Example MLP command

```bash
python scripts/delta_q_tracking/run_sequence.py \
  --manifest configs/delta_q_tracking/dataset_manifest.json \
  --object-id USB_100109 \
  --camera-id cam_000 \
  --start-frame 0 \
  --end-frame 59 \
  --num-iters 600 \
  --motion-param mlp_q \
  --mlp-time-encoding raw \
  --mlp-hidden-dim 64 \
  --mlp-num-layers 2 \
  --mlp-lr 1e-3 \
  --mlp-smoothness-weight 1e-4 \
  --mlp-acceleration-weight 1e-4 \
  --output-subdir usb_mlp_example
```

### Final report command

```bash
python scripts/delta_q_tracking/reporting/make_final_direct_vs_mlp_report.py \
  --root outputs/delta_q_tracking/final_direct_vs_mlp \
  --output outputs/delta_q_tracking/final_direct_vs_mlp/final_report.html
```

The report generator uses existing tracking outputs and writes:

```text
outputs/delta_q_tracking/final_direct_vs_mlp/final_report.html
outputs/delta_q_tracking/final_direct_vs_mlp/final_summary.metrics.json
outputs/delta_q_tracking/final_direct_vs_mlp/final_summary.md
```

### Final evaluated objects

The current final comparison covers:

```text
USB_100109: revolute-continuous
storage_45135: prismatic
```

### Active files

Core tracking code:

```text
scripts/delta_q_tracking/run_sequence.py
scripts/delta_q_tracking/motion_mlp.py
scripts/delta_q_tracking/articulation.py
scripts/delta_q_tracking/deformed_gaussian.py
scripts/delta_q_tracking/losses.py
scripts/delta_q_tracking/io_utils.py
scripts/delta_q_tracking/dataset_manifest.py
scripts/delta_q_tracking/trajectory_io.py
scripts/delta_q_tracking/make_tracking_videos.py
```

Reporting and diagnostics:

```text
scripts/delta_q_tracking/reporting/plot_tracking_diagnostics.py
scripts/delta_q_tracking/reporting/make_rgb_vs_gaussian_video.py
scripts/delta_q_tracking/reporting/make_final_direct_vs_mlp_report.py
```

Dataset manifest:

```text
configs/delta_q_tracking/dataset_manifest.json
```

### Limitations / next step

The current MLP is sequence-specific: it is optimized per sequence and is not a general motion prior. A future step is to use ForceSAPIEN to generate force-driven trajectories and train/evaluate a force-conditioned motion prior.

Generated datasets, Gaussian files, videos, rendered frames, and reports are intentionally not committed.
