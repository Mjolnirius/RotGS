# Multi-camera integration

The repository intentionally keeps the training flows separate:

- `main` owns single-camera training in `train.py`.
- `multi_cam_integration` owns multi-camera training in `train_multi.py`.
- Shared infrastructure and correctness fixes should be ported explicitly rather
  than merging the trainers.

## Commit labels

Use focused commits with these prefixes:

- `[shared]` for changes that should be cherry-picked between branches.
- `[main]` for single-camera-only behavior.
- `[multi]` for multi-camera-only behavior.

Do not mix `train.py` and `train_multi.py` changes in one commit.

## Current shared infrastructure

The multi-camera branch uses the shared optional W&B adapter and adds
multi-camera metrics, pre-validation recovery snapshots, and validation memory
cleanup. It also has a separate queue in `jobs/train_multi_queue.sh`.

The Gaussian densification cap is already implemented in the multi-camera
trainer and model; no port from main is pending.

## Deferred work

### Foreground-aware corrected evaluation

Adapt the reusable foreground metric implementation from main for arbitrary
multi-camera datasets. The existing driver on main is intentionally not copied:
it is tied to 12 single-camera runs, 73 source images, and 10 held-out views.
The multi-camera driver should discover cameras and views from dataset metadata,
report macro and per-camera metrics, and include the worst camera.

### Multi-camera orbit report

The current report on main is not just a PLY viewer. It builds a single-camera
`Scene`, loads camera 0 axis/center state, and synthesizes trajectories from
that calibrated camera. Supporting a multi-camera splat requires:

1. Loading the multi-camera bundle and constructing `GaussianModel` with the
   saved camera count and rigid-transform settings.
2. Loading per-camera axes, centers, depth, phase, sweep, and residual weights.
3. Adding a camera selector (and optionally rendering every camera).
4. Using the selected camera's intrinsics and learned motion when generating
   trajectories.
5. Extending the manifest and HTML with camera identity and calibration data.

This should be implemented and tested as a focused `[multi]` inspection
change, not folded into training instrumentation.
