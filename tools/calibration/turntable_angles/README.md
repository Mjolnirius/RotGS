# Physical turntable-angle calibration

This tool measures turntable motion from unique ArUco markers rigidly attached
to the table. It is independent of RotGS training and never uses the product as
a tracking source.

Run it on the original, calibrated-resolution sequence:

```bash
uv run python tools/calibration/estimate_turntable_angles.py \
  /path/to/images \
  --intrinsics /path/to/camera_calibration.npz \
  --marker-size-mm 40.0 \
  --nominal-step-deg 5.0
```

`--marker-size-mm` is the physical side length of the **black ArUco square**,
not the surrounding sticker or paper. The intrinsics NPZ is the format written
by `tools/calibration/get_camera_intrinsics.py`. Its image dimensions must match
the source images exactly; no intrinsics scaling is inferred.

## Model and coordinates

OpenCV camera coordinates are +X right, +Y down, and +Z forward. At theta=0,
every marker has a learned metric position and in-plane orientation in one
fitted physical reference table plane. For frame `i`, that complete plane and
marker layout rotate rigidly through the independently fitted `theta_i` about
one fixed 3D axis. The final objective is robust image-space reprojection error
over the detected corners. Nominal angles only initialize the rotation sense
and unwrap branch; they are not observations or optimization residuals.

The reported world frame is orthonormal and right-handed:

- origin: fixed-axis/reference-plane intersection;
- Z: fixed physical rotation axis, signed for positive acquisition motion;
- X: camera-to-origin direction projected perpendicular to Z;
- Y: `Z cross X`.

World XY is therefore the ideal rotation plane. The fitted physical table plane
at theta=0 is reported separately and can be slightly tilted relative to World
XY. Its normal at any frame is the theta-rotated reference normal.

## Confidence and uncertainty

`angle_uncertainty_deg` is an approximate local one-sigma value from the final
inlier Jacobian. It assumes exact intrinsics and marker size, independent corner
noise, and a correct rigid fixed-axis model. It does not include lens-calibration,
printing, mounting, table flex, vibration, or rolling-shutter systematics.

Per-frame confidence uses reprojection residual, accepted corner count, spatial
coverage, rejection fraction, local uncertainty, and a loose neighbouring-angle
diagnostic:

- `excellent`: at least 3 markers/12 corners, 2% image coverage, RMS <=0.75 px,
  uncertainty <=0.05 deg, no more than 25% rejected, and no neighbour warning;
- `good`: at least 2 markers/8 corners, 0.5% coverage, RMS <=1.5 px,
  uncertainty <=0.15 deg, no more than 50% rejected, and no neighbour warning;
- `low`: solved, but one or more good thresholds are not met;
- `unresolved`: no direct globally accepted observation or no measured angle.

These labels expose fit quality; they are not a guarantee of 0.1-degree physical
accuracy. Reliable small-step comparison requires sharply focused markers,
accurate lens calibration, rigid mounting, good radial/spatial marker spread,
and several well-resolved markers in each frame.

An optional empty-turntable reference image can connect marker groups hidden by
the product. Its angle is estimated as an auxiliary variable, so it need not be
captured at the frame-0 position.
