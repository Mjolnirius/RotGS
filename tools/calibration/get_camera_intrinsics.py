import argparse

import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SQUARES_X = 9
SQUARES_Y = 12

SQUARE_LENGTH = 0.030   # 30 mm
MARKER_LENGTH = 0.022   # 22 mm

MIN_CHARUCO_CORNERS = 15

CHECKERBOARD_SQUARES_X = 9
CHECKERBOARD_SQUARES_Y = 14
CHECKERBOARD_SQUARE_LENGTH = 0.020  # 20 mm


DICTIONARIES = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}


def run_checkerboard_calibration(
    image_folder,
    images,
    squares_x,
    squares_y,
    square_length,
    max_detection_width,
):
    """Calibrate from a plain checkerboard described by physical square counts."""
    # OpenCV expects internal corners, which are one fewer than squares per axis.
    pattern_size = (squares_x - 1, squares_y - 1)
    object_template = np.zeros(
        (pattern_size[0] * pattern_size[1], 3),
        dtype=np.float32,
    )
    object_template[:, :2] = (
        np.mgrid[0:pattern_size[0], 0:pattern_size[1]]
        .T.reshape(-1, 2)
        * square_length
    )

    detection_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    refinement_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )

    all_object_points = []
    all_image_points = []
    accepted_images = []
    rejected_images = []
    image_size = None

    print("=" * 70)
    print("STEP 1 — DETECTING CHECKERBOARD CORNERS")
    print("=" * 70)

    for image_path in images:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            rejected_images.append((image_path.name, "could not read image"))
            continue

        current_size = gray.shape[::-1]

        if image_size is None:
            image_size = current_size
            print(
                f"\nImage resolution: "
                f"{image_size[0]} x {image_size[1]}"
            )
        elif current_size != image_size:
            rejected_images.append(
                (
                    image_path.name,
                    f"wrong resolution "
                    f"{current_size[0]}x{current_size[1]}",
                )
            )
            continue

        scale = min(1.0, max_detection_width / current_size[0])

        if scale < 1.0:
            detection_image = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = gray

        found, corners = cv2.findChessboardCornersSB(
            detection_image,
            pattern_size,
            flags=detection_flags,
        )

        if not found:
            rejected_images.append(
                (image_path.name, "checkerboard not fully detected")
            )
            continue

        corners = corners.astype(np.float32)

        if scale < 1.0:
            corners /= scale

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            refinement_criteria,
        )

        all_object_points.append(object_template.copy())
        all_image_points.append(corners)
        accepted_images.append(image_path.name)

        print(
            f"OK   {image_path.name:<35} "
            f"{len(corners):>3} corners"
        )

    print()
    print("-" * 70)
    print(f"Accepted images : {len(accepted_images)} / {len(images)}")
    print(f"Rejected images : {len(rejected_images)}")

    if rejected_images:
        print("\nRejected:")

        for filename, reason in rejected_images:
            print(f"  {filename:<35} {reason}")

    if len(all_image_points) < 5:
        raise RuntimeError(
            "\nToo few valid calibration images.\n"
            "At least 5 are required, but preferably 20-40 "
            "with varied board positions and orientations."
        )

    print()
    print("=" * 70)
    print("STEP 2 — CAMERA CALIBRATION")
    print("=" * 70)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_object_points,
        imagePoints=all_image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    per_view_errors = []

    for object_points, image_points, rvec, tvec in zip(
        all_object_points,
        all_image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        residual = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
        per_view_errors.append(
            float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        )

    per_view_errors = np.asarray(per_view_errors)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print()
    print("=" * 70)
    print("FINAL CAMERA INTRINSICS")
    print("=" * 70)
    print(f"\nResolution       : {image_size[0]} x {image_size[1]}")
    print(f"Calibration imgs : {len(all_image_points)}")
    print(f"\nfx = {fx:.6f} px")
    print(f"fy = {fy:.6f} px")
    print(f"cx = {cx:.6f} px")
    print(f"cy = {cy:.6f} px")
    print("\nCamera matrix K:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    print(f"\nRMS reprojection error: {rms:.6f} px")
    print(
        f"Per-view RMS range: {per_view_errors.min():.6f} .. "
        f"{per_view_errors.max():.6f} px"
    )

    output_file = image_folder / "camera_calibration.npz"
    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        image_size=np.array(image_size),
        board_type="checkerboard",
        squares_x=squares_x,
        squares_y=squares_y,
        pattern_size=np.array(pattern_size),
        square_length=square_length,
        accepted_images=np.array(accepted_images),
        per_view_errors=per_view_errors,
    )

    cameras_file = image_folder.parent / "cameras.txt"
    cameras_contents = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {image_size[0]} {image_size[1]} "
        f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n"
    )

    with cameras_file.open("w", encoding="utf-8", newline="\n") as file:
        file.write(cameras_contents)

    print(f"\nSaved calibration to:\n{output_file.resolve()}")
    print(f"\nSaved COLMAP camera to:\n{cameras_file.resolve()}")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# LOAD IMAGES
# ============================================================

parser = argparse.ArgumentParser(
    description="Calibrate a camera from ChArUco or checkerboard images."
)
parser.add_argument("image_folder", type=Path)
parser.add_argument(
    "--board",
    choices=("charuco", "checkerboard"),
    default="charuco",
    help="calibration target type (default: charuco)",
)
parser.add_argument(
    "--squares-x",
    type=int,
    help="number of physical board squares across",
)
parser.add_argument(
    "--squares-y",
    type=int,
    help="number of physical board squares down",
)
parser.add_argument(
    "--square-size-mm",
    type=float,
    help="physical square side length in millimetres",
)
parser.add_argument(
    "--marker-size-mm",
    type=float,
    help="ChArUco marker side length in millimetres",
)
parser.add_argument(
    "--max-detection-width",
    type=int,
    default=2400,
    help=(
        "downsample checkerboard images to at most this width for detection; "
        "corners are refined at full resolution (default: 2400)"
    ),
)
args = parser.parse_args()

if args.board == "checkerboard":
    SQUARES_X = (
        args.squares_x
        if args.squares_x is not None
        else CHECKERBOARD_SQUARES_X
    )
    SQUARES_Y = (
        args.squares_y
        if args.squares_y is not None
        else CHECKERBOARD_SQUARES_Y
    )
    SQUARE_LENGTH = (
        args.square_size_mm / 1000.0
        if args.square_size_mm is not None
        else CHECKERBOARD_SQUARE_LENGTH
    )
else:
    SQUARES_X = args.squares_x if args.squares_x is not None else SQUARES_X
    SQUARES_Y = args.squares_y if args.squares_y is not None else SQUARES_Y
    SQUARE_LENGTH = (
        args.square_size_mm / 1000.0
        if args.square_size_mm is not None
        else SQUARE_LENGTH
    )
    MARKER_LENGTH = (
        args.marker_size_mm / 1000.0
        if args.marker_size_mm is not None
        else MARKER_LENGTH
    )

if SQUARES_X < 3 or SQUARES_Y < 3:
    parser.error("--squares-x and --squares-y must both be at least 3")
if SQUARE_LENGTH <= 0:
    parser.error("--square-size-mm must be positive")
if args.max_detection_width <= 0:
    parser.error("--max-detection-width must be positive")
if args.board == "charuco" and not 0 < MARKER_LENGTH < SQUARE_LENGTH:
    parser.error("--marker-size-mm must be positive and smaller than the square")

image_folder = args.image_folder

if not image_folder.is_dir():
    raise FileNotFoundError(
        f"Folder does not exist: {image_folder}"
    )


extensions = [
    "*.png",
    "*.PNG",
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
]

images = []

for ext in extensions:
    images.extend(image_folder.glob(ext))

images = sorted(set(images))


if not images:
    raise RuntimeError(
        f"No supported images found in: {image_folder}"
    )


print()
print("=" * 70)
print(f"{args.board.upper()} CAMERA CALIBRATION")
print("=" * 70)

print(f"Folder          : {image_folder.resolve()}")
print(f"Images found    : {len(images)}")
print(f"Physical squares: {SQUARES_X} x {SQUARES_Y}")
if args.board == "checkerboard":
    print(f"Internal corners: {SQUARES_X - 1} x {SQUARES_Y - 1}")
print(f"Square size     : {SQUARE_LENGTH * 1000:.1f} mm")
if args.board == "charuco":
    print(f"Marker size     : {MARKER_LENGTH * 1000:.1f} mm")

print()

if args.board == "checkerboard":
    run_checkerboard_calibration(
        image_folder,
        images,
        SQUARES_X,
        SQUARES_Y,
        SQUARE_LENGTH,
        args.max_detection_width,
    )
    raise SystemExit(0)


# ============================================================
# STEP 1: DETECT BEST ARUCO DICTIONARY
# ============================================================

print("=" * 70)
print("STEP 1 — TESTING ARUCO DICTIONARIES")
print("=" * 70)

dictionary_results = {}


for dict_name, dict_id in DICTIONARIES.items():

    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    detector_params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        detector_params
    )

    total_markers = 0
    successful_images = 0
    unique_ids = set()

    print()
    print(f"Testing {dict_name}...")

    for image_path in images:

        img = cv2.imread(str(image_path))

        if img is None:
            continue

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        marker_corners, marker_ids, rejected = \
            detector.detectMarkers(gray)

        if marker_ids is not None:

            ids_flat = marker_ids.flatten()

            total_markers += len(ids_flat)

            successful_images += 1

            unique_ids.update(
                ids_flat.tolist()
            )


    dictionary_results[dict_name] = {
        "dict_id": dict_id,
        "total_markers": total_markers,
        "successful_images": successful_images,
        "unique_ids": unique_ids,
    }


    print(f"  Markers detected : {total_markers}")

    print(
        f"  Images detected  : "
        f"{successful_images}/{len(images)}"
    )

    print(
        f"  Unique IDs       : "
        f"{len(unique_ids)}"
    )

    if unique_ids:
        print(
            f"  ID range         : "
            f"{min(unique_ids)} ... {max(unique_ids)}"
        )


# ============================================================
# RANK DICTIONARIES
# ============================================================

ranking = sorted(
    dictionary_results.items(),
    key=lambda x: (
        x[1]["total_markers"],
        x[1]["successful_images"],
    ),
    reverse=True
)


print()
print("=" * 70)
print("DICTIONARY RANKING")
print("=" * 70)

for rank, (name, stats) in enumerate(
    ranking,
    start=1
):

    print(
        f"#{rank} {name:<18} "
        f"markers={stats['total_markers']:<5} "
        f"images={stats['successful_images']}/{len(images):<5} "
        f"unique_ids={len(stats['unique_ids'])}"
    )


best_name, best_stats = ranking[0]
best_dict_id = best_stats["dict_id"]


print()
print("=" * 70)
print(f"BEST DICTIONARY: {best_name}")
print("=" * 70)


# ============================================================
# STEP 2: BUILD CHARUCO BOARD
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    best_dict_id
)

board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    dictionary
)

board.setLegacyPattern(True)


detector_params = cv2.aruco.DetectorParameters()

charuco_params = cv2.aruco.CharucoParameters()

charuco_detector = cv2.aruco.CharucoDetector(
    board,
    charuco_params,
    detector_params
)


# ============================================================
# STEP 3: DETECT CHARUCO CORNERS
# ============================================================

print()
print("=" * 70)
print("STEP 2 — DETECTING CHARUCO CORNERS")
print("=" * 70)


all_charuco_corners = []
all_charuco_ids = []

accepted_images = []
rejected_images = []

image_size = None


for image_path in images:

    img = cv2.imread(str(image_path))

    if img is None:
        rejected_images.append(
            (image_path.name, "could not read image")
        )

        continue


    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    current_size = gray.shape[::-1]


    if image_size is None:

        image_size = current_size

        print(
            f"\nImage resolution: "
            f"{image_size[0]} x {image_size[1]}"
        )


    elif current_size != image_size:

        rejected_images.append(
            (
                image_path.name,
                f"wrong resolution "
                f"{current_size[0]}x{current_size[1]}"
            )
        )

        continue


    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ) = charuco_detector.detectBoard(gray)


    if charuco_ids is None:

        rejected_images.append(
            (image_path.name, "no ChArUco corners detected")
        )

        continue


    num_corners = len(charuco_ids)


    if num_corners < MIN_CHARUCO_CORNERS:

        rejected_images.append(
            (
                image_path.name,
                f"only {num_corners} ChArUco corners"
            )
        )

        continue


    all_charuco_corners.append(
        charuco_corners
    )

    all_charuco_ids.append(
        charuco_ids
    )

    accepted_images.append(
        (
            image_path.name,
            num_corners
        )
    )


    print(
        f"OK   {image_path.name:<35} "
        f"{num_corners:>3} corners"
    )


# ============================================================
# CHECK ENOUGH IMAGES
# ============================================================

print()
print("-" * 70)

print(
    f"Accepted images : "
    f"{len(accepted_images)} / {len(images)}"
)

print(
    f"Rejected images : "
    f"{len(rejected_images)}"
)


if rejected_images:

    print()
    print("Rejected:")

    for filename, reason in rejected_images:

        print(
            f"  {filename:<35} {reason}"
        )


if len(all_charuco_corners) < 5:

    raise RuntimeError(
        "\nToo few valid calibration images.\n"
        "At least 5 are required, but preferably 20-40 "
        "with varied board positions and orientations."
    )


# ============================================================
# STEP 4: CAMERA CALIBRATION
# ============================================================

print()
print("=" * 70)
print("STEP 3 — CAMERA CALIBRATION")
print("=" * 70)


if hasattr(cv2.aruco, "calibrateCameraCharuco"):

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

else:

    all_object_points = []
    all_image_points = []

    for charuco_corners, charuco_ids in zip(
        all_charuco_corners,
        all_charuco_ids,
    ):

        object_points, image_points = \
            board.matchImagePoints(
                charuco_corners,
                charuco_ids,
            )

        all_object_points.append(object_points)
        all_image_points.append(image_points)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.calibrateCamera(
            objectPoints=all_object_points,
            imagePoints=all_image_points,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )


# ============================================================
# PRINT RESULTS
# ============================================================

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]

cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]


print()
print("=" * 70)
print("FINAL CAMERA INTRINSICS")
print("=" * 70)

print()
print(f"Dictionary       : {best_name}")

print(
    f"Resolution       : "
    f"{image_size[0]} x {image_size[1]}"
)

print(
    f"Calibration imgs : "
    f"{len(all_charuco_corners)}"
)

print()

print(
    f"fx = {fx:.6f} px"
)

print(
    f"fy = {fy:.6f} px"
)

print(
    f"cx = {cx:.6f} px"
)

print(
    f"cy = {cy:.6f} px"
)


print()
print("Camera matrix K:")

print(camera_matrix)


print()
print("Distortion coefficients:")

print(dist_coeffs)


print()
print(
    f"RMS reprojection error: "
    f"{rms:.6f} px"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = image_folder / "camera_calibration.npz"


np.savez(
    output_file,

    camera_matrix=camera_matrix,

    dist_coeffs=dist_coeffs,

    rms=rms,

    image_size=np.array(image_size),

    dictionary=best_name,

    square_length=SQUARE_LENGTH,

    marker_length=MARKER_LENGTH,
)


cameras_file = image_folder.parent / "cameras.txt"

cameras_contents = (
    "# Camera list with one line of data per camera:\n"
    "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
    "# Number of cameras: 1\n"
    f"1 PINHOLE {image_size[0]} {image_size[1]} "
    f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n"
)

with cameras_file.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as file:

    file.write(cameras_contents)


print()
print(f"Saved calibration to:")
print(output_file.resolve())

print()
print(f"Saved COLMAP camera to:")
print(cameras_file.resolve())

print()
print("=" * 70)
print("DONE")
print("=" * 70)
