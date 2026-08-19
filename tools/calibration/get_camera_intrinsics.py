import cv2
import glob
import sys
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SQUARES_X = 12
SQUARES_Y = 9

SQUARE_LENGTH = 0.030   # 30 mm
MARKER_LENGTH = 0.022   # 22 mm

MIN_CHARUCO_CORNERS = 15


DICTIONARIES = {
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
}


# ============================================================
# LOAD IMAGES
# ============================================================

if len(sys.argv) != 2:
    print(
        "Usage:\n"
        "python detect_dictionary_and_calibrate.py <image_folder>\n\n"
        "Example:\n"
        "python detect_dictionary_and_calibrate.py Calibration_02_PNG"
    )
    sys.exit(1)


image_folder = Path(sys.argv[1])

if not image_folder.exists():
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
print("CHARUCO CAMERA CALIBRATION")
print("=" * 70)

print(f"Folder          : {image_folder.resolve()}")
print(f"Images found    : {len(images)}")
print(f"Board           : {SQUARES_X} x {SQUARES_Y}")
print(f"Square size     : {SQUARE_LENGTH * 1000:.1f} mm")
print(f"Marker size     : {MARKER_LENGTH * 1000:.1f} mm")

print()


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


rms, camera_matrix, dist_coeffs, rvecs, tvecs = \
    cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
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


print()
print(f"Saved calibration to:")
print(output_file.resolve())

print()
print("=" * 70)
print("DONE")
print("=" * 70)