"""Validated discovery helpers for RotGS multi-camera datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypeVar


METADATA_FILENAME = "multi_camera_metadata.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CameraT = TypeVar("CameraT")


@dataclass(frozen=True)
class MultiCameraPass:
    """One physical camera pass in semantic camera-index order."""

    camera_index: int
    directory_name: str
    path: Path
    rough_elevation_degrees: float | None


@dataclass(frozen=True)
class MultiCameraBundle:
    """Validated top-level multi-camera dataset description."""

    root: Path
    metadata_path: Path | None
    passes: tuple[MultiCameraPass, ...]
    metadata: dict[str, Any] | None

    @property
    def rough_elevations_degrees(self) -> tuple[float, ...] | None:
        elevations = tuple(item.rough_elevation_degrees for item in self.passes)
        if any(value is None for value in elevations):
            return None
        return tuple(float(value) for value in elevations)


def _is_camera_directory(path: Path) -> bool:
    if not path.is_dir() or not (path / "images").is_dir():
        return False
    sparse = path / "sparse" / "0"
    return (sparse / "cameras.txt").is_file() or (
        sparse / "cameras.bin"
    ).is_file()


def _validate_camera_directory(path: Path, label: str) -> None:
    if not _is_camera_directory(path):
        raise ValueError(
            f"{label} is not a RotGS camera directory with images/ and "
            f"sparse/0/cameras.txt or cameras.bin: {path}"
        )
    if not any(
        item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        for item in (path / "images").iterdir()
    ):
        raise ValueError(f"{label} contains no supported images: {path / 'images'}")


def _legacy_bundle(root: Path) -> MultiCameraBundle:
    camera_paths = sorted(
        (item for item in root.iterdir() if _is_camera_directory(item)),
        key=lambda item: (item.name.casefold(), item.name),
    )
    if len(camera_paths) < 2:
        raise ValueError(
            "multi-camera source must contain at least two camera directories"
        )
    passes = tuple(
        MultiCameraPass(index, path.name, path, None)
        for index, path in enumerate(camera_paths)
    )
    return MultiCameraBundle(root, None, passes, None)


def load_multi_camera_bundle(source_path: str | Path) -> MultiCameraBundle:
    """Load new metadata bundles while retaining legacy directory discovery."""
    root = Path(source_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"multi-camera source is not a directory: {root}")

    metadata_path = root / METADATA_FILENAME
    if not metadata_path.is_file():
        return _legacy_bundle(root)

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {metadata_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    if payload.get("dataset_type") != "rotgs_multi_camera":
        raise ValueError(
            f"{metadata_path} has unsupported dataset_type "
            f"{payload.get('dataset_type')!r}"
        )

    camera_payloads = payload.get("cameras")
    if not isinstance(camera_payloads, list) or len(camera_payloads) < 2:
        raise ValueError(
            f"{metadata_path} must describe at least two camera passes"
        )
    declared_count = payload.get("camera_count")
    if declared_count != len(camera_payloads):
        raise ValueError(
            f"{metadata_path} camera_count={declared_count!r} does not match "
            f"its {len(camera_payloads)} camera entries"
        )

    parsed: list[MultiCameraPass] = []
    directory_names: set[str] = set()
    for entry in camera_payloads:
        if not isinstance(entry, dict):
            raise ValueError(f"{metadata_path} contains a non-object camera entry")
        camera_index = entry.get("camera_index")
        if not isinstance(camera_index, int) or isinstance(camera_index, bool):
            raise ValueError("every camera entry needs an integer camera_index")
        directory_name = entry.get("directory")
        if (
            not isinstance(directory_name, str)
            or not directory_name
            or Path(directory_name).name != directory_name
        ):
            raise ValueError(
                "every camera directory must be one safe top-level directory name"
            )
        if directory_name in directory_names:
            raise ValueError(f"duplicate camera directory: {directory_name}")
        directory_names.add(directory_name)

        elevation = entry.get("rough_elevation_degrees")
        if (
            not isinstance(elevation, (int, float))
            or isinstance(elevation, bool)
            or not math.isfinite(float(elevation))
        ):
            raise ValueError(
                f"camera {camera_index} has an invalid rough elevation: {elevation!r}"
            )
        camera_path = root / directory_name
        _validate_camera_directory(camera_path, f"camera {camera_index}")
        parsed.append(
            MultiCameraPass(
                camera_index=camera_index,
                directory_name=directory_name,
                path=camera_path,
                rough_elevation_degrees=float(elevation),
            )
        )

    parsed.sort(key=lambda item: item.camera_index)
    expected_indices = list(range(len(parsed)))
    actual_indices = [item.camera_index for item in parsed]
    if actual_indices != expected_indices:
        raise ValueError(
            "camera_index values must be contiguous and start at zero; "
            f"found {actual_indices}"
        )

    top_elevations = payload.get("rough_camera_elevations_degrees")
    parsed_elevations = [
        float(item.rough_elevation_degrees) for item in parsed
    ]
    if (
        not isinstance(top_elevations, list)
        or len(top_elevations) != len(parsed_elevations)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in top_elevations
        )
        or any(
            not math.isclose(float(declared), parsed_value, abs_tol=1e-9)
            for declared, parsed_value in zip(top_elevations, parsed_elevations)
        )
    ):
        raise ValueError(
            "top-level rough_camera_elevations_degrees must match camera entries"
        )

    return MultiCameraBundle(
        root=root,
        metadata_path=metadata_path,
        passes=tuple(parsed),
        metadata=payload,
    )


def axis_vectors_from_elevations(
    elevations_degrees: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Map camera elevation hints to turntable axes in camera coordinates."""
    vectors: list[tuple[float, float, float]] = []
    for elevation in elevations_degrees:
        value = float(elevation)
        if not math.isfinite(value):
            raise ValueError("camera elevations must be finite")
        radians = math.radians(value)
        vectors.append((0.0, math.cos(radians), math.sin(radians)))
    return tuple(vectors)


def group_cameras_by_index(
    cameras: Sequence[CameraT], number_of_cameras: int
) -> list[list[CameraT]]:
    """Group loaded views without relying on list order or equal-size chunks."""
    if number_of_cameras < 1:
        raise ValueError("number_of_cameras must be positive")

    grouped: list[list[CameraT]] = [[] for _ in range(number_of_cameras)]
    for camera in cameras:
        camera_index = getattr(camera, "cam_idx", None)
        if (
            not isinstance(camera_index, int)
            or isinstance(camera_index, bool)
            or not 0 <= camera_index < number_of_cameras
        ):
            raise ValueError(
                f"camera has invalid cam_idx {camera_index!r}; expected an integer "
                f"in [0, {number_of_cameras - 1}]"
            )
        grouped[camera_index].append(camera)

    missing = [index for index, group in enumerate(grouped) if not group]
    if missing:
        raise ValueError(f"no training views found for camera indices {missing}")
    return grouped
