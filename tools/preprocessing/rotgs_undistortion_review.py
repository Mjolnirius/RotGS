"""Browser review for comparing original and undistorted sequence frames."""

from __future__ import annotations

import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, ImageDraw


MAX_REQUEST_BYTES = 16_384
PREVIEW_CACHE_SIZE = 3
PAGE_HTML = Path(__file__).with_name("rotgs_undistortion_review.html").read_text(
    encoding="utf-8"
)


def _display_image(
    image: Image.Image,
    mode: str,
    maximum_size: tuple[int, int],
) -> Image.Image:
    display = image.convert(mode)
    display.thumbnail(maximum_size, Image.Resampling.LANCZOS)
    return display


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _grid_positions(length: int, spacing: int) -> list[int]:
    positions = list(range(0, length, spacing))
    if not positions or positions[-1] != length - 1:
        positions.append(length - 1)
    return positions


def _draw_straight_grid(
    image: Image.Image,
    source_size: tuple[int, int],
    spacing: int,
) -> None:
    source_width, source_height = source_size
    scale_x = (image.width - 1) / max(1, source_width - 1)
    scale_y = (image.height - 1) / max(1, source_height - 1)
    color = (255, 0, 0, 255) if image.mode == "RGBA" else (255, 0, 0)
    draw = ImageDraw.Draw(image)
    for source_x in _grid_positions(source_width, spacing):
        x = round(source_x * scale_x)
        draw.line((x, 0, x, image.height - 1), fill=color, width=1)
    for source_y in _grid_positions(source_height, spacing):
        y = round(source_y * scale_y)
        draw.line((0, y, image.width - 1, y), fill=color, width=1)


def _draw_curve(
    draw: ImageDraw.ImageDraw,
    source_x: np.ndarray,
    source_y: np.ndarray,
    valid: np.ndarray,
    scale_x: float,
    scale_y: float,
    color: tuple[int, ...],
) -> None:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return
    split_points = np.flatnonzero(np.diff(valid_indices) > 1) + 1
    for segment in np.split(valid_indices, split_points):
        if segment.size < 2:
            continue
        points = [
            (
                float(source_x[index] * scale_x),
                float(source_y[index] * scale_y),
            )
            for index in segment
        ]
        draw.line(points, fill=color, width=1)


def _draw_distorted_grid(
    image: Image.Image,
    map_x: np.ndarray,
    map_y: np.ndarray,
    spacing: int,
) -> None:
    source_height, source_width = map_x.shape
    scale_x = (image.width - 1) / max(1, source_width - 1)
    scale_y = (image.height - 1) / max(1, source_height - 1)
    color = (255, 0, 0, 255) if image.mode == "RGBA" else (255, 0, 0)
    draw = ImageDraw.Draw(image)
    sampling_step = max(1, min(source_width, source_height) // 1000)

    sampled_y = np.arange(0, source_height, sampling_step)
    if sampled_y[-1] != source_height - 1:
        sampled_y = np.append(sampled_y, source_height - 1)
    for destination_x in _grid_positions(source_width, spacing):
        source_x = map_x[sampled_y, destination_x]
        source_y = map_y[sampled_y, destination_x]
        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0)
            & (source_x <= source_width - 1)
            & (source_y >= 0)
            & (source_y <= source_height - 1)
        )
        _draw_curve(
            draw,
            source_x,
            source_y,
            valid,
            scale_x,
            scale_y,
            color,
        )

    sampled_x = np.arange(0, source_width, sampling_step)
    if sampled_x[-1] != source_width - 1:
        sampled_x = np.append(sampled_x, source_width - 1)
    for destination_y in _grid_positions(source_height, spacing):
        source_x = map_x[destination_y, sampled_x]
        source_y = map_y[destination_y, sampled_x]
        valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0)
            & (source_x <= source_width - 1)
            & (source_y >= 0)
            & (source_y <= source_height - 1)
        )
        _draw_curve(
            draw,
            source_x,
            source_y,
            valid,
            scale_x,
            scale_y,
            color,
        )


PreviewPair = tuple[bytes, bytes]
PreviewVariants = tuple[PreviewPair, PreviewPair]


@dataclass
class UndistortionReviewState:
    frames: tuple[tuple[int, Path], ...]
    correct_image: Callable[[Image.Image], Image.Image]
    display_mode: str
    maximum_display_size: tuple[int, int]
    map_x: np.ndarray
    map_y: np.ndarray
    context_label: str | None = None
    decision: bool | None = None
    cache: OrderedDict[int, PreviewVariants] = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.map_x.shape != self.map_y.shape or self.map_x.ndim != 2:
            raise ValueError("undistortion coordinate maps must be matching 2D arrays")

    def frame_pair(
        self,
        index: int,
        helper_lines: bool = True,
    ) -> PreviewPair:
        if not 0 <= index < len(self.frames):
            raise IndexError("frame index is outside the sequence")
        with self.lock:
            cached = self.cache.get(index)
            if cached is None:
                cached = self._build_frame_variants(index)
                self.cache[index] = cached
                while len(self.cache) > PREVIEW_CACHE_SIZE:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            return cached[1 if helper_lines else 0]

    def _build_frame_variants(self, index: int) -> PreviewVariants:
        _angle, path = self.frames[index]
        original_display: Image.Image | None = None
        corrected_display: Image.Image | None = None
        original_with_grid: Image.Image | None = None
        corrected_with_grid: Image.Image | None = None
        try:
            try:
                with Image.open(path) as source:
                    source.load()
                    expected_size = self.map_x.shape[1], self.map_x.shape[0]
                    if source.size != expected_size:
                        raise ValueError(
                            f"review image has wrong dimensions: {path.name}"
                        )
                    original_display = _display_image(
                        source,
                        self.display_mode,
                        self.maximum_display_size,
                    )
                    corrected_image = self.correct_image(source)
            except OSError as exc:
                raise ValueError(
                    f"could not load review image: {path.name}"
                ) from exc
            try:
                corrected_display = _display_image(
                    corrected_image,
                    self.display_mode,
                    self.maximum_display_size,
                )
            finally:
                corrected_image.close()
            if original_display.size != corrected_display.size:
                raise ValueError("original and undistorted preview sizes differ")

            original_with_grid = original_display.copy()
            corrected_with_grid = corrected_display.copy()
            spacing = max(64, round(min(self.map_x.shape) / 8))
            _draw_distorted_grid(
                original_with_grid,
                self.map_x,
                self.map_y,
                spacing,
            )
            _draw_straight_grid(
                corrected_with_grid,
                (self.map_x.shape[1], self.map_x.shape[0]),
                spacing,
            )
            return (
                (_png_bytes(original_display), _png_bytes(corrected_display)),
                (_png_bytes(original_with_grid), _png_bytes(corrected_with_grid)),
            )
        finally:
            for image in (
                original_display,
                corrected_display,
                original_with_grid,
                corrected_with_grid,
            ):
                if image is not None:
                    image.close()


class UndistortionReviewServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        state: UndistortionReviewState,
    ) -> None:
        self.state = state
        super().__init__(address, handler)


def _handler_for():
    class UndistortionReviewHandler(BaseHTTPRequestHandler):
        server_version = "RotGSUndistortionReview/1.0"

        @property
        def state(self) -> UndistortionReviewState:
            return self.server.state  # type: ignore[attr-defined, no-any-return]

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            request = urlsplit(self.path)
            if request.path == "/":
                self._send_bytes(
                    PAGE_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            if request.path == "/api/config":
                self._send_json(
                    {
                        "kind": "undistortion-review",
                        "context_label": self.state.context_label,
                        "frames": [
                            {
                                "index": index,
                                "angle": angle,
                                "name": path.name,
                            }
                            for index, (angle, path) in enumerate(self.state.frames)
                        ],
                    }
                )
                return
            if request.path == "/api/frame":
                try:
                    query = parse_qs(request.query, strict_parsing=True)
                    index = int(query["index"][0])
                    kind = query["kind"][0]
                    helpers = query.get("helpers", ["1"])[0] != "0"
                    original, corrected = self.state.frame_pair(index, helpers)
                    if kind == "original":
                        data = original
                    elif kind == "undistorted":
                        data = corrected
                    else:
                        raise ValueError("frame kind must be original or undistorted")
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))
                    return
                self._send_bytes(data, "image/png")
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            request_path = urlsplit(self.path).path
            if request_path != "/api/decision":
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                payload = self._read_json()
                decision = payload.get("continue")
                if not isinstance(decision, bool):
                    raise ValueError("continue must be true or false")
                with self.state.lock:
                    self.state.decision = decision
                self._send_json({"continue": decision})
                threading.Thread(
                    target=self.server.shutdown,
                    daemon=True,
                ).start()
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc))

        def _read_json(self) -> dict[str, object]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _send_json(self, payload: object) -> None:
            self._send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            data = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status)
            self._headers("application/json; charset=utf-8", len(data))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self._headers(content_type, len(data))
            self.end_headers()
            self.wfile.write(data)

        def _headers(self, content_type: str, length: int) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src 'self'; connect-src 'self'; "
                "style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )

    return UndistortionReviewHandler


def review_undistortion_web(
    frames: list[tuple[int, Path]],
    correct_image: Callable[[Image.Image], Image.Image],
    display_max_width: int,
    display_max_height: int,
    preserve_alpha: bool,
    *,
    destination_to_source_maps: tuple[np.ndarray, np.ndarray],
    port: int,
    context_label: str | None = None,
) -> bool:
    """Block until the user accepts or rejects browser-reviewed correction."""
    if not frames:
        raise ValueError("at least one frame is required for undistortion review")
    if not 1 <= port <= 65535:
        raise ValueError("web port must be between 1 and 65535")
    if display_max_width <= 0 or display_max_height <= 0:
        raise ValueError("display dimensions must be positive")

    tile_width = max(1, display_max_width // 2)
    state = UndistortionReviewState(
        frames=tuple(frames),
        correct_image=correct_image,
        display_mode="RGBA" if preserve_alpha else "RGB",
        maximum_display_size=(tile_width, display_max_height),
        map_x=destination_to_source_maps[0],
        map_y=destination_to_source_maps[1],
        context_label=context_label,
    )
    print("Preparing first original/undistorted comparison...", flush=True)
    state.frame_pair(0)
    server = UndistortionReviewServer(
        ("127.0.0.1", port),
        _handler_for(),
        state,
    )
    print("WARNING: lens undistortion changes image geometry and intrinsics.", flush=True)
    print(f"Undistortion review listening on: http://127.0.0.1:{port}/", flush=True)
    print(f"In VS Code, forward port {port}, then open: http://localhost:{port}/", flush=True)
    print("Review the sequence and answer Yes or No in the browser.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return False
    finally:
        server.server_close()
    return state.decision is True
