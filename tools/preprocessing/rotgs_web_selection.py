"""Local browser UI for RotGS product selection and mask preview."""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

if __package__:
    from .prepare_rotgs_sequence import (
        CropBox,
        SelectionCancelled,
        _relative_product_box,
        crop_box_for_selection,
        segment_product_alpha,
    )
else:
    from prepare_rotgs_sequence import (
        CropBox,
        SelectionCancelled,
        _relative_product_box,
        crop_box_for_selection,
        segment_product_alpha,
    )


DEFAULT_WEB_PORT = 8765
MAX_REQUEST_BYTES = 16_384


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RotGS product selection</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #101218; color: #edf0f7; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    .app { min-height: 100vh; display: flex; flex-direction: column; }
    header { padding: 12px 16px; border-bottom: 1px solid #30343e; background: #191c23; }
    h1 { margin: 0 0 4px; font-size: 18px; }
    #instruction, #status { color: #b9c1d0; font-size: 14px; }
    main { flex: 1; min-height: 0; padding: 12px; display: grid; place-items: center; }
    .viewer { position: relative; max-width: 100%; max-height: calc(100vh - 150px); overflow: auto; border: 1px solid #3b414f; border-radius: 6px; background-color: #d1d1d1; background-image: linear-gradient(45deg, #999 25%, transparent 25%), linear-gradient(-45deg, #999 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #999 75%), linear-gradient(-45deg, transparent 75%, #999 75%); background-position: 0 0, 0 12px, 12px -12px, -12px 0; background-size: 24px 24px; }
    canvas { display: block; max-width: 100%; height: auto; cursor: crosshair; touch-action: none; }
    #preview { display: block; max-width: 100%; max-height: calc(100vh - 150px); object-fit: contain; }
    footer { display: flex; align-items: center; justify-content: center; gap: 9px; min-height: 66px; padding: 10px 16px; border-top: 1px solid #30343e; background: #191c23; flex-wrap: wrap; }
    button { border: 1px solid #465064; border-radius: 7px; padding: 8px 13px; background: #282e3a; color: inherit; font: inherit; cursor: pointer; }
    button:hover { background: #343d4c; }
    button.primary { background: #245ca8; border-color: #4386dd; }
    button.danger { background: #59252b; border-color: #804049; }
    button:disabled { opacity: .45; cursor: default; }
    #threshold-wrap { display: inline-flex; align-items: center; gap: 8px; }
    #threshold { min-width: 62px; text-align: center; font-variant-numeric: tabular-nums; }
    [hidden] { display: none !important; }
  </style>
</head>
<body>
<div class="app">
  <header><h1>RotGS product selection</h1><div id="instruction">Loading first image…</div><div id="status"></div></header>
  <main><div class="viewer"><canvas id="canvas" hidden></canvas><img id="preview" alt="Segmentation preview" hidden></div></main>
  <footer>
    <button id="reselect" hidden>Reselect ROI (R)</button>
    <span id="threshold-wrap" hidden><button id="include">[ Include more</button><span id="threshold"></span><button id="exclude">] Include less</button></span>
    <button class="primary" id="continue" disabled>Continue</button>
    <button class="danger" id="cancel">Cancel (Esc)</button>
  </footer>
</div>
<script>
"use strict";
const canvas = document.getElementById("canvas"), ctx = canvas.getContext("2d");
const preview = document.getElementById("preview"), instruction = document.getElementById("instruction"), status = document.getElementById("status");
const continueButton = document.getElementById("continue"), reselectButton = document.getElementById("reselect"), thresholdWrap = document.getElementById("threshold-wrap"), thresholdText = document.getElementById("threshold");
const source = new Image();
let config = null, selection = null, dragStart = null, stage = "select", threshold = 12, busy = false;

function setBusy(value, message = "") {
  busy = value; status.textContent = message;
  continueButton.disabled = value || (stage === "select" && !selection);
  document.getElementById("include").disabled = value; document.getElementById("exclude").disabled = value; reselectButton.disabled = value;
}
function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)), y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)) };
}
function normalizedBox(a, b) {
  return { left: Math.floor(Math.min(a.x, b.x) * config.source_width / canvas.width), top: Math.floor(Math.min(a.y, b.y) * config.source_height / canvas.height), right: Math.ceil(Math.max(a.x, b.x) * config.source_width / canvas.width), bottom: Math.ceil(Math.max(a.y, b.y) * config.source_height / canvas.height) };
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  if (!selection) return;
  const left = selection.left * canvas.width / config.source_width, top = selection.top * canvas.height / config.source_height, right = selection.right * canvas.width / config.source_width, bottom = selection.bottom * canvas.height / config.source_height;
  ctx.save(); ctx.fillStyle = "rgba(20,105,255,.16)"; ctx.strokeStyle = "#ff344d"; ctx.lineWidth = Math.max(2, canvas.width / 700); ctx.fillRect(left, top, right-left, bottom-top); ctx.strokeRect(left, top, right-left, bottom-top); ctx.restore();
}
function showSelection() {
  stage = "select"; canvas.hidden = false; preview.hidden = true; reselectButton.hidden = true; thresholdWrap.hidden = true;
  continueButton.textContent = config.overwrite_alpha_mask ? "Preview mask" : "Use selection";
  instruction.textContent = "Drag a tight rectangle around the product, then continue."; draw(); setBusy(false);
}
function showPreview() {
  stage = "preview"; canvas.hidden = true; preview.hidden = false; reselectButton.hidden = false; thresholdWrap.hidden = false;
  thresholdText.textContent = `Threshold ${threshold}`; continueButton.textContent = "Accept and process";
  instruction.textContent = "Left: foreground on checkerboard. Right: generated alpha mask."; setBusy(false);
}
async function post(path, payload = {}) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const data = await response.json(); if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); return data;
}
async function requestPreview(nextThreshold = threshold) {
  if (!selection || busy) return; setBusy(true, "Generating segmentation preview…");
  try { const data = await post("/api/preview", { selection, threshold: nextThreshold }); threshold = data.threshold; preview.src = `/preview.png?v=${data.version}`; await preview.decode(); showPreview(); }
  catch (error) { setBusy(false, error.message); }
}
async function accept() {
  if (busy) return; if (stage === "select" && config.overwrite_alpha_mask) { await requestPreview(); return; }
  setBusy(true, "Accepting selection…");
  try { await post("/api/accept", stage === "select" ? { selection } : {}); instruction.textContent = "Accepted. Processing continues in the terminal."; status.textContent = "This page can be closed."; continueButton.hidden = true; reselectButton.hidden = true; thresholdWrap.hidden = true; }
  catch (error) { setBusy(false, error.message); }
}
canvas.addEventListener("pointerdown", event => { if (busy || stage !== "select" || event.button !== 0) return; canvas.setPointerCapture(event.pointerId); dragStart = canvasPoint(event); selection = normalizedBox(dragStart, dragStart); draw(); });
canvas.addEventListener("pointermove", event => { if (!dragStart || busy) return; selection = normalizedBox(dragStart, canvasPoint(event)); draw(); });
function finishDrag(event) { if (!dragStart) return; selection = normalizedBox(dragStart, canvasPoint(event)); dragStart = null; if (selection.right <= selection.left || selection.bottom <= selection.top) selection = null; draw(); setBusy(false); }
canvas.addEventListener("pointerup", finishDrag); canvas.addEventListener("pointercancel", () => { dragStart = null; });
continueButton.addEventListener("click", () => void accept()); reselectButton.addEventListener("click", showSelection);
document.getElementById("include").addEventListener("click", () => void requestPreview(Math.max(1, threshold - 2)));
document.getElementById("exclude").addEventListener("click", () => void requestPreview(threshold + 2));
document.getElementById("cancel").addEventListener("click", async () => { if (busy) return; setBusy(true, "Cancelling…"); try { await post("/api/cancel"); instruction.textContent = "Cancelled."; status.textContent = "Return to the terminal."; } catch (error) { setBusy(false, error.message); } });
window.addEventListener("keydown", event => {
  if (event.key === "Escape") { event.preventDefault(); document.getElementById("cancel").click(); }
  else if ((event.key === "Enter" || event.key === " ") && !continueButton.hidden) { event.preventDefault(); continueButton.click(); }
  else if ((event.key === "r" || event.key === "R") && stage === "preview") { event.preventDefault(); showSelection(); }
  else if (event.key === "[" && stage === "preview") { event.preventDefault(); document.getElementById("include").click(); }
  else if (event.key === "]" && stage === "preview") { event.preventDefault(); document.getElementById("exclude").click(); }
});
fetch("/api/config").then(response => response.json().then(data => ({ response, data }))).then(({ response, data }) => {
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); config = data; threshold = data.threshold; selection = data.initial_selection;
  canvas.width = data.display_width; canvas.height = data.display_height; source.src = "/source.png"; source.onload = showSelection; source.onerror = () => { status.textContent = "Could not load source image."; };
}).catch(error => { status.textContent = error.message; });
</script>
</body>
</html>
"""

# Keep the browser client separate from the local HTTP/image implementation.
PAGE_HTML = Path(__file__).with_name("rotgs_web_selection.html").read_text(
    encoding="utf-8"
)


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _display_source(
    image: Image.Image,
    overwrite_alpha_mask: bool,
    maximum_size: tuple[int, int],
) -> tuple[bytes, tuple[int, int]]:
    display = image.convert("RGB" if overwrite_alpha_mask else "RGBA")
    try:
        display.thumbnail(maximum_size, Image.Resampling.LANCZOS)
        return _png_bytes(display), display.size
    finally:
        display.close()

def _display_comparison_sources(
    comparison_images: list[tuple[int, Path]] | None,
    overwrite_alpha_mask: bool,
    maximum_size: tuple[int, int],
    source_size: tuple[int, int],
) -> tuple[tuple[int, bytes, tuple[int, int]], ...]:
    tile_size = (max(1, maximum_size[0] // 3), max(1, maximum_size[1] // 2))
    if comparison_images is None:
        raise ValueError("six comparison images are required for crop review")
    views: list[tuple[int, bytes, tuple[int, int]]] = []
    for angle, path in comparison_images:
        try:
            with Image.open(path) as comparison:
                comparison.load()
                if comparison.size != source_size:
                    raise ValueError(f"comparison image has wrong dimensions: {path.name}")
                data, size = _display_source(
                    comparison,
                    overwrite_alpha_mask,
                    tile_size,
                )
        except OSError as exc:
            raise ValueError(f"could not load comparison image: {path.name}") from exc
        views.append((angle, data, size))
    return tuple(views)


def _segmentation_preview_bytes(
    cropped: Image.Image,
    alpha: Image.Image,
    maximum_size: tuple[int, int],
) -> bytes:
    rgb = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
    alpha_array = np.asarray(alpha, dtype=np.uint8)
    checker = np.full_like(rgb, 224)
    tile = max(8, min(cropped.size) // 40)
    yy, xx = np.indices(alpha_array.shape)
    checker[(xx // tile + yy // tile) % 2 == 0] = 176
    weight = alpha_array.astype(np.float32)[..., None] / 255.0
    composited = np.clip(rgb * weight + checker * (1.0 - weight), 0, 255).astype(np.uint8)
    mask_rgb = np.repeat(alpha_array[..., None], 3, axis=2)
    combined = Image.fromarray(np.concatenate((composited, mask_rgb), axis=1))
    try:
        combined.thumbnail(maximum_size, Image.Resampling.LANCZOS)
        return _png_bytes(combined)
    finally:
        combined.close()


def _selection_from_payload(payload: object) -> CropBox:
    if not isinstance(payload, dict):
        raise ValueError("selection must be an object")
    values: list[int] = []
    for key in ("left", "top", "right", "bottom"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"selection {key} must be numeric")
        values.append(round(value))
    return values[0], values[1], values[2], values[3]


@dataclass
class WebSelectionState:
    image: Image.Image
    square_crop: bool
    overwrite_alpha_mask: bool
    threshold: float
    working_max_width: int
    maximum_display_size: tuple[int, int]
    initial_selection: CropBox | None = None
    result: tuple[CropBox, CropBox, float | None] | None = None
    cancelled: bool = False
    preview: bytes | None = None
    preview_version: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def preview_selection(self, selection: CropBox, threshold: float) -> dict[str, object]:
        if not self.overwrite_alpha_mask:
            raise ValueError("segmentation preview is unavailable in source-alpha mode")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a positive finite number")
        crop_box = crop_box_for_selection(selection, self.image.size, self.square_crop)
        product_box = _relative_product_box(selection, crop_box)
        cropped = self.image.crop(crop_box)
        try:
            alpha = segment_product_alpha(
                cropped,
                product_box,
                background_threshold=threshold,
                working_max_width=self.working_max_width,
            )
            try:
                preview = _segmentation_preview_bytes(cropped, alpha, self.maximum_display_size)
            finally:
                alpha.close()
        finally:
            cropped.close()
        with self.lock:
            self.initial_selection = selection
            self.threshold = threshold
            self.preview = preview
            self.preview_version += 1
            version = self.preview_version
        return {"threshold": threshold, "crop_box": list(crop_box), "version": version}

    def accept_source_selection(self, selection: CropBox) -> None:
        crop_box = crop_box_for_selection(selection, self.image.size, self.square_crop)
        with self.lock:
            self.result = selection, crop_box, None

    def accept_preview(self) -> None:
        with self.lock:
            selection = self.initial_selection
            threshold = self.threshold
            has_preview = self.preview is not None
        if selection is None or not has_preview:
            raise ValueError("generate a segmentation preview before accepting")
        crop_box = crop_box_for_selection(selection, self.image.size, self.square_crop)
        with self.lock:
            self.result = selection, crop_box, threshold


class WebSelectionServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        state: WebSelectionState,
    ) -> None:
        self.state = state
        super().__init__(address, handler)


def _handler_for(
    source_bytes: bytes,
    display_size: tuple[int, int],
    comparison_views: tuple[tuple[int, bytes, tuple[int, int]], ...],
):
    class WebSelectionHandler(BaseHTTPRequestHandler):
        server_version = "RotGSWebSelection/1.0"

        @property
        def state(self) -> WebSelectionState:
            return self.server.state  # type: ignore[attr-defined, no-any-return]

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            request_path = urlsplit(self.path).path
            if request_path == "/":
                self._send_bytes(PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif request_path == "/api/config":
                selection = self.state.initial_selection
                self._send_json({
                    "source_width": self.state.image.width,
                    "source_height": self.state.image.height,
                    "display_width": display_size[0],
                    "display_height": display_size[1],
                    "overwrite_alpha_mask": self.state.overwrite_alpha_mask,
                    "threshold": self.state.threshold,
                    "comparisons": [
                        {
                            "index": index,
                            "angle": angle,
                            "display_width": size[0],
                            "display_height": size[1],
                        }
                        for index, (angle, _data, size) in enumerate(comparison_views)
                    ],
                    "initial_selection": dict(zip(("left", "top", "right", "bottom"), selection)) if selection is not None else None,
                })
            elif request_path == "/source.png":
                self._send_bytes(source_bytes, "image/png")
            elif request_path.startswith("/comparison/") and request_path.endswith(".png"):
                try:
                    index = int(Path(request_path).stem)
                except ValueError:
                    index = -1
                if 0 <= index < len(comparison_views):
                    self._send_bytes(comparison_views[index][1], "image/png")
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "comparison image not found")
            elif request_path == "/preview.png":
                with self.state.lock:
                    preview = self.state.preview
                if preview is None:
                    self._send_error(HTTPStatus.NOT_FOUND, "preview not generated")
                else:
                    self._send_bytes(preview, "image/png")
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            request_path = urlsplit(self.path).path
            try:
                payload = self._read_json()
                if request_path == "/api/preview":
                    selection = _selection_from_payload(payload.get("selection"))
                    threshold = payload.get("threshold")
                    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                        raise ValueError("threshold must be numeric")
                    self._send_json(self.state.preview_selection(selection, float(threshold)))
                elif request_path == "/api/accept":
                    if self.state.overwrite_alpha_mask:
                        self.state.accept_preview()
                    else:
                        self.state.accept_source_selection(_selection_from_payload(payload.get("selection")))
                    self._send_json({"accepted": True})
                    self._shutdown_server()
                elif request_path == "/api/cancel":
                    self.state.cancelled = True
                    self._send_json({"cancelled": True})
                    self._shutdown_server()
                else:
                    self._send_error(HTTPStatus.NOT_FOUND, "not found")
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

        def _shutdown_server(self) -> None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def _send_json(self, payload: object) -> None:
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

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
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")

    return WebSelectionHandler


def choose_processing_settings_web(
    image: Image.Image,
    background_threshold: float,
    working_max_width: int,
    display_max_width: int,
    display_max_height: int,
    square_crop: bool,
    overwrite_alpha_mask: bool,
    *,
    port: int = DEFAULT_WEB_PORT,
    initial_selection: CropBox | None = None,
    comparison_images: list[tuple[int, Path]] | None = None,
) -> tuple[CropBox, CropBox, float | None]:
    """Select and confirm processing settings through a local temporary server."""
    if not 1 <= port <= 65535:
        raise ValueError("web port must be between 1 and 65535")
    if display_max_width <= 0 or display_max_height <= 0:
        raise ValueError("display dimensions must be positive")
    if initial_selection is not None:
        crop_box_for_selection(initial_selection, image.size, square_crop)
    source_bytes, display_size = _display_source(
        image,
        overwrite_alpha_mask,
        (display_max_width, display_max_height),
    )
    comparison_views = _display_comparison_sources(
        comparison_images,
        overwrite_alpha_mask,
        (display_max_width, display_max_height),
        image.size,
    )
    state = WebSelectionState(
        image=image,
        square_crop=square_crop,
        overwrite_alpha_mask=overwrite_alpha_mask,
        threshold=background_threshold,
        working_max_width=working_max_width,
        maximum_display_size=(display_max_width, display_max_height),
        initial_selection=initial_selection,
    )
    server = WebSelectionServer(
        ("127.0.0.1", port),
        _handler_for(source_bytes, display_size, comparison_views),
        state,
    )
    print(f"Web selection server listening on: http://127.0.0.1:{port}/", flush=True)
    print(f"In VS Code, forward port {port}, then open: http://localhost:{port}/", flush=True)
    print("Waiting for selection in the browser. Press Ctrl+C to cancel.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt as exc:
        raise SelectionCancelled("web selection cancelled") from exc
    finally:
        server.server_close()
    if state.cancelled or state.result is None:
        raise SelectionCancelled("web selection cancelled")
    return state.result
