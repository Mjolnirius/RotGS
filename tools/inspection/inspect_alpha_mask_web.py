"""Inspect PNG alpha masks in a browser without modifying the source images."""

from __future__ import annotations

import argparse
import io
import json
import secrets
import shutil
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError

if __package__:
    from .inspect_alpha_mask import find_alpha_pngs
else:
    from inspect_alpha_mask import find_alpha_pngs


DEFAULT_PORT = 8765
VIEW_MODES = {"rgba", "alpha", "rgb"}
RENDER_CACHE_ENTRIES = 16
PARTIAL_HIGHLIGHT_STRENGTH = 176
_RENDER_SLOTS = BoundedSemaphore(1)


PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alpha mask inspector</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #edf0f7; }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body { min-width: 320px; }
    .app { height: 100vh; display: flex; flex-direction: column; }
    header { display: flex; align-items: center; gap: 14px; padding: 10px 14px; border-bottom: 1px solid #30343e; background: #191c23; flex-wrap: wrap; }
    .identity { min-width: 180px; flex: 1; }
    h1 { margin: 0 0 3px; overflow: hidden; font-size: 16px; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
    .subline { color: #aeb5c3; font-size: 13px; }
    .controls { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
    button, .toggle { border: 1px solid #3b414f; border-radius: 7px; padding: 6px 10px; background: #252a34; color: inherit; font: inherit; }
    button { cursor: pointer; }
    button:hover, .toggle:hover { background: #303744; border-color: #545d70; }
    button:focus-visible, input:focus-visible { outline: 2px solid #76a9ff; outline-offset: 2px; }
    button.active { background: #245ca8; border-color: #4386dd; }
    .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
    .toggle input { margin: 0; accent-color: #e33d4f; }
    .viewer { min-height: 0; flex: 1; position: relative; padding: 10px; background: #101218; }
    .panes { width: 100%; height: 100%; display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; }
    .panes.side { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .pane { position: relative; min-width: 0; min-height: 0; overflow: hidden; cursor: grab; touch-action: none; background-color: #d1d1d1; background-image: linear-gradient(45deg, #9d9d9d 25%, transparent 25%), linear-gradient(-45deg, #9d9d9d 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #9d9d9d 75%), linear-gradient(-45deg, transparent 75%, #9d9d9d 75%); background-position: 0 0, 0 12px, 12px -12px, -12px 0; background-size: 24px 24px; border: 1px solid #343945; border-radius: 5px; }
    .pane.dragging { cursor: grabbing; }
    .pane[hidden] { display: none; }
    .pane img { position: absolute; left: 0; top: 0; max-width: none; max-height: none; transform-origin: 0 0; user-select: none; -webkit-user-drag: none; will-change: transform; }
    .pane-label { position: absolute; z-index: 2; top: 8px; left: 8px; padding: 4px 7px; border-radius: 5px; background: #111b; color: #fff; font-size: 12px; pointer-events: none; }
    #message { display: none; position: absolute; z-index: 4; left: 50%; top: 50%; max-width: min(680px, 80%); transform: translate(-50%, -50%); border-radius: 8px; padding: 16px; background: #401d22; color: #ffd9dc; }
    footer { display: flex; align-items: center; gap: 16px; min-height: 50px; padding: 8px 14px; border-top: 1px solid #30343e; background: #191c23; color: #c8ced9; font-size: 13px; flex-wrap: wrap; }
    .stat strong { color: #fff; font-weight: 600; }
    .legend { color: #ff6b76; }
    .legend[hidden] { display: none; }
    .hint { margin-left: auto; color: #8e97a8; }
    @media (max-width: 760px) { .panes.side { grid-template-columns: minmax(0, 1fr); grid-template-rows: repeat(2, minmax(0, 1fr)); } .hint { width: 100%; margin-left: 0; } }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="identity"><h1 id="filename">Loading…</h1><div class="subline"><span id="counter">— / —</span> · <span id="dimensions">—</span> · <span id="zoom">Fit</span></div></div>
      <div class="controls" aria-label="Image navigation"><button id="previous" type="button" title="Previous (Left or Up)">← Previous</button><button id="next" type="button" title="Next (Right or Down)">Next →</button></div>
      <div class="controls" aria-label="View mode"><button class="mode active" data-mode="rgba" type="button">RGBA</button><button class="mode" data-mode="alpha" type="button">Alpha</button><button class="mode" data-mode="rgb" type="button">RGB</button><button id="side" type="button" aria-pressed="false">RGBA + Alpha</button></div>
      <div class="controls" aria-label="Viewport controls"><button id="fit" type="button">Fit</button><button id="actual" type="button">100%</button><label class="toggle"><input id="highlight" type="checkbox"> Highlight partial alpha</label></div>
    </header>
    <main class="viewer" id="viewer">
      <div class="panes" id="panes">
        <section class="pane" id="pane-primary"><span class="pane-label" id="label-primary">RGBA</span><img id="image-primary" alt="Current PNG" draggable="false"></section>
        <section class="pane" id="pane-secondary" hidden><span class="pane-label">Alpha</span><img id="image-secondary" alt="Alpha view" draggable="false"></section>
      </div>
      <div id="message" role="alert"></div>
    </main>
    <footer>
      <span class="stat">Alpha range <strong id="range">—</strong></span><span class="stat">mean <strong id="mean">—</strong></span><span class="stat">transparent <strong id="transparent">—</strong></span><span class="stat">partial <strong id="partial">—</strong></span><span class="stat">opaque <strong id="opaque">—</strong></span><span class="stat">levels <strong id="levels">—</strong></span><span class="legend" id="legend" hidden>■ red = partial alpha (1..254) in RGBA/Alpha</span><span class="hint"><span id="cache-status">Cache —</span> · wheel zoom · drag pan · double-click Fit</span>
    </footer>
  </div>
  <script>
    "use strict";
    const state = { images: [], cacheToken: "", index: 0, mode: "rgba", side: false, highlight: false, request: 0, prefetchGeneration: 0, imageWidth: 0, imageHeight: 0, viewport: { scale: 1, centerX: 0, centerY: 0, fit: true } };
    const elements = Object.fromEntries(["filename", "counter", "dimensions", "zoom", "panes", "message", "range", "mean", "transparent", "partial", "opaque", "levels", "legend", "cache-status", "label-primary"].map(id => [id, document.getElementById(id)]));
    const paneViews = [
      { pane: document.getElementById("pane-primary"), image: document.getElementById("image-primary") },
      { pane: document.getElementById("pane-secondary"), image: document.getElementById("image-secondary") },
    ];
    const blobCache = new Map();
    const metadataCache = new Map();
    let drag = null;

    function percent(value) { return `${value.toFixed(2)}%`; }
    function visiblePanes() { return state.side ? paneViews : paneViews.slice(0, 1); }
    function paneModes() { return state.side ? ["rgba", "alpha"] : [state.mode]; }
    function imageUrl(index, mode) { const highlight = state.highlight && (mode === "rgba" || mode === "alpha"); return `/images/${index}?mode=${mode}&highlight=${highlight ? 1 : 0}&session=${encodeURIComponent(state.cacheToken)}`; }
    function neighborhoodUrls() {
      const urls = new Set(); const modes = paneModes(); const first = Math.max(0, state.index - 3); const last = Math.min(state.images.length - 1, state.index + 3);
      for (let index = first; index <= last; index += 1) modes.forEach(mode => urls.add(imageUrl(index, mode)));
      return urls;
    }
    function disposeEntry(entry) { entry.controller.abort(); if (entry.objectUrl) URL.revokeObjectURL(entry.objectUrl); }
    function retainNeighborhood(desired) {
      for (const [url, entry] of blobCache) if (!desired.has(url)) { disposeEntry(entry); blobCache.delete(url); }
      const first = Math.max(0, state.index - 3); const last = Math.min(state.images.length - 1, state.index + 3);
      for (const index of metadataCache.keys()) if (index < first || index > last) metadataCache.delete(index);
      updateCacheStatus(desired.size);
    }
    function ensureBlob(url) {
      const existing = blobCache.get(url); if (existing) return existing.promise;
      const controller = new AbortController(); const entry = { controller, blob: null, objectUrl: null, promise: null };
      entry.promise = fetch(url, { signal: controller.signal, cache: "force-cache" }).then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.blob(); }).then(blob => { if (blobCache.get(url) !== entry) throw new DOMException("Evicted", "AbortError"); entry.blob = blob; updateCacheStatus(neighborhoodUrls().size); return blob; }).catch(error => { if (blobCache.get(url) === entry) blobCache.delete(url); throw error; });
      blobCache.set(url, entry); return entry.promise;
    }
    function objectUrlFor(url) { const entry = blobCache.get(url); if (!entry || !entry.blob) throw new Error("image is not cached"); if (!entry.objectUrl) entry.objectUrl = URL.createObjectURL(entry.blob); return entry.objectUrl; }
    function releaseNonVisibleObjectUrls(visibleUrls) { for (const [url, entry] of blobCache) if (entry.objectUrl && !visibleUrls.has(url)) { URL.revokeObjectURL(entry.objectUrl); entry.objectUrl = null; } }
    function updateCacheStatus(expected) { let ready = 0; for (const entry of blobCache.values()) if (entry.blob) ready += 1; elements["cache-status"].textContent = `Cache ${ready}/${expected} compressed views`; }
    async function prefetchNeighborhood() {
      const generation = ++state.prefetchGeneration; const desired = neighborhoodUrls(); retainNeighborhood(desired); const offsets = [0, 1, -1, 2, -2, 3, -3]; const modes = paneModes();
      for (const offset of offsets) { if (generation !== state.prefetchGeneration) return; const index = state.index + offset; if (index < 0 || index >= state.images.length) continue; try { await Promise.all(modes.map(mode => ensureBlob(imageUrl(index, mode)))); } catch (error) { if (error.name !== "AbortError") console.warn("Prefetch failed", error); } }
    }
    function metadataFor(index) {
      const existing = metadataCache.get(index); if (existing) return existing;
      const promise = fetch(`/api/images/${index}`).then(response => response.json().then(data => ({ response, data }))).then(({ response, data }) => { if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); return data; });
      metadataCache.set(index, promise); return promise;
    }
    function clearStats() { ["range", "mean", "transparent", "partial", "opaque", "levels"].forEach(key => { elements[key].textContent = "—"; }); elements.dimensions.textContent = "Loading metadata…"; }
    function showError(message) { elements.message.style.display = "block"; elements.message.textContent = message; }
    function clearError() { elements.message.style.display = "none"; }
    function fitScale() { if (!state.imageWidth || !state.imageHeight) return 1; return Math.min(...visiblePanes().map(({ pane }) => Math.min(pane.clientWidth / state.imageWidth, pane.clientHeight / state.imageHeight))); }
    function applyTransforms() {
      if (!state.imageWidth || !state.imageHeight) return;
      for (const { pane, image } of visiblePanes()) { const x = pane.clientWidth / 2 - state.viewport.centerX * state.viewport.scale; const y = pane.clientHeight / 2 - state.viewport.centerY * state.viewport.scale; image.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${state.viewport.scale})`; }
      const prefix = state.viewport.fit ? "Fit · " : ""; elements.zoom.textContent = `${prefix}${(state.viewport.scale * 100).toFixed(1)}%`;
    }
    function resetFit() { if (!state.imageWidth || !state.imageHeight) return; state.viewport.scale = fitScale(); state.viewport.centerX = state.imageWidth / 2; state.viewport.centerY = state.imageHeight / 2; state.viewport.fit = true; applyTransforms(); }
    function resetActual() { if (!state.imageWidth || !state.imageHeight) return; state.viewport.scale = 1; state.viewport.centerX = state.imageWidth / 2; state.viewport.centerY = state.imageHeight / 2; state.viewport.fit = false; applyTransforms(); }
    function setImageDimensions(width, height) { if (!width || !height) return; const changed = width !== state.imageWidth || height !== state.imageHeight; state.imageWidth = width; state.imageHeight = height; if (changed) { state.viewport.centerX = width / 2; state.viewport.centerY = height / 2; } if (state.viewport.fit) resetFit(); else applyTransforms(); }
    async function renderPanes(request) {
      state.prefetchGeneration += 1;
      const modes = paneModes(); const desired = neighborhoodUrls(); retainNeighborhood(desired); paneViews[1].pane.hidden = !state.side; elements.panes.classList.toggle("side", state.side); elements["label-primary"].textContent = modes[0].toUpperCase(); const urls = modes.map(mode => imageUrl(state.index, mode));
      try {
        await Promise.all(urls.map(ensureBlob)); if (request !== state.request) return; const visibleUrls = new Set(urls); releaseNonVisibleObjectUrls(visibleUrls);
        visiblePanes().forEach(({ image }, paneIndex) => { const mode = modes[paneIndex]; image.alt = `${mode.toUpperCase()} view of ${state.images[state.index].filename}`; image.onerror = () => showError(`Could not render ${state.images[state.index].filename}.`); image.onload = () => setImageDimensions(image.naturalWidth, image.naturalHeight); image.src = objectUrlFor(urls[paneIndex]); });
        requestAnimationFrame(() => { if (state.viewport.fit) resetFit(); else applyTransforms(); }); void prefetchNeighborhood();
      } catch (error) { if (request === state.request && error.name !== "AbortError") showError(`${state.images[state.index].filename}: ${error.message}`); }
    }
    function navigate(index) {
      if (!state.images.length) return; state.index = (index + state.images.length) % state.images.length; const request = ++state.request; const item = state.images[state.index]; state.imageWidth = 0; state.imageHeight = 0; state.viewport.fit = true; elements.filename.textContent = item.filename; elements.counter.textContent = `${state.index + 1} / ${state.images.length}`; clearError(); clearStats(); void renderPanes(request);
      metadataFor(state.index).then(data => { if (request !== state.request) return; elements.dimensions.textContent = `${data.width} × ${data.height} px`; elements.range.textContent = `${data.alpha.min}–${data.alpha.max}`; elements.mean.textContent = data.alpha.mean.toFixed(2); elements.transparent.textContent = percent(data.alpha.transparent_percent); elements.partial.textContent = percent(data.alpha.partial_percent); elements.opaque.textContent = percent(data.alpha.opaque_percent); elements.levels.textContent = data.alpha.unique_values; setImageDimensions(data.width, data.height); }).catch(error => { if (request === state.request) showError(`${item.filename}: ${error.message}`); });
    }
    function refreshViews() { const request = ++state.request; clearError(); void renderPanes(request); }
    function zoomAt(pane, clientX, clientY, factor) {
      if (!state.imageWidth || !state.imageHeight) return; const rect = pane.getBoundingClientRect(); const cursorX = clientX - rect.left; const cursorY = clientY - rect.top; const oldScale = state.viewport.scale; const imageX = state.viewport.centerX + (cursorX - pane.clientWidth / 2) / oldScale; const imageY = state.viewport.centerY + (cursorY - pane.clientHeight / 2) / oldScale; const minScale = Math.max(0.002, fitScale() * 0.2); const newScale = Math.min(16, Math.max(minScale, oldScale * factor)); state.viewport.centerX = imageX - (cursorX - pane.clientWidth / 2) / newScale; state.viewport.centerY = imageY - (cursorY - pane.clientHeight / 2) / newScale; state.viewport.scale = newScale; state.viewport.fit = false; applyTransforms();
    }
    for (const { pane } of paneViews) {
      pane.addEventListener("wheel", event => { event.preventDefault(); zoomAt(pane, event.clientX, event.clientY, Math.exp(-event.deltaY * 0.0015)); }, { passive: false });
      pane.addEventListener("pointerdown", event => { if (event.button !== 0) return; pane.setPointerCapture(event.pointerId); pane.classList.add("dragging"); drag = { pane, pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, centerX: state.viewport.centerX, centerY: state.viewport.centerY }; });
      pane.addEventListener("pointermove", event => { if (!drag || drag.pointerId !== event.pointerId) return; state.viewport.centerX = drag.centerX - (event.clientX - drag.clientX) / state.viewport.scale; state.viewport.centerY = drag.centerY - (event.clientY - drag.clientY) / state.viewport.scale; state.viewport.fit = false; applyTransforms(); });
      const stopDrag = event => { if (!drag || drag.pointerId !== event.pointerId) return; drag.pane.classList.remove("dragging"); drag = null; };
      pane.addEventListener("pointerup", stopDrag); pane.addEventListener("pointercancel", stopDrag); pane.addEventListener("dblclick", resetFit);
    }
    document.getElementById("previous").addEventListener("click", () => navigate(state.index - 1)); document.getElementById("next").addEventListener("click", () => navigate(state.index + 1)); document.getElementById("fit").addEventListener("click", resetFit); document.getElementById("actual").addEventListener("click", resetActual);
    document.querySelectorAll(".mode").forEach(button => { button.addEventListener("click", () => { state.mode = button.dataset.mode; document.querySelectorAll(".mode").forEach(candidate => candidate.classList.toggle("active", candidate === button)); if (!state.side) refreshViews(); }); });
    document.getElementById("side").addEventListener("click", event => { state.side = !state.side; event.currentTarget.classList.toggle("active", state.side); event.currentTarget.setAttribute("aria-pressed", String(state.side)); refreshViews(); });
    document.getElementById("highlight").addEventListener("change", event => { state.highlight = event.currentTarget.checked; elements.legend.hidden = !state.highlight; refreshViews(); });
    window.addEventListener("keydown", event => { if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return; const actions = { ArrowLeft: () => navigate(state.index - 1), ArrowUp: () => navigate(state.index - 1), ArrowRight: () => navigate(state.index + 1), ArrowDown: () => navigate(state.index + 1), Home: () => navigate(0), End: () => navigate(state.images.length - 1) }; if (actions[event.key]) { event.preventDefault(); actions[event.key](); } });
    new ResizeObserver(() => { if (state.viewport.fit) resetFit(); else applyTransforms(); }).observe(elements.panes);
    fetch("/api/images").then(response => response.json().then(data => ({ response, data }))).then(({ response, data }) => { if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); state.images = data.images; state.cacheToken = data.cache_token; navigate(0); }).catch(error => showError(`Could not load the image list: ${error.message}`));
  </script>
</body>
</html>
"""


def _port_number(value: str) -> int:
    """Parse and validate a TCP port number for argparse."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _safe_image_files(folder: Path, image_files: list[Path]) -> tuple[Path, ...]:
    """Resolve discovered files and exclude symlinks escaping the dataset."""
    safe_files: list[Path] = []
    for image_file in image_files:
        resolved = image_file.resolve(strict=True)
        if not resolved.is_relative_to(folder):
            continue
        safe_files.append(resolved)
    if not safe_files:
        raise ValueError(f"no in-folder PNG files found directly inside: {folder}")
    return tuple(safe_files)


def _open_alpha_png(path: Path) -> Image.Image:
    """Open and fully load one PNG, requiring an alpha/transparency channel."""
    try:
        image = Image.open(path)
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not read PNG image: {path.name}") from exc
    if image.format != "PNG":
        image.close()
        raise ValueError(f"file is not a PNG image: {path.name}")
    if "A" not in image.getbands() and "transparency" not in image.info:
        image.close()
        raise ValueError(f"PNG has no alpha channel: {path.name}")
    return image


@lru_cache(maxsize=16)
def _image_metadata(path: Path) -> dict[str, object]:
    """Return dimensions and exact histogram-derived alpha statistics."""
    with _RENDER_SLOTS, _open_alpha_png(path) as image:
        alpha = image.convert("RGBA").getchannel("A")
        histogram = alpha.histogram()
        width, height = image.size

    pixel_count = width * height
    transparent = histogram[0]
    opaque = histogram[255]
    partial = pixel_count - transparent - opaque
    present_values = [value for value, count in enumerate(histogram) if count]
    weighted_sum = sum(value * count for value, count in enumerate(histogram))
    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "alpha": {
            "min": present_values[0],
            "max": present_values[-1],
            "mean": weighted_sum / pixel_count,
            "transparent_pixels": transparent,
            "transparent_percent": transparent * 100 / pixel_count,
            "partial_pixels": partial,
            "partial_percent": partial * 100 / pixel_count,
            "opaque_pixels": opaque,
            "opaque_percent": opaque * 100 / pixel_count,
            "unique_values": len(present_values),
        },
    }


@lru_cache(maxsize=RENDER_CACHE_ENTRIES)
def _render_view(path: Path, mode: str, highlight_partial: bool) -> bytes:
    """Render and cache one compressed display representation in memory."""
    with _RENDER_SLOTS, _open_alpha_png(path) as image:
        rgba = image.convert("RGBA")
        rendered: Image.Image | None = None
        try:
            if highlight_partial and mode in {"rgba", "alpha"}:
                alpha = rgba.getchannel("A")
                try:
                    mask = alpha.point(
                        lambda value: (
                            PARTIAL_HIGHLIGHT_STRENGTH
                            if 0 < value < 255
                            else 0
                        )
                    )
                    try:
                        if mode == "rgba":
                            rendered = rgba
                            red = (255, 0, 0, 255)
                        else:
                            rendered = Image.merge("RGB", (alpha, alpha, alpha))
                            red = (255, 0, 0)
                        rendered.paste(red, (0, 0), mask)
                    finally:
                        mask.close()
                finally:
                    alpha.close()
            elif mode == "alpha":
                rendered = rgba.getchannel("A")
            elif mode == "rgb":
                rendered = rgba.convert("RGB")
            else:
                rendered = rgba

            output = io.BytesIO()
            rendered.save(output, format="PNG")
            return output.getvalue()
        finally:
            if rendered is not None and rendered is not rgba:
                rendered.close()
            rgba.close()


def _handler_for(
    image_files: tuple[Path, ...], cache_token: str
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler closed over the validated image allow-list."""

    class AlphaMaskHandler(BaseHTTPRequestHandler):
        server_version = "AlphaMaskInspector/2.0"

        def do_GET(self) -> None:  # noqa: N802 - required HTTP handler name
            self._route(send_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - required HTTP handler name
            self._route(send_body=False)

        def _route(self, *, send_body: bool) -> None:
            request = urlsplit(self.path)
            path_parts = request.path.strip("/").split("/")

            if request.path == "/":
                self._send_bytes(
                    PAGE_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                    send_body=send_body,
                    cache_control="no-store",
                )
                return
            if request.path == "/api/images":
                payload = {
                    "count": len(image_files),
                    "cache_token": cache_token,
                    "images": [
                        {"index": index, "filename": path.name}
                        for index, path in enumerate(image_files)
                    ],
                }
                self._send_json(payload, send_body=send_body)
                return
            if len(path_parts) == 3 and path_parts[:2] == ["api", "images"]:
                index = self._parse_index(path_parts[2], send_body=send_body)
                if index is None:
                    return
                try:
                    payload = _image_metadata(image_files[index])
                except (OSError, ValueError) as exc:
                    self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), send_body)
                    return
                self._send_json(payload, send_body=send_body)
                return
            if len(path_parts) == 2 and path_parts[0] == "images":
                index = self._parse_index(path_parts[1], send_body=send_body)
                if index is None:
                    return
                query = parse_qs(request.query)
                mode = query.get("mode", ["rgba"])[0]
                highlight_value = query.get("highlight", ["0"])[0]
                if mode not in VIEW_MODES:
                    self._send_error(HTTPStatus.BAD_REQUEST, "invalid view mode", send_body)
                    return
                if highlight_value not in {"0", "1"}:
                    self._send_error(
                        HTTPStatus.BAD_REQUEST, "invalid highlight value", send_body
                    )
                    return
                highlight_partial = highlight_value == "1" and mode in {
                    "rgba",
                    "alpha",
                }
                try:
                    if mode == "rgba" and not highlight_partial:
                        self._send_file(image_files[index], send_body=send_body)
                    else:
                        data = _render_view(
                            image_files[index], mode, highlight_partial
                        )
                        self._send_bytes(
                            data,
                            "image/png",
                            send_body=send_body,
                            cache_control="private, max-age=3600",
                        )
                except (OSError, ValueError) as exc:
                    self._send_error(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), send_body)
                return

            self._send_error(HTTPStatus.NOT_FOUND, "not found", send_body)

        def _parse_index(
            self, value: str, *, send_body: bool
        ) -> int | None:
            try:
                index = int(value)
            except ValueError:
                index = -1
            if not 0 <= index < len(image_files):
                self._send_error(
                    HTTPStatus.NOT_FOUND, "image index not found", send_body
                )
                return None
            return index

        def _send_json(self, payload: object, *, send_body: bool) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(
                data,
                "application/json; charset=utf-8",
                send_body=send_body,
                cache_control="no-store",
            )

        def _send_error(self, status: HTTPStatus, message: str, send_body: bool) -> None:
            data = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status)
            self._common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(data)

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            *,
            send_body: bool,
            cache_control: str,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            if send_body:
                self.wfile.write(data)

        def _send_file(self, path: Path, *, send_body: bool) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self._common_headers()
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            if send_body:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, self.wfile)

        def _common_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src 'self' blob:; connect-src 'self'; "
                "style-src 'unsafe-inline'; script-src 'unsafe-inline'",
            )

    return AlphaMaskHandler


class AlphaMaskServer(ThreadingHTTPServer):
    """Threaded local utility server with prompt connection cleanup."""

    allow_reuse_address = True
    daemon_threads = True


def serve(folder_name: str | Path, *, port: int = DEFAULT_PORT) -> None:
    """Discover alpha PNGs and serve the read-only browser inspector."""
    folder, discovered_files = find_alpha_pngs(folder_name)
    image_files = _safe_image_files(folder, discovered_files)
    cache_token = secrets.token_urlsafe(12)
    server = AlphaMaskServer(
        ("0.0.0.0", port), _handler_for(image_files, cache_token)
    )

    print(f"Found {len(image_files)} PNG files in: {folder}", flush=True)
    print(f"Server listening on: http://0.0.0.0:{port}/", flush=True)
    print(f"In VS Code, forward port {port}, then open: http://localhost:{port}/", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping alpha mask inspector.", flush=True)
    finally:
        server.server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PNG RGB content and alpha masks in a browser."
    )
    parser.add_argument("folder_name", help="folder directly containing alpha PNGs")
    parser.add_argument(
        "--port",
        type=_port_number,
        default=DEFAULT_PORT,
        help=f"HTTP server port (default: {DEFAULT_PORT})",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        serve(args.folder_name, port=args.port)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
