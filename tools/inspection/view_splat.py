"""Serve a trained Gaussian PLY and open it in the SuperSplat web editor.

The server listens on localhost by default so it can be accessed safely through
VS Code port forwarding. With no model path, the most recently modified valid
run below ``./output`` is selected automatically.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
DEFAULT_EDITOR_URL = "https://superspl.at/editor"
ITERATION_PATTERN = re.compile(r"^iteration_(\d+)$")


@dataclass(frozen=True)
class SplatSelection:
    """Resolved model information shown when the viewer starts."""

    run_directory: Path
    ply_path: Path
    iteration: int | None


def _iteration_number(path: Path) -> int | None:
    match = ITERATION_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _selection_from_ply(ply_path: Path) -> SplatSelection:
    ply_path = ply_path.expanduser().resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY does not exist: {ply_path}")
    if ply_path.suffix.lower() != ".ply":
        raise ValueError(f"expected a .ply file, got: {ply_path}")

    iteration = _iteration_number(ply_path.parent)
    if iteration is not None and ply_path.parent.parent.name == "point_cloud":
        run_directory = ply_path.parent.parent.parent
    else:
        run_directory = ply_path.parent
    return SplatSelection(run_directory, ply_path, iteration)


def _latest_iteration(run_directory: Path) -> SplatSelection:
    point_cloud_root = run_directory / "point_cloud"
    candidates: list[tuple[int, Path]] = []
    if point_cloud_root.is_dir():
        for iteration_directory in point_cloud_root.iterdir():
            iteration = _iteration_number(iteration_directory)
            ply_path = iteration_directory / "point_cloud.ply"
            if iteration is not None and ply_path.is_file():
                candidates.append((iteration, ply_path))

    if not candidates:
        direct_ply = run_directory / "point_cloud.ply"
        if direct_ply.is_file():
            return _selection_from_ply(direct_ply)
        raise FileNotFoundError(
            f"no point_cloud/iteration_*/point_cloud.ply found below: "
            f"{run_directory}"
        )

    iteration, ply_path = max(candidates, key=lambda candidate: candidate[0])
    return SplatSelection(run_directory.resolve(), ply_path.resolve(), iteration)


def resolve_splat(
    model_path: str | Path | None = None,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> SplatSelection:
    """Resolve a PLY from an explicit path or the newest run in output_root.

    Explicit paths may identify a run directory, an ``iteration_*`` directory,
    or a PLY file. Automatic selection ranks runs by the newest PLY modification
    time and selects the highest numbered iteration within that run.
    """

    if model_path is not None:
        path = Path(model_path).expanduser().resolve()
        if path.is_file():
            return _selection_from_ply(path)
        if not path.is_dir():
            raise FileNotFoundError(f"model path does not exist: {path}")
        direct_ply = path / "point_cloud.ply"
        if direct_ply.is_file():
            return _selection_from_ply(direct_ply)
        return _latest_iteration(path)

    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"output root does not exist: {root}; pass a model path explicitly"
        )

    newest_ply_by_run: dict[Path, float] = {}
    for ply_path in root.glob("**/point_cloud/iteration_*/point_cloud.ply"):
        iteration = _iteration_number(ply_path.parent)
        if iteration is None or not ply_path.is_file():
            continue
        run_directory = ply_path.parent.parent.parent
        modified = ply_path.stat().st_mtime
        newest_ply_by_run[run_directory] = max(
            modified,
            newest_ply_by_run.get(run_directory, float("-inf")),
        )

    for ply_path in root.glob("**/point_cloud.ply"):
        if ply_path.parent.parent.name == "point_cloud":
            continue
        run_directory = ply_path.parent
        modified = ply_path.stat().st_mtime
        newest_ply_by_run[run_directory] = max(
            modified,
            newest_ply_by_run.get(run_directory, float("-inf")),
        )

    if not newest_ply_by_run:
        raise FileNotFoundError(
            f"no Gaussian point_cloud.ply found below output root: {root}"
        )

    newest_run = max(
        newest_ply_by_run,
        key=lambda run: (newest_ply_by_run[run], str(run)),
    )
    return _latest_iteration(newest_run)


def _landing_page(selection: SplatSelection, editor_url: str) -> bytes:
    model_name = html.escape(selection.run_directory.name)
    editor_url_json = json.dumps(editor_url)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Opening {model_name} in SuperSplat</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    body {{ display: grid; min-height: 100vh; margin: 0; place-items: center;
            background: #111318; color: #f4f5f7; }}
    main {{ max-width: 44rem; padding: 2rem; text-align: center; }}
    a {{ display: inline-block; margin-top: 1rem; padding: .7rem 1rem;
         border-radius: .4rem; background: #4c7dff; color: white;
         text-decoration: none; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <main>
    <h1>Opening SuperSplat</h1>
    <p>Loading <strong>{model_name}</strong> from this forwarded port.</p>
    <p id="status">Preparing the editor URL…</p>
    <a id="open" href="#">Open SuperSplat manually</a>
  </main>
  <script>
    const editor = new URL({editor_url_json});
    editor.searchParams.set('load', new URL('model.ply', location.href).href);
    const link = document.querySelector('#open');
    link.href = editor.href;
    document.querySelector('#status').textContent =
      'If the editor does not open automatically, use the button below.';
    if (!new URLSearchParams(location.search).has('stay')) {{
      setTimeout(() => location.replace(editor.href), 250);
    }}
  </script>
</body>
</html>
""".encode("utf-8")


def _handler_for(
    selection: SplatSelection,
    editor_url: str,
) -> type[BaseHTTPRequestHandler]:
    landing_page = _landing_page(selection, editor_url)
    ply_path = selection.ply_path

    class SplatRequestHandler(BaseHTTPRequestHandler):
        server_version = "RotGSSplatViewer/1.0"

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def _send_bytes(self, body: bytes, content_type: str, head: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors_headers()
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _serve(self, head: bool = False) -> None:
            route = urlsplit(self.path).path
            if route in {"/", "/index.html"}:
                self._send_bytes(landing_page, "text/html; charset=utf-8", head)
                return
            if route == "/healthz":
                self._send_bytes(b"ok\n", "text/plain; charset=utf-8", head)
                return
            if route != "/model.ply":
                self.send_error(404)
                return

            try:
                size = ply_path.stat().st_size
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", 'inline; filename="point_cloud.ply"')
                self.send_header("Cache-Control", "public, max-age=3600")
                self._cors_headers()
                self.end_headers()
                if not head:
                    with ply_path.open("rb") as source:
                        shutil.copyfileobj(source, self.wfile, length=1024 * 1024)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._serve()

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self._serve(head=True)

        def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(204)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

    return SplatRequestHandler


def view_splat(
    model_path: str | Path | None = None,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    host: str = "127.0.0.1",
    port: int = 8080,
    editor_url: str = DEFAULT_EDITOR_URL,
    open_browser: bool = False,
) -> None:
    """Serve a selected Gaussian PLY until interrupted with Ctrl+C."""

    selection = resolve_splat(model_path, output_root=output_root)
    server = ThreadingHTTPServer(
        (host, port),
        _handler_for(selection, editor_url),
    )
    actual_host, actual_port = server.server_address[:2]
    display_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    local_url = f"http://{display_host}:{actual_port}/"

    iteration_label = selection.iteration if selection.iteration is not None else "direct PLY"
    print(f"Model:    {selection.run_directory}")
    print(f"Iteration: {iteration_label}")
    print(f"PLY:      {selection.ply_path}")
    print(f"Viewer:   {local_url}")
    print("Forward this port in VS Code, open its URL, and press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(local_url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping splat viewer.")
    finally:
        server.server_close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="run directory, iteration directory, or point_cloud.ply",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        help="explicit model path (equivalent to the positional path)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"automatic-search root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--open", action="store_true", help="open a browser automatically")
    parser.add_argument("--editor-url", default=DEFAULT_EDITOR_URL, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.path is not None and args.model_path is not None:
        raise SystemExit("provide either the positional path or --model-path, not both")
    try:
        view_splat(
            args.model_path or args.path,
            output_root=args.output_root,
            host=args.host,
            port=args.port,
            editor_url=args.editor_url,
            open_browser=args.open,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
