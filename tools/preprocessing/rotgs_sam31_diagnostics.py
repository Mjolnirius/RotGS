"""Persistent, opt-in diagnostics for SAM 3.1 preprocessing; no admin required."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import csv
from datetime import datetime, timezone
import faulthandler
from functools import wraps
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback


_ACTIVE = ContextVar("sam31_diagnostics", default=None)
_GPU_FIELDS = (
    "index", "uuid", "name", "driver_version", "memory.used", "memory.total",
    "memory.free", "temperature.gpu", "power.draw", "power.limit", "utilization.gpu",
)


class _Tee:
    def __init__(self, stream, file, lock):
        self.stream, self.file, self.lock = stream, file, lock

    def write(self, value):
        with self.lock:
            self.file.write(value)
            self.file.flush()
            return self.stream.write(value)

    def flush(self):
        with self.lock:
            self.file.flush()
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class RunDiagnostics:
    def __init__(self, parent, *, cuda_sync=False):
        parent = Path(parent).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix="sam31-", dir=parent))
        self.cuda_sync = cuda_sync
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.context = {}
        self.last_frame = None
        self.events = (self.folder / "events.jsonl").open("x", encoding="utf-8")
        self.telemetry = (self.folder / "telemetry.jsonl").open("x", encoding="utf-8")
        self.console = (self.folder / "console.log").open("x", encoding="utf-8")
        self.stop = threading.Event()
        self.thread = None

    def event(self, event, *, telemetry=False, **fields):
        with self.lock:
            if not telemetry:
                if event == "phase":
                    for key in ("frame_index", "source_image", "angle_degrees", "method", "prompt"):
                        self.context.pop(key, None)
                self.context.update({key: fields[key] for key in (
                    "phase", "frame_index", "source_image", "angle_degrees", "method", "prompt",
                ) if key in fields})
            row = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=round(time.monotonic() - self.started, 4),
                       event=event, context=self.context.copy(), **fields)
            file = self.telemetry if telemetry else self.events
            file.write(json.dumps(row, default=str) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def write_json(self, name, data):
        # Atomic replacement keeps a readable status even if the process dies.
        temporary = self.folder / f".{name}.tmp"
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, default=str)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(self.folder / name)

    def snapshot(self):
        row = {}
        try:
            import psutil
            process = psutil.Process()
            row["process_rss_bytes"] = process.memory_info().rss
            memory = psutil.virtual_memory()
            row.update(host_ram_total_bytes=memory.total, host_ram_available_bytes=memory.available)
        except (ImportError, OSError) as exc:
            row["ram_error"] = str(exc)
        try:
            row["cgroup_memory_events"] = dict(
                (key, int(value)) for key, value in (
                    line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
                )
            )
        except (OSError, ValueError):
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + ",".join(_GPU_FIELDS),
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
            )
            if result.returncode:
                row["gpu_error"] = (result.stderr or result.stdout).strip()
            else:
                row["gpus"] = [dict(zip(_GPU_FIELDS, (value.strip() for value in values)))
                               for values in csv.reader(result.stdout.splitlines())]
        except (OSError, subprocess.SubprocessError) as exc:
            row["gpu_error"] = str(exc)
        self.event("sample", telemetry=True, **row)

    def _monitor(self):
        while not self.stop.is_set():
            try:
                self.snapshot()
            except Exception as exc:
                self.event("telemetry_error", error=str(exc))
            self.stop.wait(1)

    def save_frame(self, path, frame_index):
        if path is None:
            return
        key = (str(path), frame_index)
        if key == self.last_frame:
            return
        data = Path(path).read_bytes()
        (self.folder / "last_frame.jpg").write_bytes(data)
        self.last_frame = key
        self.event("inference_input", frame_index=frame_index, working_image=str(path),
                   sha256=hashlib.sha256(data).hexdigest())

    @contextmanager
    def span(self, operation, *, torch=None, **fields):
        def cuda_memory():
            if torch is None:
                return {}
            try:
                return dict(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                            cuda_reserved_bytes=torch.cuda.memory_reserved(),
                            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            except Exception as exc:
                return dict(cuda_memory_error=str(exc))

        self.event("operation_begin", operation=operation, **fields, **cuda_memory())
        started = time.monotonic()
        try:
            if self.cuda_sync and torch is not None:
                self.event("cuda_sync_before", operation=operation)
                torch.cuda.synchronize()
            yield
            if self.cuda_sync and torch is not None:
                self.event("cuda_sync_after", operation=operation)
                torch.cuda.synchronize()
        except BaseException as exc:
            self.event("operation_error", operation=operation, error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc(), **cuda_memory())
            raise
        else:
            self.event("operation_end", operation=operation,
                       duration_seconds=round(time.monotonic() - started, 4), **cuda_memory())


def diagnostic_event(event, **fields):
    diagnostics = _ACTIVE.get()
    if diagnostics is not None:
        diagnostics.event(event, **fields)


def diagnostic_save_frame(path, frame_index):
    diagnostics = _ACTIVE.get()
    if diagnostics is not None:
        diagnostics.save_frame(path, frame_index)


def diagnostic_sources(*objects):
    diagnostics = _ACTIVE.get()
    if diagnostics is None:
        return
    sources = []
    for obj in objects:
        try:
            path = Path(inspect.getsourcefile(obj))
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except (TypeError, OSError):
            continue
    diagnostics.write_json("implementation.json", sources)


def diagnostic_span(operation, *, torch=None, **fields):
    diagnostics = _ACTIVE.get()
    return nullcontext() if diagnostics is None else diagnostics.span(operation, torch=torch, **fields)


def diagnostic_run(function):
    @wraps(function)
    def run(*args, **kwargs):
        parent, cuda_sync = kwargs.get("diagnostics_dir"), kwargs.get("cuda_sync", False)
        if parent is None and not cuda_sync:
            return function(*args, **kwargs)
        if parent is None:
            source = kwargs.get("folder_name", args[0] if args else ".")
            parent = Path(source).expanduser().resolve().parent / "sam31_diagnostics"
        diagnostics = RunDiagnostics(parent, cuda_sync=cuda_sync)
        print(f"SAM31 diagnostics: {diagnostics.folder}", flush=True)
        token = _ACTIVE.set(diagnostics)
        original_stdout, original_stderr = sys.stdout, sys.stderr
        previous_env = {key: os.environ.get(key) for key in (
            "CUDA_LAUNCH_BLOCKING", "TORCH_SHOW_CPP_STACKTRACES",
        )}
        enabled_fault_handler = not faulthandler.is_enabled()
        try:
            if cuda_sync:
                torch = sys.modules.get("torch")
                cuda = getattr(torch, "cuda", None)
                if getattr(cuda, "is_initialized", lambda: False)():
                    raise ValueError("--cuda-sync requires a new process before CUDA initialization")
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
            sys.stdout = _Tee(original_stdout, diagnostics.console, diagnostics.lock)
            sys.stderr = _Tee(original_stderr, diagnostics.console, diagnostics.lock)
            if enabled_fault_handler:
                faulthandler.enable(file=diagnostics.console, all_threads=True)
            versions = {}
            for package in ("torch", "sam3", "numpy", "Pillow"):
                try:
                    versions[package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    versions[package] = "unknown"
            diagnostics.write_json("environment.json", dict(
                python=sys.version, executable=sys.executable, platform=platform.platform(),
                argv=sys.argv, arguments=kwargs, packages=versions,
                cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
                torch_show_cpp_stacktraces=os.environ.get("TORCH_SHOW_CPP_STACKTRACES"),
            ))
            diagnostics.write_json("status.json", dict(status="running", pid=os.getpid()))
            diagnostics.thread = threading.Thread(target=diagnostics._monitor, daemon=True)
            diagnostics.thread.start()
            diagnostics.event("run_start", cuda_sync=cuda_sync)
            result = function(*args, **kwargs)
        except BaseException as exc:
            failure = dict(error_type=type(exc).__name__, error=str(exc),
                           context=diagnostics.context.copy(), traceback=traceback.format_exc())
            diagnostics.write_json("failure.json", failure)
            diagnostics.write_json("status.json", dict(status="failed", **failure))
            traceback.print_exc(file=diagnostics.console)
            diagnostics.event("run_failed", error=str(exc))
            raise
        else:
            diagnostics.write_json("status.json", dict(status="completed",
                                   output_folder=result.get("output_folder"), processed=result.get("processed")))
            diagnostics.event("run_completed", output_folder=result.get("output_folder"))
            return result
        finally:
            diagnostics.stop.set()
            if diagnostics.thread is not None:
                diagnostics.thread.join(timeout=5)
            if enabled_fault_handler:
                faulthandler.disable()
            sys.stdout, sys.stderr = original_stdout, original_stderr
            _ACTIVE.reset(token)
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            diagnostics.events.close()
            diagnostics.telemetry.close()
            diagnostics.console.close()
    return run
