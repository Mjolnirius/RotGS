"""Diagnostic persistence and CUDA error attribution without GPU inference."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.preprocessing import rotgs_sam31_diagnostics as diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_success_logs_console_status_and_restores_cuda_environment(self):
        @diagnostics.diagnostic_run
        def run(*, diagnostics_dir=None, cuda_sync=False):
            self.assertEqual(os.environ['CUDA_LAUNCH_BLOCKING'], '1')
            print('stdout evidence', flush=True)
            print('stderr evidence', file=sys.stderr, flush=True)
            return dict(output_folder='output', processed=73)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(diagnostics.RunDiagnostics, '_monitor', lambda self: None), \
             patch.dict(os.environ, {'CUDA_LAUNCH_BLOCKING': '0'}), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run(diagnostics_dir=directory, cuda_sync=True)
            self.assertEqual(result['processed'], 73)
            self.assertEqual(os.environ['CUDA_LAUNCH_BLOCKING'], '0')
            folder = next(Path(directory).iterdir())
            self.assertEqual(json.loads((folder / 'status.json').read_text())['status'], 'completed')
            console = (folder / 'console.log').read_text()
            self.assertIn('stdout evidence', console)
            self.assertIn('stderr evidence', console)
            self.assertEqual(json.loads((folder / 'environment.json').read_text())['cuda_launch_blocking'], '1')

    def test_sync_failure_is_attributed_to_request_and_saved_with_frame(self):
        cuda = SimpleNamespace(memory_allocated=lambda: 1024, memory_reserved=lambda: 2048,
                               max_memory_allocated=lambda: 4096,
                               synchronize=Mock(side_effect=[None, RuntimeError('kernel failed at sync')]))

        @diagnostics.diagnostic_run
        def run(*, diagnostics_dir=None, cuda_sync=False):
            diagnostics.diagnostic_event('frame_begin', frame_index=45, source_image='product_225.png')
            with diagnostics.diagnostic_span('add_prompt', torch=SimpleNamespace(cuda=cuda), prompt='can'):
                pass

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(diagnostics.RunDiagnostics, '_monitor', lambda self: None), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'kernel failed at sync'):
                run(diagnostics_dir=directory, cuda_sync=True)
            folder = next(Path(directory).iterdir())
            failure = json.loads((folder / 'failure.json').read_text())
            self.assertEqual(failure['context']['frame_index'], 45)
            self.assertEqual(failure['context']['prompt'], 'can')
            self.assertIn('kernel failed at sync', failure['traceback'])
            events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
            error = next(event for event in events if event['event'] == 'operation_error')
            self.assertEqual(error['operation'], 'add_prompt')
            self.assertEqual(error['cuda_peak_allocated_bytes'], 4096)
            self.assertTrue(any(event['event'] == 'cuda_sync_after' for event in events))

    def test_telemetry_records_gpu_loss_and_ram_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = diagnostics.RunDiagnostics(directory)
            try:
                with patch.object(diagnostics.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=1, stdout='', stderr='GPU device handle: Unknown Error',
                )):
                    recorder.snapshot()
                event = json.loads((recorder.folder / 'telemetry.jsonl').read_text())
                self.assertIn('Unknown Error', event['gpu_error'])
                self.assertGreater(event['process_rss_bytes'], 0)
                self.assertGreater(event['host_ram_available_bytes'], 0)
            finally:
                recorder.events.close()
                recorder.telemetry.close()
                recorder.console.close()

    def test_last_operation_and_input_survive_sigkill(self):
        code = '''
from pathlib import Path
import sys
import time
from tools.preprocessing import rotgs_sam31_diagnostics as d
d.RunDiagnostics._monitor = lambda self: None
@d.diagnostic_run
def run(*, diagnostics_dir=None, cuda_sync=False):
    image = Path(diagnostics_dir) / 'input.jpg'
    image.write_bytes(b'exact working image bytes')
    d.diagnostic_save_frame(image, 45)
    d.diagnostic_event('frame_begin', frame_index=45, source_image='product_225.png')
    with d.diagnostic_span('add_prompt', prompt='can'):
        print('READY', flush=True)
        time.sleep(30)
run(diagnostics_dir=sys.argv[1])
'''
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([sys.executable, '-c', code, directory],
                                       cwd=Path(__file__).resolve().parents[2],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(process.stdout.readline().startswith('SAM31 diagnostics:'))
                self.assertEqual(process.stdout.readline().strip(), 'READY')
                process.kill()
                process.communicate(timeout=5)
                folder = next(path for path in Path(directory).iterdir() if path.is_dir())
                self.assertEqual(json.loads((folder / 'status.json').read_text())['status'], 'running')
                events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['event'], 'operation_begin')
                self.assertEqual(events[-1]['operation'], 'add_prompt')
                self.assertEqual(events[-1]['context']['frame_index'], 45)
                self.assertEqual((folder / 'last_frame.jpg').read_bytes(), b'exact working image bytes')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
