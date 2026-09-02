import os
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from view_splat import resolve_splat


class ResolveSplatTests(unittest.TestCase):
    def _ply(self, path: Path, contents: bytes = b"ply\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def test_explicit_run_uses_highest_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory) / "run"
            self._ply(run / "point_cloud/iteration_7/point_cloud.ply")
            expected = self._ply(run / "point_cloud/iteration_12/point_cloud.ply")

            selection = resolve_splat(run)

            self.assertEqual(selection.iteration, 12)
            self.assertEqual(selection.ply_path, expected.resolve())

    def test_explicit_iteration_directory_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            iteration = Path(temporary_directory) / "run/point_cloud/iteration_30"
            expected = self._ply(iteration / "point_cloud.ply")

            selection = resolve_splat(iteration)

            self.assertEqual(selection.iteration, 30)
            self.assertEqual(selection.ply_path, expected.resolve())

    def test_automatic_selection_uses_newest_run_then_highest_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            old_ply = self._ply(root / "old/point_cloud/iteration_100/point_cloud.ply")
            new_low = self._ply(root / "new/point_cloud/iteration_5/point_cloud.ply")
            expected = self._ply(root / "new/point_cloud/iteration_20/point_cloud.ply")
            old_ply.touch()
            new_low.touch()
            expected.touch()
            old_time = old_ply.stat().st_mtime - 10
            old_ply.touch()
            os.utime(old_ply, (old_time, old_time))

            selection = resolve_splat(output_root=root)

            self.assertEqual(selection.run_directory, (root / "new").resolve())
            self.assertEqual(selection.iteration, 20)
            self.assertEqual(selection.ply_path, expected.resolve())

    def test_empty_output_root_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "no Gaussian point_cloud.ply"):
                resolve_splat(output_root=temporary_directory)


if __name__ == "__main__":
    unittest.main()
