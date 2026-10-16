"""Shared research destinations, independent of the producing codebase."""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get('SCANNING_DATA_ROOT', '/shared/datasets/3D_Scanning_MT'))
RESEARCH_ROOT = Path(os.environ.get('SCANNING_RESEARCH_ROOT', str(DATA_ROOT / 'research')))
REPORTS_ROOT = RESEARCH_ROOT / 'reports'
ROTGS_OUTPUT = DATA_ROOT / 'output_RotGS'
