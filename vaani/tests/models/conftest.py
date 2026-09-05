"""
Shared pytest fixtures/setup for models tests.

`registry` is a plain module (not pip-installed), so we add the
repo root (`vaani/`) to sys.path here to make it importable as
`registry` and other vaani modules from the test suite.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
