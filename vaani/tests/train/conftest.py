"""
Shared pytest fixtures/setup for train/ tests.

`train` and `registry` are plain modules (not pip-installed), so we add the
repo root (`vaani/`) to sys.path here to make them importable as `train`
and `registry` from the test suite, regardless of pytest's invocation cwd.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
