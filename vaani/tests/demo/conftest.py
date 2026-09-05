"""
Shared pytest setup for tests/demo — add repo root to sys.path so
`app.*` imports resolve (app is a plain package, not pip-installed).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
