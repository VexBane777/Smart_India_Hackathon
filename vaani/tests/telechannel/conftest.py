"""
Shared pytest fixtures/setup for telechannel/ stage tests.

`telechannel.stages.*` is a plain package (not pip-installed), so we add the
repo root (`vaani_documentation/`) to sys.path here to make it importable as
`telechannel.stages.<module>` from the test suite.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
