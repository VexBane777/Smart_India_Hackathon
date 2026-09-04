"""
Shared pytest fixtures/setup for legal/ script tests.

`legal/revocation.py` and `legal/data_purge.py` are standalone scripts, not
part of an installed package, so we add the `legal/` directory to sys.path
here to make them importable as plain modules from the test suite.
"""

import sys
from pathlib import Path

LEGAL_DIR = Path(__file__).resolve().parents[2] / "legal"
if str(LEGAL_DIR) not in sys.path:
    sys.path.insert(0, str(LEGAL_DIR))
