"""Shared pytest fixtures.

Adds the project root to sys.path so `from workers import ...` works when
pytest is invoked from any cwd. Mirrors what scripts/smoke_concurrent.py
already does at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
