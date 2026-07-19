"""
Pytest root conftest for the backend service.

Adds `backend/` to sys.path so that `from src.X import ...` resolves
correctly regardless of how pytest is invoked (e.g. `python -m pytest tests/`).
"""
import sys
from pathlib import Path

# Ensure the backend/ directory (this file's parent) is on sys.path
_backend_root = str(Path(__file__).resolve().parent)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)
