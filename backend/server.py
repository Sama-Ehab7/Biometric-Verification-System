"""Kept so the old command still works:
    python -m uvicorn backend.server:app --port 8000
The app now lives in backend/app.py (see backend/README.md)."""
from .app import app  # noqa: F401
