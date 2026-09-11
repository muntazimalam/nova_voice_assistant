"""Shared pytest fixtures: make the project root importable and keep app
imports offline (no model downloads, no API calls).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Ensure importing app.main never triggers HTTP/model downloads in tests."""
    from app import main

    # Do real warmup in dev, never in tests.
    main._warmup = lambda: None
    # Skip advanced feature init (no torch/opus in test env)
    main._init_advanced_features = lambda: None
