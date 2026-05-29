"""
tests/conftest.py
-----------------
Pytest configuration for brain2text-experiments tests.
"""
import sys
from pathlib import Path

# Ensure experiments root is on path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require GPU + model downloads (skip with -m 'not slow')"
    )
