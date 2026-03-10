"""Shared fixtures for SoloDet tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_image_rgb():
    """Return a factory that creates a synthetic BGR image (numpy array)."""

    def _make(width: int = 640, height: int = 480, color=(128, 128, 128)):
        img = np.full((height, width, 3), color, dtype=np.uint8)
        return img

    return _make
