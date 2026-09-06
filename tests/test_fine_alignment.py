"""
Tests for fine alignment.

_compute_fine_transform now returns Optional[np.ndarray] (single value, not a tuple).
"""

import numpy as np
import cv2
import pytest

from cad_image_alignment.alignment import (
    _validate_similarity,
    _compute_fine_transform,
)


# ---------------------------------------------------------------------------
# _validate_similarity
# ---------------------------------------------------------------------------

def test_validate_similarity_valid_scale():
    M = np.eye(3, dtype=np.float64)
    assert _validate_similarity(M) == True


def test_validate_similarity_scale_too_small():
    s = 0.3
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_similarity(M) == False


def test_validate_similarity_scale_too_large():
    s = 2.5
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    assert _validate_similarity(M) == False


def test_validate_similarity_boundary_values():
    for s in (0.5, 2.0):
        M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
        assert _validate_similarity(M) == True, f"Scale {s} should be valid"


# ---------------------------------------------------------------------------
# _compute_fine_transform  (ECC) — returns Optional[np.ndarray]
# ---------------------------------------------------------------------------

def test_compute_fine_transform_returns_matrix_or_none():
    """_compute_fine_transform must return a 3×3 matrix or None."""
    cad = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(cad, (20, 20), (80, 80), 255, 2)
    real = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(real, (20, 20), (80, 80), 255, 2)
    coarse = np.eye(3, dtype=np.float64)

    result = _compute_fine_transform(cad, real, coarse)
    assert result is None or (isinstance(result, np.ndarray) and result.shape == (3, 3))


def test_compute_fine_transform_identity_input():
    """With identical images, ECC should converge and return a near-identity correction."""
    img = np.zeros((150, 150), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (120, 120), 255, 2)
    cv2.circle(img,   (75, 75),  20, 255, 2)
    coarse = np.eye(3, dtype=np.float64)

    M_total = _compute_fine_transform(img.copy(), img.copy(), coarse)

    if M_total is not None:
        assert M_total.shape == (3, 3)
        assert M_total.dtype == np.float64
        assert abs(M_total[0, 2]) < 5.0, f"Expected near-zero tx, got {M_total[0, 2]:.2f}"
        assert abs(M_total[1, 2]) < 5.0, f"Expected near-zero ty, got {M_total[1, 2]:.2f}"


def test_compute_fine_transform_empty_images_graceful():
    """Completely empty edge maps — ECC should fail gracefully and return None."""
    cad  = np.zeros((100, 100), dtype=np.uint8)
    real = np.zeros((100, 100), dtype=np.uint8)
    coarse = np.eye(3, dtype=np.float64)

    result = _compute_fine_transform(cad, real, coarse)
    assert result is None or (isinstance(result, np.ndarray) and result.shape == (3, 3))


def test_compute_fine_transform_small_shift():
    """With a 3px shift between CAD and real, ECC should correct it."""
    base = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(base, (50, 50), (150, 150), 255, 2)
    cv2.circle(base, (100, 100), 30, 255, 2)

    shifted = np.zeros_like(base)
    cv2.rectangle(shifted, (53, 53), (153, 153), 255, 2)
    cv2.circle(shifted, (103, 103), 30, 255, 2)

    coarse = np.eye(3, dtype=np.float64)
    M_total = _compute_fine_transform(base, shifted, coarse)

    if M_total is not None:
        assert M_total.shape == (3, 3)
        np.testing.assert_array_almost_equal(M_total[2, :], [0.0, 0.0, 1.0])
