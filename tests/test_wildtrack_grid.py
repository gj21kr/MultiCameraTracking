"""Unit tests for WILDTRACK positionID -> world (X, Y) conversion.

Boundary values are derived directly from the grid constants:
    GRID_W=480, GRID_H=1440, CELL_M=0.025, ORIGIN_X=-3.0, ORIGIN_Y=-9.0
    MAX_ID = 480 * 1440 - 1 = 691 199

All expected values in meters.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wildtrack_grid import (
    position_id_to_world,
    GRID_W,
    GRID_H,
    CELL_M,
    ORIGIN_X,
    ORIGIN_Y,
    MAX_ID,
)


# ── Boundary values ───────────────────────────────────────────────────────────

def test_origin_cell():
    """positionID=0 maps to the grid origin (-3.0, -9.0) m."""
    x, y = position_id_to_world(0)
    assert x == pytest.approx(-3.0, abs=1e-9)
    assert y == pytest.approx(-9.0, abs=1e-9)


def test_last_column_first_row():
    """positionID=479 -> last column, first row: X=8.975, Y=-9.0."""
    # col = 479 % 480 = 479   ->  X = -3.0 + 0.025 * 479 = 8.975
    # row = 479 // 480 = 0    ->  Y = -9.0
    x, y = position_id_to_world(GRID_W - 1)
    assert x == pytest.approx(ORIGIN_X + CELL_M * (GRID_W - 1), abs=1e-9)
    assert y == pytest.approx(ORIGIN_Y, abs=1e-9)


def test_first_column_second_row():
    """positionID=480 -> first column, second row: X=-3.0, Y=-8.975."""
    # col = 480 % 480 = 0     ->  X = -3.0
    # row = 480 // 480 = 1    ->  Y = -9.0 + 0.025 * 1 = -8.975
    x, y = position_id_to_world(GRID_W)
    assert x == pytest.approx(ORIGIN_X, abs=1e-9)
    assert y == pytest.approx(ORIGIN_Y + CELL_M * 1, abs=1e-9)


def test_max_id_top_right_corner():
    """positionID=691199 maps to the top-right corner of the grid."""
    # col = 691199 % 480 = 479   ->  X = -3.0 + 0.025 * 479 =  8.975
    # row = 691199 // 480 = 1439 ->  Y = -9.0 + 0.025 * 1439 = 26.975
    x, y = position_id_to_world(MAX_ID)
    assert MAX_ID == GRID_W * GRID_H - 1
    assert x == pytest.approx(ORIGIN_X + CELL_M * (GRID_W - 1), abs=1e-9)
    assert y == pytest.approx(ORIGIN_Y + CELL_M * (GRID_H - 1), abs=1e-9)


# ── A sample interior point from the dataset's first frame ───────────────────

def test_sample_interior_point():
    """positionID=456826 (from 00000000.json, first person) round-trips."""
    # col = 456826 % 480 = 106  ->  X = -3.0 + 0.025 * 106 =  -0.35
    # row = 456826 // 480 = 951 ->  Y = -9.0 + 0.025 * 951 = 14.775
    pid = 456826
    x, y = position_id_to_world(pid)
    assert x == pytest.approx(ORIGIN_X + CELL_M * (pid % GRID_W), abs=1e-9)
    assert y == pytest.approx(ORIGIN_Y + CELL_M * (pid // GRID_W), abs=1e-9)


# ── Out-of-range clamping ─────────────────────────────────────────────────────

def test_negative_id_clamped(caplog):
    """Negative positionID is clamped to 0 with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.data.wildtrack_grid"):
        x, y = position_id_to_world(-1)

    assert x == pytest.approx(ORIGIN_X, abs=1e-9)
    assert y == pytest.approx(ORIGIN_Y, abs=1e-9)
    assert any("clamping" in rec.message for rec in caplog.records)


def test_overflow_id_clamped(caplog):
    """positionID > MAX_ID is clamped to MAX_ID with a warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.data.wildtrack_grid"):
        x, y = position_id_to_world(MAX_ID + 1)

    expected_x, expected_y = position_id_to_world(MAX_ID)
    assert x == pytest.approx(expected_x, abs=1e-9)
    assert y == pytest.approx(expected_y, abs=1e-9)
    assert any("clamping" in rec.message for rec in caplog.records)


# ── Type safety ───────────────────────────────────────────────────────────────

def test_float_input_raises_type_error():
    """Passing a float raises TypeError."""
    with pytest.raises(TypeError):
        position_id_to_world(100.0)  # type: ignore[arg-type]


def test_string_input_raises_type_error():
    """Passing a string raises TypeError."""
    with pytest.raises(TypeError):
        position_id_to_world("0")  # type: ignore[arg-type]


# ── Constants sanity ──────────────────────────────────────────────────────────

def test_constants():
    """Verify the documented grid constants."""
    assert GRID_W == 480
    assert GRID_H == 1440
    assert CELL_M == pytest.approx(0.025)
    assert ORIGIN_X == pytest.approx(-3.0)
    assert ORIGIN_Y == pytest.approx(-9.0)
    assert MAX_ID == GRID_W * GRID_H - 1
