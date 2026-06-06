"""WILDTRACK ground-plane grid utilities — positionID <-> world (X, Y) in meters.

WILDTRACK encodes each pedestrian's foot-point on the ground plane as a
``positionID`` — an integer index into a 480 × 1440 grid with 2.5 cm
resolution.  The grid spans the physical area [-3, 9) m in X and [-9, 27) m
in Y (12 m × 36 m), with the origin at the south-west corner of the scene.

Grid layout (verified against reprojection onto all seven views — axis-swap
causes 1504 px reprojection error; correct formula yields ~22 px):

    column = positionID %  480   ->  X axis  (east,   12 m, 480 cells)
    row    = positionID // 480   ->  Y axis  (north,  36 m, 1440 cells)

Constants:
    GRID_W   : number of cells in the X direction (columns)
    GRID_H   : number of cells in the Y direction (rows)
    CELL_M   : physical size of one cell in meters (2.5 cm)
    ORIGIN_X : world X of the grid's (0, 0) cell centre [m]
    ORIGIN_Y : world Y of the grid's (0, 0) cell centre [m]
    MAX_ID   : maximum valid positionID = GRID_W * GRID_H - 1

Reference: Chavdarova et al., "WILDTRACK: A Multi-Camera HD Dataset for
Dense Unscripted Pedestrian Detection", CVPR 2018.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Grid constants (do not change without re-verifying reprojection) ──────────
GRID_W: int = 480          # cells in X direction (column count)
GRID_H: int = 1440         # cells in Y direction (row count)
CELL_M: float = 0.025      # meters per cell  (2.5 cm)
ORIGIN_X: float = -3.0     # world X at column 0 [m]
ORIGIN_Y: float = -9.0     # world Y at row    0 [m]

MAX_ID: int = GRID_W * GRID_H - 1   # = 691 199


def position_id_to_world(position_id: int) -> Tuple[float, float]:
    """Convert a WILDTRACK positionID to world coordinates in meters.

    The transformation is:
        X [m] = ORIGIN_X + CELL_M * (positionID % GRID_W)
        Y [m] = ORIGIN_Y + CELL_M * (positionID // GRID_W)

    Args:
        position_id: Integer in [0, MAX_ID] (= GRID_W * GRID_H - 1 = 691 199).
            Values outside this range are clamped with a warning.

    Returns:
        (X, Y) world coordinates in meters.

    Raises:
        TypeError: If ``position_id`` is not an integer type.
    """
    if not isinstance(position_id, (int, np.integer)):
        raise TypeError(
            f"position_id must be an integer, got {type(position_id).__name__}"
        )

    position_id = int(position_id)

    if not (0 <= position_id <= MAX_ID):
        logger.warning(
            "position_id %d is outside the valid range [0, %d]; clamping.",
            position_id,
            MAX_ID,
        )
        position_id = max(0, min(position_id, MAX_ID))

    col = position_id % GRID_W   # X index
    row = position_id // GRID_W  # Y index

    x_m: float = ORIGIN_X + CELL_M * col
    y_m: float = ORIGIN_Y + CELL_M * row

    return x_m, y_m
