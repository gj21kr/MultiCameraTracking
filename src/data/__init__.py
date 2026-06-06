"""Data sources for multi-camera tracking (file / image-sequence / WILDTRACK)."""

from .sources import (
    FrameSource,
    ImageSequenceSource,
    VideoFileSource,
    from_wildtrack,
    make_synthetic_source,
)
from .wildtrack_grid import (
    position_id_to_world,
    GRID_W,
    GRID_H,
    CELL_M,
    ORIGIN_X,
    ORIGIN_Y,
    MAX_ID,
)

__all__ = [
    "FrameSource",
    "ImageSequenceSource",
    "VideoFileSource",
    "from_wildtrack",
    "make_synthetic_source",
    "position_id_to_world",
    "GRID_W",
    "GRID_H",
    "CELL_M",
    "ORIGIN_X",
    "ORIGIN_Y",
    "MAX_ID",
]
