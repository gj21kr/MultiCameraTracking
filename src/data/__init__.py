"""Data sources for multi-camera tracking (file / image-sequence / WILDTRACK)."""

from .sources import (
    FrameSource,
    ImageSequenceSource,
    VideoFileSource,
    from_wildtrack,
    make_synthetic_source,
)

__all__ = [
    "FrameSource",
    "ImageSequenceSource",
    "VideoFileSource",
    "from_wildtrack",
    "make_synthetic_source",
]
