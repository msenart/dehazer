"""Dehazer: single-image haze removal built on the Dark Channel Prior.

Exposes the core dehazing pipeline (:mod:`dehazer.core`) and its transmission-map
refinement methods; the desktop GUI lives in :mod:`dehazer.gui` and is launched via
``python -m dehazer``.
"""

from .core import dehaze, dehazer_data

__all__ = ["dehaze", "dehazer_data"]
