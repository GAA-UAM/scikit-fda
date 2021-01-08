"""Common types."""
from typing import Optional, Sequence, Tuple, Union

import numpy as np

DomainRange = Tuple[Tuple[float, float], ...]
DomainRangeLike = Union[
    DomainRange,
    Sequence[float],
    Sequence[Sequence[float]],
]

LabelTuple = Tuple[Optional[str], ...]
LabelTupleLike = Sequence[Optional[str]]

GridPoints = Tuple[np.ndarray, ...]
GridPointsLike = Sequence[np.ndarray]
