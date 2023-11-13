"""Types and functions associated with the Python array API standard."""

from typing import TypeVar

import numpy as np
import numpy.typing as nptypes

DType = np.generic
DTypeT = TypeVar("DTypeT", bound=DType)
Array = nptypes.NDArray[DTypeT]
BoolDType = np.bool_
