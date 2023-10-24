from typing import TYPE_CHECKING

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_functions": [
            "average_function_value",
        ],
        "_ndfunction": [
            "NDFunction", "concatenate",
        ],
    },
)

if TYPE_CHECKING:

    from ._functions import (
        average_function_value as average_function_value,
    )
    from ._ndfunction import (
        NDFunction as NDFunction,
        concatenate as concatenate,
    )
