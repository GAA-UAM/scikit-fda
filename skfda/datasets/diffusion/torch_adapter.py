import numpy as np
import torch
from skfda.misc.validation import validate_random_state
from skfda.typing._base import RandomStateLike

def make_torch_generator(
    random_state: RandomStateLike = None,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    rs = validate_random_state(random_state)

    if isinstance(rs, np.random.Generator):
        seed = int(rs.integers(0, 2**63 - 1))
    else:
        seed = int(rs.randint(0, 2**63 - 1))

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g