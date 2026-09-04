"""DN-WTA learning policy (marl): CTDE set-attention actor.

Public modules:
    perceive.M1  perception reconstruction + feature packing
    network.M2/3 set-attention encoder + decision head
    masking.M4   feasibility masking
    policy       MarlPolicy (dn_policies interface adapter)
    reward.M5    per-step team reward + kill credit
    train.M6     CTDE training CLI (MAPPO-style PPO)
"""

__version__ = "0.1.0"

from .perceive import AgentMemory, build_inputs          # noqa: F401
from .network import MarlNet                              # noqa: F401
from .masking import feasible_mask                        # noqa: F401
