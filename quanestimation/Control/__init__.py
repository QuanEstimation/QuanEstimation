from quanestimation.Control import DDPG
from quanestimation.Control import DiffEvo
from quanestimation.Control import GRAPE
from quanestimation.Control import PSO

from quanestimation.Control.DDPG import (DDPG, ddpg_actor, ddpg_critic,)
from quanestimation.Control.DiffEvo import (DiffEvo,)
from quanestimation.Control.GRAPE import (GRAPE,)
from quanestimation.Control.PSO import (PSO,)
from quanestimation.Control.Control import (control, )

__all__ = ['control','DDPG', 'DiffEvo', 'GRAPE', 'PSO', 'ddpg_actor', 'ddpg_critic']
