
"""Top-level package for quanestimation."""

from quanestimation import AsymptoticBound
from quanestimation import Common
from quanestimation import Control
from quanestimation import Dynamics
from quanestimation import Resources
from quanestimation import StateOptimization

from quanestimation.AsymptoticBound import (CramerRao, Holevo, CFIM, LLD, QFIM, RLD, SLD, )
from quanestimation.Common import (common, mat_vec_convert, suN_generator, )
from quanestimation.Control import (ControlOpt, DDPG, DiffEvo, GRAPE, PSO, )
from quanestimation.Dynamics import (dynamics, Lindblad, )
from quanestimation.Resources import (Resources, )
from quanestimation.StateOptimization import (StateOpt_DE, StateOpt_PSO, StateOpt_NM, StateOpt_AD, StateOpt, StateOptSystem)

# import julia
from julia import Main

Main.include('quanestimation/JuliaSrc/QuanEstimation.jl')

__all__ = ['AsymptoticBound', 'Common', 'Control', 'Dynamics', 'Resources', 
            'StateOpt_DE', 'StateOpt_PSO', 'StateOpt_NM', 'StateOpt_AD','StateOpt', 'StateOptSystem',
            'CramerRao', 'Holevo', 'common', 'DDPG', 'DiffEvo',  'GRAPE', 'PSO', 'dynamics', 'Lindblad', 'Resources',
            'ControlOpt', 'DDPG', 
            'CFIM', 'SLD', 'LLD', 'RLD', 'QFIM', 'mat_vec_convert', 'suN_generator']
