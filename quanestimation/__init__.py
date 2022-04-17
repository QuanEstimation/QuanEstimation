"""Top-level package for quanestimation."""
import os
from julia import Main

__version__ = "0.1.0"

from quanestimation.AsymptoticBound.CramerRao import (
    CFIM,
    QFIM,
    QFIM_Bloch,
    QFIM_Gauss,
    QFIM_Kraus,
    FIM,
    LLD,
    RLD,
    SLD,
)
from quanestimation.AsymptoticBound.Holevo import (
    HCRB,
)
from quanestimation.BayesianBound.BayesianCramerRao import (
    BCFIM,
    BQFIM,
    BCRB,
    BQCRB,
    QVTB,
    VTB,
    OBB,
)
from quanestimation.BayesianBound.ZivZakai import (
    QZZB,
)
from quanestimation.BayesianBound.BayesEstimation import (
    Bayes,
    MLE,
)

from quanestimation.Common.common import (
    mat_vec_convert,
    suN_generator,
    gramschmidt,
    basis,
    SIC,
    annihilation,
    AdaptiveInput,
)

from quanestimation.ComprehensiveOpt.ComprehensiveStruct import (
    ComprehensiveSystem,
    ComprehensiveOpt,
)
from quanestimation.ComprehensiveOpt.AD_Compopt import (
    AD_Compopt,
)
from quanestimation.ComprehensiveOpt.DE_Compopt import (
    DE_Compopt,
)
from quanestimation.ComprehensiveOpt.PSO_Compopt import (
    PSO_Compopt,
)

from quanestimation.ControlOpt.ControlStruct import (
    ControlSystem,
    ControlOpt,
    csv2npy_controls,
)
from quanestimation.ControlOpt.GRAPE_Copt import (
    GRAPE_Copt,
)
from quanestimation.ControlOpt.DE_Copt import (
    DE_Copt,
)
from quanestimation.ControlOpt.PSO_Copt import (
    PSO_Copt,
)
from quanestimation.ControlOpt.DDPG_Copt import (
    DDPG_Copt,
)

from quanestimation.Dynamics.dynamics import (
    Lindblad, Kraus, 
)

from quanestimation.MeasurementOpt.MeasurementStruct import (
    MeasurementSystem,
    MeasurementOpt,
    csv2npy_measurements,
)
from quanestimation.MeasurementOpt.AD_Mopt import (
    AD_Mopt,
)
from quanestimation.MeasurementOpt.PSO_Mopt import (
    PSO_Mopt,
)
from quanestimation.MeasurementOpt.DE_Mopt import (
    DE_Mopt,
)

from quanestimation.Resources.Resources import (
    SpinSqueezing,
    TargetTime,
)

from quanestimation.StateOpt.StateStruct import (
    StateSystem,
    StateOpt,
    csv2npy_states,
)
from quanestimation.StateOpt.AD_Sopt import (
    AD_Sopt,
)
from quanestimation.StateOpt.DE_Sopt import (
    DE_Sopt,
)
from quanestimation.StateOpt.PSO_Sopt import (
    PSO_Sopt,
)
from quanestimation.StateOpt.DDPG_Sopt import (
    DDPG_Sopt,
)
from quanestimation.StateOpt.NM_Sopt import (
    NM_Sopt,
)

from quanestimation.Adaptive.adaptive import adaptive
from quanestimation.Adaptive.adaptMZI import adaptMZI

pkgpath = os.path.abspath(os.path.dirname(__file__))
Main.include(pkgpath + "/JuliaSrc/QuanEstimation.jl")

__all__ = [
    "ControlOpt",
    "StateOpt",
    "MeasurementOpt",
    "ComprehensiveOpt",
    "CFIM",
    "QFIM",
    "QFIM_Bloch",
    "LLD",
    "RLD",
    "SLD",
    "HCRB",
    "QFIM_Gauss",
    "QFIM_Kraus",
    "FIM",
    "BCFIM",
    "BQFIM",
    "BCRB",
    "BQCRB",
    "OBB",
    "QVTB",
    "VTB",
    "QZZB",
    "Bayes",
    "MLE",
    "Lindblad",
    "Kraus",
    "SpinSqueezing",
    "TargetTime",
    "GRAPE_Copt",
    "DE_Copt",
    "PSO_Copt",
    "DDPG_Copt",
    "AD_Mopt",
    "PSO_Mopt",
    "DE_Mopt",
    "AD_Sopt",
    "DE_Sopt",
    "PSO_Sopt",
    "DDPG_Sopt",
    "NM_Sopt",
    "mat_vec_convert",
    "suN_generator",
    "gramschmidt",
    "basis",
    "SIC",
    "annihilation",
    "AdaptiveInput",
    "csv2npy_controls",
    "csv2npy_states",
    "csv2npy_measurements",
    "AD_Compopt",
    "DE_Compopt",
    "PSO_Compopt",
    "adaptive",
    "adaptMZI",
]
