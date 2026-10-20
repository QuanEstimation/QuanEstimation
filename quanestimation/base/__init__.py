"""Top-level package for quanestimation."""
__version__ = "0.4.0"

_QJL = None

def __getattr__(name):
    if name == "QJL":
        global _QJL
        if _QJL is None:
            from .Common.Common import load_julia
            _QJL = load_julia()
        return _QJL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from quanestimation.base.AsymptoticBound.CramerRao import (
    CFIM,
    QFIM,
    QFIM_Bloch,
    QFIM_Gauss,
    QFIM_Kraus,
    QFIM_pure,
    Williamson_form,
    FIM,
    FI_Expt,
    LLD,
    RLD,
    SLD,
)
from quanestimation.base.AsymptoticBound.AnalogCramerRao import (
    HCRB, NHB, 
)
from quanestimation.base.BayesianBound.BayesCramerRao import (
    BCFIM,
    BQFIM,
    BCRB,
    BQCRB,
    QVTB,
    VTB,
    OBB,
)
from quanestimation.base.BayesianBound.ZivZakai import (
    QZZB,
)
from quanestimation.base.BayesianBound.BayesEstimation import (
    Bayes,
    MLE,
    BCB,
    BayesCost
)

from quanestimation.base.Common.Common import (
    load_julia,
    mat_vec_convert,
    suN_generator,
    gramschmidt,
    basis,
    SIC,
    annihilation,
    BayesInput,
    fidelity,
    BellState,
    PlusState,
    MinusState,
    SigmaX,
    SigmaY,
    SigmaZ,
    sx,
    sy,
    sz,
    error_evaluation,
    error_control,
)

from quanestimation.base.ComprehensiveOpt.ComprehensiveStruct import (
    ComprehensiveSystem,
    ComprehensiveOpt,
)
from quanestimation.base.ComprehensiveOpt.AD_Compopt import (
    AD_Compopt,
)
from quanestimation.base.ComprehensiveOpt.DE_Compopt import (
    DE_Compopt,
)
from quanestimation.base.ComprehensiveOpt.PSO_Compopt import (
    PSO_Compopt,
)

from quanestimation.base.ControlOpt.ControlStruct import (
    ControlSystem,
    ControlOpt,
    csv2npy_controls,
)
from quanestimation.base.ControlOpt.GRAPE_Copt import (
    GRAPE_Copt,
)
from quanestimation.base.ControlOpt.DE_Copt import (
    DE_Copt,
)
from quanestimation.base.ControlOpt.PSO_Copt import (
    PSO_Copt,
)

from quanestimation.base.Parameterization.GeneralDynamics import (
    Lindblad,
    QubitDephasing,
)
from quanestimation.base.Parameterization.NonDynamics import (
    Kraus,
)
from quanestimation.base.Parameterization.ControlWaveform import (
    ZeroCTRL,
    LinearCTRL,
    SineCTRL,
    SawCTRL,
    TriangleCTRL,
    GaussianCTRL,
    GaussianEdgeCTRL,
)

from quanestimation.base.MeasurementOpt.MeasurementStruct import (
    MeasurementSystem,
    MeasurementOpt,
    csv2npy_measurements,
)
from quanestimation.base.MeasurementOpt.AD_Mopt import (
    AD_Mopt,
)
from quanestimation.base.MeasurementOpt.PSO_Mopt import (
    PSO_Mopt,
)
from quanestimation.base.MeasurementOpt.DE_Mopt import (
    DE_Mopt,
)

from quanestimation.base.Resource.Resource import (
    SpinSqueezing,
    TargetTime,
)

from quanestimation.base.StateOpt.StateStruct import (
    StateSystem,
    StateOpt,
    csv2npy_states,
)
from quanestimation.base.StateOpt.AD_Sopt import (
    AD_Sopt,
)
from quanestimation.base.StateOpt.DE_Sopt import (
    DE_Sopt,
)
from quanestimation.base.StateOpt.PSO_Sopt import (
    PSO_Sopt,
)
from quanestimation.base.StateOpt.NM_Sopt import (
    NM_Sopt,
)
from quanestimation.base.StateOpt.RI_Sopt import (
    RI_Sopt,
)

from quanestimation.base.AdaptiveScheme.Adapt import Adapt
from quanestimation.base.AdaptiveScheme.Adapt_MZI import Adapt_MZI


__all__ = [
    "ControlSystem",
    "ControlOpt",
    "StateSystem",
    "StateOpt",
    "MeasurementSystem",
    "MeasurementOpt",
    "ComprehensiveSystem",
    "ComprehensiveOpt",
    "CFIM",
    "QFIM",
    "QFIM_Bloch",
    "LLD",
    "RLD",
    "SLD",
    "HCRB",
    "NHB",
    "QFIM_Gauss",
    "QFIM_Kraus",
    "QFIM_pure",
    "Williamson_form",
    "FIM",
    "FI_Expt",
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
    "BCB",
    "BayesCost",
    "Lindblad",
    "Kraus",
    "QubitDephasing",
    "ZeroCTRL",
    "LinearCTRL",
    "SineCTRL",
    "SawCTRL",
    "TriangleCTRL",
    "GaussianCTRL",
    "GaussianEdgeCTRL",
    "SpinSqueezing",
    "TargetTime",
    "GRAPE_Copt",
    "DE_Copt",
    "PSO_Copt",
    "AD_Mopt",
    "PSO_Mopt",
    "DE_Mopt",
    "AD_Sopt",
    "DE_Sopt",
    "PSO_Sopt",
    "NM_Sopt",
    "RI_Sopt",
    "mat_vec_convert",
    "suN_generator",
    "gramschmidt",
    "basis",
    "SIC",
    "annihilation",
    "BayesInput",
    "csv2npy_controls",
    "csv2npy_states",
    "csv2npy_measurements",
    "AD_Compopt",
    "DE_Compopt",
    "PSO_Compopt",
    "Adapt",
    "Adapt_MZI",
    "load_julia",
    "fidelity",
    "BellState",
    "PlusState",
    "MinusState",
    "SigmaX",
    "SigmaY",
    "SigmaZ",
    "sx",
    "sy",
    "sz",
    "error_evaluation",
    "error_control",
]
