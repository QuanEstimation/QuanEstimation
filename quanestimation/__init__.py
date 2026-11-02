"""QuanEstimation — quantum parameter estimation.

Thin re-export wrapper: loads QuanEstimationBase (core) and
NVMagnetometer (extension) from their respective subpackages.
"""

__version__ = "0.4.0"

from quanestimation.base import *  # noqa: F403, F401, E402
from quanestimation.nv import *    # noqa: F403, F401, E402
