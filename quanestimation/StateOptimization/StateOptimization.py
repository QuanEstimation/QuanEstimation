import numpy as np
import warnings
import math
import quanestimation.StateOptimization as stateoptimize

def StateOpt(*args, method = 'AD', **kwargs):

    if method == 'AD':
        return stateoptimize.StateOpt_AD(*args, **kwargs)
    elif method == 'PSO':
        return stateoptimize.StateOpt_PSO(*args, **kwargs)
    elif method == 'DE':
        return stateoptimize.StateOpt_DE(*args, **kwargs)
    elif method == 'NM':
        return stateoptimize.StateOpt_NM(*args, **kwargs)
    elif method == 'DDPG':
        return stateoptimize.StateOpt_DDPG(*args, **kwargs)
    else:
        raise ValueError("{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE', 'DDPG'.".format(method))