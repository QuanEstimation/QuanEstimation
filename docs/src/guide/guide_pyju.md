# Python-Julia calling patterns

This page documents the internal bridge between Python and Julia. Most users
do **not** need this — the standard `from quanestimation import *` provides the
public API. This reference is for advanced use cases that require direct
interaction with the Julia runtime.

## How the bridge works

Python is a compatibility layer over Julia. Numerical work such as dynamics
evolution, Fisher-information kernels, and optimization algorithms is
forwarded to Julia; Python handles input validation, data conversion, and
legacy return shapes. In most cases you pass Python data (NumPy arrays, lists,
strings) and get Python results back.

## Direct Julia access

The `QJL` proxy provides access to the loaded QuanEstimation Julia module:
``` py
from quanestimation.base import QJL
```
`QJL` is initialized on first access and cached, so all modules share the same
Julia module.

## Calling patterns

### Constructing Julia objects

The standard Python classes construct the corresponding Julia objects
internally. For example, the Python `Lindblad` wrapper takes Python data and
delegates the evolution to Julia:
``` py
from quanestimation import Lindblad

dynamics = Lindblad(tspan, rho0, H0, dH)
rho, drho = dynamics.expm()
```
To construct a Julia object directly, use `QJL`:
``` py
from quanestimation.base import QJL

dynamics_jl = QJL.Lindblad(H0, dH, tspan)
scheme = QJL.GeneralScheme(probe=rho0, param=dynamics_jl)
```

### Bang functions

Julia functions ending in `!` (in-place mutation) are not valid Python
identifiers. Use `getattr` to call them:
``` py
import quanestimation as qe
getattr(qe.QJL, "optimize!")(scheme, opt, algorithm=alg, objective=obj,
                             savefile=False)
```

### Explicit type conversion

When Julia cannot infer types from Python data (empty lists, ambiguous shapes,
complex numbers), use `juliacall.convert`:
``` py
from juliacall import Main as jl, convert as jlconvert
ctrl_bound = jlconvert(jl.Vector[jl.Float64], [0.0, 1.0])
```

### Return values

Optimization results are written by the Julia core to JLD2 `.dat` files
(for example `controls.dat`, `states.dat`, and `measurements.dat`). The Python
`load_save` helpers in the scheme classes convert them to `.npy` files and
remove the temporary `.dat` file. Functions that return values directly are
converted to NumPy arrays by `_unwrap()` or by the kernel wrappers.
