# **Output files**

QuanEstimation writes optimization and estimation results to files in the current working
directory. Julia users receive the `.dat` (JLD2) result files together with `f.csv`. The Python
interface converts the `.dat` files into `.npy` files after an optimization finishes; the
`.dat` files are removed once they have been read. The objective-function history is written
directly to `f.csv` by the Julia backend and is not converted.

## **Objective function values**

The objective function is evaluated once per optimization episode. The values are written
sequentially, one value per line, to `f.csv`.

=== "Python"
    ``` py
    import numpy as np
    f = np.loadtxt("f.csv")
    ```

## **Control optimization**

The Python wrapper saves the control optimization results in `controls.npy`. The shape
depends on `savefile`:

- `savefile=False` (default): the final control coefficients are saved with shape
  `(n_ctrl, n_ctrl_segments)`, where `n_ctrl` is the number of control Hamiltonians and
  `n_ctrl_segments` is the number of control segments.
- `savefile=True`: the control coefficients after every optimization round are saved with
  shape `(n_rounds, n_ctrl, n_ctrl_segments)`.

=== "Python"
    ``` py
    import numpy as np
    controls = np.load("controls.npy")
    ```

If the controls were exported as a flat CSV array, use `csv2npy_controls` to reshape them
and save `controls.npy`:
``` py
from quanestimation import csv2npy_controls
csv2npy_controls(controls_csv_data, num=n_ctrl_segments)
```

See also: [Control optimization](guide_Copt.md)

## **State optimization**

The Python wrapper saves the state optimization results in `states.npy`.

- `savefile=False` (default): the final optimized state vector is saved.
- `savefile=True`: the state vectors after every optimization round are saved.

=== "Python"
    ``` py
    import numpy as np
    states = np.load("states.npy")
    ```

If the states were exported as a flat CSV array, use `csv2npy_states` to regroup them and
save `states.npy`:
``` py
from quanestimation import csv2npy_states
csv2npy_states(states_csv_data, num=1)
```

See also: [State optimization](guide_Sopt.md)

## **Measurement optimization**

The Python wrapper saves the measurement optimization results in `measurements.npy`.

- `savefile=False` (default): the final optimized POVM is saved as a list of measurement
  operators.
- `savefile=True`: the POVM lists after every optimization round are saved.

=== "Python"
    ``` py
    import numpy as np
    M = np.load("measurements.npy", allow_pickle=True)
    ```

If the measurements were exported as a flat CSV array, use `csv2npy_measurements` to reshape
them and save `measurements.npy`:
``` py
from quanestimation import csv2npy_measurements
csv2npy_measurements(M_csv_data, num=n_operators)
```

See also: [Measurement optimization](guide_Mopt.md)

## **Comprehensive optimization**

Comprehensive optimization combines several variable types. Each result file follows the
same convention as the corresponding single-variable optimization:

- controls are saved in `controls.npy`;
- states are saved in `states.npy`;
- measurements are saved in `measurements.npy`.

A file is present only if that variable type was optimized. For example, SC optimization
produces `controls.npy` and `states.npy`, but not `measurements.npy`; SCM optimization
produces all three.

See also: [Comprehensive optimization](guide_Compopt.md)

## **Bayesian estimation**

`Bayes()` and `MLE()` save their results with `numpy.save`, so the files are read with
`numpy.load`.

### Maximum a posteriori / mean estimation

`Bayes()` produces:

| File | Contents | Shape |
|------|----------|-------|
| `pout.npy` | Posterior distribution. With `savefile=False` it is the final posterior; with `savefile=True` it contains the posterior after every iteration. | grid shape of `p`; `(n_iter, *grid_shape)` when `savefile=True` |
| `xout.npy` | Estimated parameter values for every iteration. | `(n_iter,)` for one parameter, `(n_iter, n_params)` for several parameters |

The return value of `Bayes()` is `(p, x_last)`: the final posterior distribution and the
estimate from the final iteration. The complete history is in `xout.npy`.

### Maximum likelihood estimation

`MLE()` produces:

| File | Contents | Shape |
|------|----------|-------|
| `Lout.npy` | Log-likelihood values. With `savefile=False` it is the final log-likelihood; with `savefile=True` it contains the log-likelihood after every iteration. | grid shape of the parameter grid; `(n_iter, *grid_shape)` when `savefile=True` |
| `xout.npy` | Estimated parameter values for every iteration. | `(n_iter,)` for one parameter, `(n_iter, n_params)` for several parameters |

=== "Python"
    ``` py
    import numpy as np
    pout = np.load("pout.npy")
    xout = np.load("xout.npy")
    ```

See also: [Quantum metrological tools](guide_bounds.md)

## **Adaptive measurement schemes**

The `Adapt` module writes CSV files. With `savefile=False`, `pout.csv` contains only the
final posterior distribution while `xout.csv` and `y.csv` contain the results of all
iterations. With `savefile=True`, a new row is appended for every iteration.

| File | Contents |
|------|----------|
| `pout.csv` | Posterior distribution; the final one, or one row per iteration when `savefile=True` |
| `xout.csv` | Estimated parameter value(s) at each iteration |
| `y.csv` | Experimental outcome at each iteration |

=== "Python"
    ``` py
    import numpy as np
    pout = np.loadtxt("pout.csv")
    xout = np.loadtxt("xout.csv")
    y = np.loadtxt("y.csv")
    ```

Offline optimization with `Adapt_MZI` additionally writes `deltaphi.csv` (the optimized
phase differences) and `f.csv` (the objective-function history).

See also: [Adaptive measurement schemes](guide_adaptive.md)
