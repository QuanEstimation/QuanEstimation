# **Installation**
Run the command in the terminal to install QuanEstimation:
=== "Python"
    ```
    pip install quanestimation
    ```
    To install the latest development version from GitHub, clone the
    repository and install it locally:
    ```
    git clone https://github.com/QuanEstimation/QuanEstimation.git
    cd QuanEstimation
    pip install .
    ```

=== "Julia"
    ``` jl
    import Pkg
    Pkg.add("QuanEstimation")
    ```
    If the users want to install the package via a Julia mirror, please
    click [here](https://help.mirrors.cernet.edu.cn/julia/) for usage.

---

# **Requirements**
QuanEstimation requires Python >= 3.10 and Julia 1.11 or newer in the 1.x
series. The Python package pins the Julia core to QuanEstimation 0.3.1. The
Julia core's dependency list is declared in its `Project.toml` and is installed
automatically by `juliacall`/`Pkg` when the bridge is first used.

## **Python packages**
| $~~~~~~~~~~~$Package$~~~~~~~$| Version      |
| :----------:                 | :----------: |
| numpy                        | >=1.21       |
| sympy                        | >=1.9        |
| scipy                        | >=1.7        |
| cvxpy                        | >=1.2        |
| juliacall                    | >=0.9.35     |
| more-itertools               | >=8.12       |
| h5py                         | >=3.6        |

Installing the Python package installs the Python dependencies listed above.
The pinned Julia core and its Julia dependencies are installed automatically
when the bridge is first used.

## **Next steps**

[Getting started](guide/guide_import.md) — import QuanEstimation and understand the Python-Julia bridge.
