# **Installation**
Run the command in the terminal to install QuanEstimation:  
=== "Python"
    ```
    pip install quanestimation
    ```
    The users can also run 
    ```
    git clone https://github.com/QuanEstimation/QuanEstimation.git
    ```
    to install the latest development version of QuanEstimation from Github.

=== "Julia"
    ``` jl
    import Pkg
    Pkg.add("QuanEstimation")
    ```
    If the users want to install the package via a Julia mirror, please 
    click [here](https://help.mirrors.cernet.edu.cn/julia/) for usage.

---

# **Requirements**
QuanEstimation requires Python >= 3.10 and Julia >= 1.11. The Julia core is
pinned to QuanEstimation 0.3.x; the complete dependency list is declared in the
Julia package's `Project.toml` and is installed automatically by
`juliacall`/`juliapkg`.

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

All Python and Julia dependencies are installed automatically when the user
installs QuanEstimation.

## **Next steps**

[Getting started](guide/guide_import.md) — import QuanEstimation and understand the Python-Julia bridge.
