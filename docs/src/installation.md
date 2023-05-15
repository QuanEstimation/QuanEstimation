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

    P.S. Julia and the julia environment will be downloaded and precompiled automatically through an installation guide when the first time `quanestimation` is imported. However, if you want to install python version of QuanEstimation on **Windows** currently, please  try as follow to set up `pyjulia` after `pip install quanestimation` (also see [here](https://pyjulia.readthedocs.io/en/stable/installation.html) for instruction):  
    1. [Download julia](https://julialang.org/downloads/) and install. Or simply via `pip install jill` and `jill install`,  
    2. Inside the julia REPL, `using Pkg; Pkg.add("QuanEstimation")`,  
    3. In the python command line, `import julia; julia.install()` to initialize pyjulia,   
    4. After the julia and pyjulia are successfully set up, `import quanestimation`.
=== "Julia"
    ``` jl
    import Pkg
    Pkg.add("QuanEstimation")
    ```
    If the users want to install the package via a Julia mirror, please 
    click [here](https://mirror.tuna.tsinghua.edu.cn/help/julia/) for usage.

---

# **Requirements**
QuanEstimation requires several open-source packages in Python and Julia. The versions 
of Python and Julia should be above 3.6 and 1.7, respectively.
## **Python packages**
| $~~~~~~~~~~~$Package$~~~~~~~$| Version      |
| :----------:                 | :----------: |
| numpy                        | >=1.22       |
| sympy                        | >=1.10       |
| scipy                        | >=1.8        |
| cvxpy                        | >=1.2        |
| more-itertools               | >=8.12.0     |

## **Julia packages**
| $~~~~~~~~~~~~~~~~$Package$~~~~~~~~~~~~$| Version     |$~~~~~~~~~~~~~~~~$Package$~~~~~~~~~~~~$| Version     |
| :----------:                           | :---------: |:----------:                           | :---------: |
| LinearAlgebra                          | --          |BoundaryValueDiffEq                    | 2.7.2       |
| Zygote                                 | 0.6.37      |SCS                                    | 0.8.1       |
| Convex                                 | 0.14.18     |Trapz                                  | 2.0.3       |
| ReinforcementLearning                  | 0.10.0      |Interpolations                         | 0.13.5      |
| IntervalSets                           | 0.5.4       |SparseArrays                           | --          |
| Flux                                   | 0.12.4      |DelimitedFiles                         | --          |
| StatsBase                              | 0.33.16     |Random                                 | --          |
| Printf                                 | --          |StableRNGs                             | --          |
| Distributions                          | --          |QuadGK                                 | --          |
| DifferentialEquations                          | --          |                                   |             |

The version information of the packages without the version number is the same with the 
corresponding packages in Julia 1.7. Besides, the version information of the full Julia package 
is also the same as the table above. All of these packages will be automatically installed when
the users install QuanEstimation.
