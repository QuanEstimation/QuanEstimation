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
    click [here](https://mirror.tuna.tsinghua.edu.cn/help/julia/) for usage.

---

# **Requirements**
QuanEstimation requires several open-source packages in Python and Julia. The versions 
of Python and Julia should be above 3.10 and 1.10, respectively.
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
corresponding packages in Julia 1.10. Besides, the version information of the full Julia package 
is also the same as the table above. All of these packages will be automatically installed when
the users install QuanEstimation.
