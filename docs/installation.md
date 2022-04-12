# **Installation**
Run the command in the terminal to install QuanEstimation:  
```
pip install quanestimation
```
The users can also run 
```
git clone https://github.com/QuanEstimation/QuanEstimation.git
```
to install the latest development version of QuanEstimation from Github.

---

# **Requirements**
QuanEstimation requires several open-source packages in Python and Julia. The versions 
of Python and Julia should be above 3.6 and 1.7, respectively.
## **Python packages**
| Package$~~~~~~~~~~~~~~~~~~~~~~~~~~~~$| Version     |
| ----------                           | ----------  |
| numpy                                | 1.22+       |
| sympy                                | 1.10+       |
| scipy                                | 1.8+        |
| cvxpy                                | 1.2+        |

QuanEstimation can be used in combination with other code packages such as QuTip, the users can 
install these packages as needed.

## **Julia packages**
| Package$~~~~~~~~~~~~~~~~~~~~~~~~~~~~$| Version     |
| ----------                           | ----------  |
| LinearAlgebra                        | -           |
| Zygote                               | 0.6.37      |
| Convex                               | 0.14.18     |
| ReinforcementLearning                | 0.10.0      |
| SparseArrays                         | -           |
| DelimitedFiles                       | -           |
| StatsBase                            | 0.33.16     |
| BoundaryValueDiffEq                  | 2.7.2       |
| Random                               | -           |
| Trapz                                | 2.0.3       |
| Interpolation                        | 0.13.5      |
| IntervalSets                         | 0.5.4       |
| Flux                                 | 0.12.4      |
| SCS                                  | 0.8.1       |

The packages without version are the same as Julia 1.7.
