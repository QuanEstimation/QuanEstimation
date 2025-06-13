# QuanEstimation

![GitHub release (latest by date)](https://img.shields.io/github/v/release/QuanEstimation/QuanEstimation)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Downloads](https://static.pepy.tech/badge/quanestimation)](https://pepy.tech/project/quanestimation)

QuanEstimation is a Python-Julia-based open-source toolkit for quantum parameter estimation, which can be used to perform 
general evaluations of many metrological tools and scheme designs in quantum parameter estimation. 


## Documentation
[![Docs](https://github.com/QuanEstimation/QuanEstimation/actions/workflows/gh-deploy.yml/badge.svg)](https://github.com/QuanEstimation/QuanEstimation/actions/workflows/gh-deploy.yml)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://quanestimation.github.io/QuanEstimation/) 

The documentation of QuanEstimation can be found [here](https://quanestimation.github.io/QuanEstimation/).

## Installation
![PyPI](https://img.shields.io/pypi/v/QuanEstimation)

1. Install QuanEstimation via pip:  

~~~
pip install quanestimation
~~~

2. Download the package and install it in the terminal: 

~~~
git clone https://github.com/QuanEstimation/QuanEstimation.git
~~~
~~~
cd QuanEstimation
~~~
~~~
pip install .
~~~

## Citation
* If you use QuanEstimation in your research, please cite the following papers:

  [1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu, 
  QuanEstimation: An open-source toolkit for quantum parameter estimation, 
  [Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

  [2] H.-M. Yu and J. Liu, QuanEstimation.jl: An open-source Julia framework for quantum parameter estimation, 
  [Fundam. Res. (2025).](https://doi.org/10.1016/j.fmre.2025.02.020) 

* Development of the GRAPE algorithm in quantum parameter estimation can be found in the following papers:

  * **auto-GRAPE**:

    M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu, 
    QuanEstimation: An open-source toolkit for quantum parameter estimation, 
    [Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

  * **GRAPE for single-parameter estimation**:

    J. Liu and H. Yuan, Quantum parameter estimation with optimal control, 
    [Phys. Rev. A **96**, 012117 (2017).](https://doi.org/10.1103/PhysRevA.96.012117)

  * **GRAPE for multiparameter estimation**:

    J. Liu and H. Yuan, Control-enhanced multiparameter quantum estimation, 
    [Phys. Rev. A **96**, 042114 (2017).](https://doi.org/10.1103/PhysRevA.96.042114)
