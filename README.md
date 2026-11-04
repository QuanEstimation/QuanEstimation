# QuanEstimation

![GitHub release (latest by date)](https://img.shields.io/github/v/release/QuanEstimation/QuanEstimation)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/QuanEstimation/QuanEstimation/actions/workflows/ci.yml/badge.svg)](https://github.com/QuanEstimation/QuanEstimation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuanEstimation/QuanEstimation/branch/main/graph/badge.svg?token=SSOSZ9OSBU)](https://codecov.io/gh/QuanEstimation/QuanEstimation)
[![Downloads](https://static.pepy.tech/badge/quanestimation)](https://pepy.tech/project/quanestimation)
[![Tutorial](https://img.shields.io/badge/Tutorial-Link-brightgreen)](https://github.com/QuanEstimation/tutorial/tree/main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16730593.svg)](https://doi.org/10.5281/zenodo.16730593)

QuanEstimation is an open-source Python–Julia toolkit for quantum parameter
estimation. It supports general evaluations of metrological tools and the
design of quantum estimation schemes.

## Documentation

[![Docs](https://github.com/QuanEstimation/QuanEstimation/actions/workflows/gh-deploy.yml/badge.svg)](https://github.com/QuanEstimation/QuanEstimation/actions/workflows/gh-deploy.yml)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://quanestimation.github.io/QuanEstimation/)

The documentation of QuanEstimation can be found [here](https://quanestimation.github.io/QuanEstimation/).
Docstrings generally follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings); some older modules still use NumPy-style sections.


## Installation

![PyPI](https://img.shields.io/pypi/v/QuanEstimation)

1. Install QuanEstimation via PyPI:

~~~
pip install quanestimation
~~~

2. Clone the repository and install from the checkout:

~~~
git clone https://github.com/QuanEstimation/QuanEstimation.git
~~~
~~~
cd QuanEstimation
~~~
~~~
pip install .
~~~

The Python package requires Python >= 3.10 and Julia 1.11 or newer in the 1.x
series, and pins the Julia core to QuanEstimation 0.5.0. The core is installed
and loaded through `juliacall` on first use. For local development against a
QuanEstimation.jl checkout, set `QuanEstimation_DEV_PATH` to that checkout
before the Julia core is first loaded; the bridge still checks that the
checkout reports version 0.5.0.

## Code Style

This project follows the [PEP 8](https://peps.python.org/pep-0008/) Python code
style guide. It is recommended to use tools such as black and flake8 for code
formatting and linting.

## Citation

* If you use QuanEstimation in your research, please cite:

  [1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu,
  QuanEstimation: An open-source toolkit for quantum parameter estimation,
  [Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

  [2] H.-M. Yu and J. Liu, Quanestimation.jl: An open-source Julia framework for quantum parameter estimation,
  [Fundam. Res. **6**, 2167-2174 (2026).](https://doi.org/10.1016/j.fmre.2025.02.020)

* Development of the GRAPE algorithm:

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

