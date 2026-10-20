# **Welcome to the QuanEstimation**
QuanEstimation is a Python-Julia-based open-source toolkit for quantum parameter estimation, which consist in the calculation of 
the quantum metrological tools and quantum resources, and the optimizations with respect to the probe states, the controls or 
the measurements, as well as comprehensive optimizations in quantum metrology. Furthermore, QuanEstimation can generate not only 
optimal quantum parameter estimation schemes, but also adaptive measurement schemes.The package structure of QuanEstimation can 
be seen in the following figure:

<figure markdown>
  ![Image title](Fig_schematic.png){ width="800" }
  <figcaption>The package structure of QuanEstimation. The blue boxes and the light blue boxes represent
  the folders and the python files. The orange boxes and the gray boxes represent the classes and the 
  functions. In the figure, the functions which are circled by dotted lines represent the wrapped Julia 
  methods that are calculated in Julia.
  </figcaption>
</figure>

The package contains several well-used metrological tools, such as quantum (classical) 
Cramér-Rao bounds, Hevolo Cramér-Rao bound and Bayesian versions of quantum (classical) 
Cramér-Rao bounds, quantum Ziv-Zakai bound, and Bayesian estimation. The users can use these
bounds as the objective functions to optimize the probe state, control, measurement and 
simultaneous optimizations among them. The optimization methods include the gradient-based 
algorithms such as the gradient ascent pulse engineering (GRAPE), GRAPE algorithm based on the 
automatic differentiation (auto-GRAPE), automatic differentiation (AD) and the 
gradient-free algorithms such as particle swarm optimization (PSO), differential evolution (DE), 
and reverse iterative (RI) algorithm.

The interface of QuanEstimation is written in Python, but the most calculation processes are
executed in Julia for the computational efficiency. Therefore, QuanEstimation also has a full
Julia version apart from the Python-Julia version. 

## **Julia-first / thin-shell direction**

The long-term architecture is Julia-first: numerical kernels live in
QuanEstimationBase, and Python remains a thin layer for input validation,
data conversion and convenience APIs. Some legacy Python-side numerical code
is kept for backward compatibility in 0.4; the core QFIM/CFIM kernels will be
delegated in 0.5. New numerical features should be implemented in Julia first.
This keeps the Python and Julia results identical and lets the Python tests
focus on the bridge rather than duplicating numerical stability checks.

## **Quick links**

- [Installation](installation.md)
- [Getting started](guide/guide_import.md) — import the package and understand the bridge
- [Parameterization process](guide/guide_dynamics.md) — define system dynamics
- [Quantum metrological tools](guide/guide_bounds.md) — compute bounds and perform estimation
- [Control optimization](guide/guide_Copt.md)
- [State optimization](guide/guide_Sopt.md)
- [Measurement optimization](guide/guide_Mopt.md)
- [Comprehensive optimization](guide/guide_Compopt.md)
- [Adaptive measurement schemes](guide/guide_adaptive.md)

---