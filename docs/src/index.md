# **Welcome to the QuanEstimation**
QuanEstimation is an open-source Python–Julia toolkit for quantum parameter estimation. It provides calculations of quantum
metrological tools and quantum resources, optimizations over probe states, controls, and measurements, and comprehensive
optimizations in quantum metrology. QuanEstimation can generate both optimal quantum parameter estimation schemes and
adaptive measurement schemes. The package structure of QuanEstimation is shown in the following figure:

<figure markdown>
  ![Image title](Fig_schematic.png){ width="800" }
  <figcaption>The package structure of QuanEstimation. The blue and light blue boxes represent folders
  and Python files. The orange and gray boxes represent classes and functions. Functions circled by dotted
  lines represent Julia methods wrapped by the Python interface and evaluated in Julia.
  </figcaption>
</figure>

The package contains several widely used metrological tools, such as quantum (classical)
Cramér-Rao bounds, the Holevo Cramér-Rao bound and Bayesian versions of quantum (classical)
Cramér-Rao bounds, the quantum Ziv-Zakai bound, and Bayesian estimation. Users can use these
bounds as objective functions to optimize the probe state, control, and measurement, as well as
to perform simultaneous optimizations over them. The optimization methods include gradient-based
algorithms such as the gradient ascent pulse engineering (GRAPE), GRAPE based on
automatic differentiation (auto-GRAPE), and automatic differentiation (AD), as well as
gradient-free algorithms such as particle swarm optimization (PSO), differential evolution (DE),
and the reverse iterative (RI) algorithm.

The Python interface wraps the Julia core: most numerical work is executed in Julia for
computational efficiency. The same core is also available through the native Julia interface,
QuanEstimation.jl.

## **Julia-first / thin-shell direction**

The long-term architecture is Julia-first: numerical kernels live in
QuanEstimationBase, and Python remains a thin layer for input validation,
data conversion, and convenience APIs. The core Fisher-information kernels
(``QFIM``/``CFIM``/``SLD``/``RLD``/``LLD``/``FIM``) are delegated to Julia in
0.5; specialized kernels and the optimization wrappers will follow.
New numerical features should be implemented in Julia first. This keeps the
Python and Julia results on the same numerical implementation and avoids
duplicating numerical policy in Python.

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
