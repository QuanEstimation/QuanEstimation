# **Getting started**

QuanEstimation provides interfaces in both Python and Julia. The two APIs are
closely related but not identical: Python wraps the Julia core in compatibility
classes and converts data at the boundary, so some signatures and defaults
differ. Use the examples on each page as the reference for that language.

=== "Python"
    ``` py
    from quanestimation import *
    ```
    All public functions and classes are imported into the current namespace.

=== "Julia"
    ``` jl
    using QuanEstimation
    ```

Importing `quanestimation` loads the public Python API and may initialize the
pinned Julia core when an imported wrapper requests it. Once initialized, the
Julia module is cached for subsequent calls.

For advanced usage (direct Julia access, bang functions, type conversion),
see [Python-Julia calling patterns](guide_pyju.md).

## **Next steps**

- [Parameterization process](guide_dynamics.md) — define the system dynamics
  (Hamiltonian, decay, Kraus operators)
- [Quantum metrological tools](guide_bounds.md) — compute Fisher information,
  Cramér-Rao bounds, and perform Bayesian estimation
