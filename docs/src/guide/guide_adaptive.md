# **Adaptive measurement schemes**
In QuanEstimation, the Hamiltonian of the adaptive system should be written as
$H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$
the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the
Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. In this scenario,
the adaptive estimation can be executed through
=== "Python"
    ``` py
    apt = Adapt(x, p, rho0, method="FOP", savefile=False,
                max_episode=1000, eps=1e-8)
    apt.dynamics(tspan, H, dH, Hc=[], ctrl=[], decay=[],
                 dyn_method="expm")
    apt.CFIM()
    ```
    where `x` is a list of arrays representing the range of the parameters for the integral,
    `p` is an array representing the prior distribution; it is multidimensional for
    multiparameter estimation. `rho0` is the density matrix of the probe state. The number
    of iterations can be set via `max_episode` (default 1000). `eps` is the numerical
    precision threshold (default $10^{-8}$). At the end of the program, three CSV files
    "pout.csv", "xout.csv", and "y.csv" containing the posterior distributions, the estimated
    values, and the experimental results will be generated. The package contains two methods
    for updating the tunable parameters. The first one is updating the tunable parameters at
    a fixed optimal point (`method="FOP"`), which is the default method in QuanEstimation.
    The other is `method="MI"`, which updates the tunable parameters by maximizing the mutual
    information defined as

    \begin{equation}
    I(\textbf{u})=\int\mathrm{p}(\textbf{x}) \sum_{y}\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{log}_2
    \left[\frac{\mathrm{p}(y|\textbf{x},\textbf{u})}{\int\mathrm{p}(\textbf{x})\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{d}\textbf{x}}\right]\mathrm{d}\textbf{x}.
    \end{equation}

    If `savefile=True`, these files are written during the training and "pout.csv" saves the
    posterior distribution of every iteration. If `savefile=False`, "pout.csv" contains only
    the posterior distribution of the final iteration, while "xout.csv" and "y.csv" still
    contain the results of all iterations.
=== "Julia"
    ``` jl
    strategy = AdaptiveStrategy(x=x, p=p, dp=dp)
    dynamics = Lindblad(H0_func, dH_func, tspan; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics, strat=strategy)
    res = rand(0:1, max_episode)
    adapt!(scheme; method="FOP", max_episode=max_episode, res=res)
    ```
    where `x` is a list of arrays representing the range of the parameters for the integral,
    `p` is an array representing the prior distribution; it is multidimensional for
    multiparameter estimation. `rho0` is the density matrix of the probe state. The number
    of iterations can be set via `max_episode` (default 1000). `eps` is the numerical
    precision threshold (default $10^{-8}$). At the end of the program, the JLD2 file
    "adaptive.dat" is generated. It contains the posterior distributions (key `"p"`), the
    estimated values (key `"x"`), and the experimental results (key `"y"`). The package
    contains two methods for updating the tunable parameters. The first one is updating the
    tunable parameters at a fixed optimal point (`method="FOP"`), which is the default method
    in QuanEstimation. The other is `method="MI"`, which updates the tunable parameters by
    maximizing the mutual information defined as

    \begin{equation}
    I(\textbf{u})=\int\mathrm{p}(\textbf{x}) \sum_{y}\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{log}_2
    \left[\frac{\mathrm{p}(y|\textbf{x},\textbf{u})}{\int\mathrm{p}(\textbf{x})\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{d}\textbf{x}}\right]\mathrm{d}\textbf{x}.
    \end{equation}

    If `savefile=true`, the results are written during the training and `"p"` saves the
    posterior distribution of every iteration. If `savefile=false`, `"p"` contains only
    the posterior distribution of the final iteration, while `"x"` and `"y"` still
    contain the results of all iterations.
    ``` jl
    using JLD2
    pout = load("adaptive.dat")["p"]
    xout = load("adaptive.dat")["x"]
    y = load("adaptive.dat")["y"]
    ```
If the dynamics of the system can be described by the master equation, then the dynamics data
`tspan`, `H`, and `dH` should be input. `tspan` is the time grid for the evolution; `H` and
`dH` are multidimensional lists representing the Hamiltonian and its derivatives with respect
to the unknown parameters to be estimated. They can be generated via
=== "Python"
    ``` py
    H, dH = BayesInput(x, func, dfunc, channel="dynamics")
    ```
=== "Julia"
    ``` jl
    H, dH = BayesInput(x, func, dfunc; channel="dynamics")
    ```
Here `func` and `dfunc` are functions defined by the user which return `H` and `dH`,
respectively. `BayesInput` calls these functions at each point of the parameter grid, passing
the parameter values as separate positional arguments. Furthermore, for systems with noise
and controls, the variables `decay`, `Hc` and `ctrl` should be input. Here `Hc` and `ctrl` are
two lists representing the control Hamiltonians and the corresponding control coefficients.
`decay` contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay
rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$],
[$\Gamma_2$, $\gamma_2$],...].

The objective function for adaptive measurement is the CFI, or $\mathrm{Tr}(W\mathcal{I}^{-1})$
with $\mathcal{I}$ the CFIM. `W` is the weight matrix, which defaults to the identity matrix.

If the parameterization is implemented with the Kraus operators, the codes become
=== "Python"
    ``` py
    apt = Adapt(x, p, rho0, method="FOP", savefile=False,
                max_episode=1000, eps=1e-8)
    apt.Kraus(K, dK)
    apt.CFIM()
    ```
=== "Julia"
    Julia adaptive estimation currently supports Lindblad dynamics. Kraus-parameterized
    schemes can be used with the non-adaptive bounds, but `adapt!` does not provide a
    Kraus update rule.

and
=== "Python"
    ``` py
    K, dK = BayesInput(x, func, dfunc, channel="Kraus")
    ```
=== "Julia"
    ``` jl
    K, dK = BayesInput(x, func, dfunc; channel="Kraus")
    ```
where `K` and `dK` are the Kraus operators and their derivatives with respect to the unknown
parameters. `dK[i][a]` denotes the derivative of the `i`th Kraus operator with respect to
the `a`th parameter.

**Example 9.1**
The Hamiltonian of a qubit system is
\begin{align}
H=\frac{B\omega_0}{2}(\sigma_1\cos{x}+\sigma_3\sin{x}),
\end{align}

where $B$ is the magnetic field in the XZ plane, $x$ is the unknown parameter and $\sigma_{1}$, $\sigma_{3}$ are the Pauli matrices.

The probe state is taken as $|\pm\rangle$. The measurement is
$\{|\!+\rangle\langle+\!|,|\!-\rangle\langle-\!|\}$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ $(-1)$. In this example, the prior distribution $p(x)$ is uniform.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np
    import random

    # initial state
    rho0 = 0.5 * np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    B, omega0 = 0.5 * np.pi, 1.0
    sx = np.array([[0., 1.], [1., 0.]])
    sy = np.array([[0., -1.j], [1.j, 0.]])
    sz = np.array([[1., 0.], [0., -1.]])
    H0_func = lambda x: 0.5*B*omega0*(sx*np.cos(x)+sz*np.sin(x))
    # derivative of free Hamiltonian in x
    dH_func = lambda x: [0.5*B*omega0*(-sx*np.sin(x)+sz*np.cos(x))]
    # measurement
    M1 = 0.5*np.array([[1., 1.], [1., 1.]])
    M2 = 0.5*np.array([[1., -1.], [-1., 1.]])
    M = [M1, M2]
    # time length for the evolution
    tspan = np.linspace(0., 1., 1000)
    # prior distribution
    x = np.linspace(-0.25*np.pi+0.1, 3.0*np.pi/4.0-0.1, 1000)
    p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
    # dynamics
    rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for \
           i in range(len(x))]
    for xi in range(len(x)):
        H_tp = H0_func(x[xi])
        dH_tp = dH_func(x[xi])
        dynamics = Lindblad(tspan, rho0, H_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[xi] = rho_tp[-1]
    # Bayesian estimation
    random.seed(1234)
    y = [0 for i in range(500)]
    res_rand = random.sample(range(0, len(y)), 125)
    for i in range(len(res_rand)):
        y[res_rand[i]] = 1
    pout, xout = Bayes([x], p, rho, y, M=M, estimator="MAP", savefile=False)
    # generation of H and dH
    H, dH = BayesInput([x], H0_func, dH_func, channel="dynamics")
    # adaptive measurement
    apt = Adapt([x], pout, rho0, method="FOP", savefile=False,
                max_episode=100, eps=1e-8)
    apt.dynamics(tspan, H, dH, dyn_method="expm")
    apt.CFIM(M=M)
    ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using Random
    using StatsBase

    # free Hamiltonian
    function H0_func(x)
        return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
    end
    # derivative of free Hamiltonian in x
    function dH_func(x)
        return [0.5*B*omega0*(-sx*sin(x)+sz*cos(x))]
    end

    B, omega0 = pi/2.0, 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    # initial state
    rho0 = 0.5*ones(2, 2)
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
    M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    # time length for the evolution
    tspan = range(0., stop=1., length=1000) |>Vector
    # prior distribution
    x = range(-0.25*pi+0.1, stop=3.0*pi/4.0-0.1, length=1000) |>Vector
    p = (1.0/(x[end]-x[1]))*ones(length(x))
    # dynamics
    rho = Vector{Matrix{ComplexF64}}(undef, length(x))
    for i = 1:length(x)
        H0_tp = H0_func(x[i])
        dH_tp = dH_func(x[i])
        rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
        rho[i] = rho_tp[end]
    end
    # Bayesian estimation
    Random.seed!(1234)
    y = [0 for i in 1:500]
    res_rand = sample(1:length(y), 125, replace=false)
    for i in 1:length(res_rand)
        y[res_rand[i]] = 1
    end
    pout, xout = QuanEstimation.Bayes([x], p, rho, y, M=M, estimator="MAP", savefile=false)
    # adaptive measurement
    strategy = QuanEstimation.AdaptiveStrategy(x=[x], p=pout, dp=[zeros(length(x))])
    dynamics = QuanEstimation.Lindblad(H0_func, dH_func, tspan; dyn_method=:Expm)
    scheme = QuanEstimation.GeneralScheme(; probe=rho0, param=dynamics,
                                          measurement=M, strat=strategy)
    res = rand(0:1, 100)
    QuanEstimation.adapt!(scheme; method="FOP", max_episode=100, res=res)
    ```
---
Berry et al. [[1,2]](#Berry2000) introduced a famous adaptive scheme in phase estimation. The
phase for the $(n+1)$th round is updated via $\Phi_{n+1}=\Phi_{n}-(-1)^{y^{(n)}}\Delta
\Phi_{n+1}$ with $y^{(n)}$ the experimental result in the $n$th round and $\Delta\Phi_{n+1}$
the phase difference generated by the proper algorithms. This adaptive scheme can be performed
in QuanEstimation via
=== "Python"
    ``` py
    apt = Adapt_MZI(x, p, rho0)
    apt.general()
    apt.online(target="sharpness", output="phi")
    ```
    Here `x`, `p`, and `rho0` are the same as for `Adapt`. `target="sharpness"` represents the
    target function used to calculate the tunable phase; it can also be set as `target="MI"`,
    which means the target function is the mutual information. The output can be selected with
    `output="phi"` (default) and `output="dphi"`, representing the phase and the phase
    difference, respectively. Online and offline strategies are both available in the package;
    the code for the offline strategy is `apt.offline(method="DE", **kwargs)` or
    `apt.offline(method="PSO", **kwargs)`.
=== "Julia"
    ``` jl
    apt = Adapt_MZI(x, p, rho0)
    online(apt, target=:sharpness, output="phi")
    ```
    Here `x`, `p`, and `rho0` are the same as for `Adapt`. `target=:sharpness` represents the
    target function used to calculate the tunable phase; it can also be set as `target=:MI`,
    which means the target function is the mutual information. The output can be selected with
    `output="phi"` (default) and `output="dphi"`, representing the phase and the phase
    difference, respectively. Online and offline strategies are both available in the package;
    the code for the offline strategy is `alg = QuanEstimation.DE(kwargs...)`
    (`alg = QuanEstimation.PSO(kwargs...)`) followed by `offline(apt, alg, seed=seed)`.
    `seed` is the random seed which can ensure the reproducibility of results.

If the optimization algorithm is PSO, the keywords and the default values are
=== "Python"
    ``` py
    kwargs = {"p_num":10, "deltaphi0":[], "max_episode":1000,
              "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
    ```
    The keywords and the default values of PSO can be seen in the following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                   | 10                         |
    | "deltaphi0"                      | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "c0"                             | 1.0                        |
    | "c1"                             | 2.0                        |
    | "c2"                             | 2.0                        |
    | "seed"                           | 1234                       |

    Here `p_num` is the number of particles, `deltaphi0` represents the initial
    guesses of phase difference. `max_episode` accepts both integer and array with two
    elements. If it is an integer, for example `max_episode=1000`, it means the
    program will continuously run 1000 episodes. However, if it is an array, for example
    `max_episode=[1000,100]`, the program will run 1000 episodes in total but replace the data
    of all the particles with global best every 100 episodes. `c0`, `c1` and `c2` are the PSO
    parameters representing the inertia weight, cognitive learning factor and social
    learning factor, respectively.
=== "Julia"
    ``` jl
    alg = PSO(p_num=10, ini_particle=nothing, max_episode=[1000,100],
              c0=1.0, c1=2.0, c2=2.0)
    ```
    The keywords and the default values of PSO can be seen in the following table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_particle"                   | nothing                    |
    | "max_episode"                    | [1000,100]                 |
    | "c0"                             | 1.0                        |
    | "c1"                             | 2.0                        |
    | "c2"                             | 2.0                        |

    Here `p_num` is the number of particles, `ini_particle` represents the initial guesses
    of phase difference. `max_episode` accepts both integer and array with two elements.
    If it is an integer, for example `max_episode=1000`, it means the program will
    continuously run 1000 episodes. However, if it is an array, for example
    `max_episode=[1000,100]`, the program will run 1000 episodes in total but replace the
    data of all the particles with global best every 100 episodes. `c0`, `c1` and `c2` are
    the PSO parameters representing the inertia weight, cognitive learning factor and
    social learning factor, respectively.

If the optimization algorithm is DE, the keywords and the default values are
=== "Python"
    ``` py
    kwargs = {"p_num":10, "deltaphi0":[], "max_episode":1000, "c":1.0,
              "cr":0.5, "seed":1234}
    ```
    The keywords and the default values of DE can be seen in the following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                        | 10                         |
    | "deltaphi0"                      | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |
    | "seed"                           | 1234                       |

    `p_num` and `max_episode` are the number of populations and training episodes.
    `c` and `cr` are DE parameters representing the mutation and crossover constants,
    `seed` is the random seed which can ensure the reproducibility of results.
=== "Julia"
    ```jl
    alg = DE(p_num=10, ini_population=nothing, max_episode=1000,
             c=1.0, cr=0.5)
    ```
    The keywords and the default values of DE can be seen in the following table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_population"                 | nothing                    |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |

    `ini_population` represents the initial guesses of phase difference. `p_num` and
    `max_episode` are the number of populations and training episodes. `c` and `cr` are
    DE parameters representing the mutation and crossover constants.

**Example 9.2**
In this example, the adaptive measurement scheme is designed for the MZI [[3,4]](#Hentschel2010).
The input state is
\begin{align}
\sqrt{\frac{2}{N+2}}\sum^{N/2}_{m=-N/2}\sin\left(\frac{(2m+N+2)\pi}{2(N+2)}\right)|m\rangle,
\end{align}

where $N$ is the number of photon, $|m\rangle$ is the eigenstate of $J_y$ with the eigenvalue
$m$.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # the number of photons
    N = 8
    # probe state
    psi = np.zeros((N+1)**2).reshape(-1, 1)
    for k in range(N+1):
        psi += np.sin((k+1)*np.pi/(N+2))* \
               np.kron(basis(N+1, k), basis(N+1, N-k))
    psi = np.sqrt(2/(2+N))*psi
    rho0 = np.dot(psi, psi.conj().T)
    # prior distribution
    x = np.linspace(-np.pi, np.pi, 100)
    p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
    apt = Adapt_MZI(x, p, rho0)
    apt.general()
    ```
    === "online"
        ``` py
        apt.online(target="sharpness", output="phi")
        ```
    === "offline"
        === "DE"
            ``` py
            DE_para = {"p_num":10, "deltaphi0":[], "max_episode":1000, "c":1.0,
                       "cr":0.5, "seed":1234}
            apt.offline(target="sharpness", method="DE", **DE_para)
            ```
        === "PSO"
            ``` py
            PSO_para = {"p_num":10, "deltaphi0":[], "max_episode":[1000,100],
                        "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
            apt.offline(target="sharpness", method="PSO", **PSO_para)
            ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using SparseArrays

    # the number of photons
    N = 8
    # probe state
    psi = sum([sin(k*pi/(N+2))*kron(QuanEstimation.basis(N+1,k),
          QuanEstimation.basis(N+1, N-k+2)) for k in 1:(N+1)]) |> sparse
    psi = psi*sqrt(2/(2+N))
    rho0 = psi*psi'
    # prior distribution
    x = range(-pi, pi, length=100)
    p = (1.0/(x[end]-x[1]))*ones(length(x))
    apt = QuanEstimation.Adapt_MZI(x, p, rho0)
    ```
    === "online"
        ``` jl
        QuanEstimation.online(apt, target=:sharpness, output="phi")
        ```
    === "offline"
        === "DE"
            ``` jl
            alg = QuanEstimation.DE(p_num=10, ini_population=nothing,
                                    max_episode=1000, c=1.0, cr=0.5)
            QuanEstimation.offline(apt, alg, target=:sharpness, seed=1234)
            ```
        === "PSO"
            ``` jl
            alg = QuanEstimation.PSO(p_num=10, ini_particle=nothing,
                                     max_episode=[1000,100], c0=1.0,
                                     c1=2.0, c2=2.0)
            QuanEstimation.offline(apt, alg, target=:sharpness, seed=1234)
            ```
---

## **Bibliography**
<a id="Berry2000">[1]</a>
D. W. Berry and H. M. Wiseman,
Optimal States and Almost Optimal Adaptive Measurements for Quantum Interferometry,
[Phys. Rev. Lett. **85**, 5098 (2000).](https://doi.org/10.1103/PhysRevLett.85.5098)

<a id="Berry2001">[2]</a>
D. W. Berry, H. M. Wiseman, and J. K. Breslin,
Optimal input states and feedback for interferometric phase estimation,
[Phys. Rev. A **63**, 053804 (2001).](https://doi.org/10.1103/PhysRevA.63.053804)

<a id="Hentschel2010">[3]</a>
A. Hentschel and B. C. Sanders,
Machine Learning for Precise Quantum Measurement,
[Phys. Rev. Lett. **104**, 063603 (2010).](https://doi.org/10.1103/PhysRevLett.104.063603)

<a id="Hentschel2011">[4]</a>
A. Hentschel and B. C. Sanders,
Efficient Algorithm for Optimizing Adaptive Quantum Metrology Processes,
[Phys. Rev. Lett. **104**, 063603 (2011).](https://doi.org/10.1103/PhysRevLett.107.233601)

**See also**: [Quantum metrological tools](guide_bounds.md) — Bayesian estimation provides the prior distribution for adaptive schemes.

