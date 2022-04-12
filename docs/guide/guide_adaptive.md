# **Adaptive measurement schemes**
In QuanEstimation, the Hamiltonian of the adaptive system should be written as
$H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$ 
the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the 
Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. For this scenario,
the adaptive estimation can be excuted through
=== "Python"
    ``` py
    apt = adaptive(x, p, rho0, savefile=False, max_episode=1000, eps=1e-8)
    apt.dynamics(tspan, H, dH, Hc=[], ctrl=[], decay=[])               
    apt.CFIM(M=[], W=[]) 
    ```
=== "Julia"
    ``` jl
    adaptive(x, p, rho0, tspan, H, dH; savefile=false, max_episode=1000, 
             eps=1e-8, Hc=nothing, ctrl=nothing, decay=nothing, M=nothing, 
             W=nothing)
    ```
where `x` is a list of arrays representing the regime of the parameters for the integral, 
`p` is an array representing the prior distribution, it is multidimensional for multiparameter
estimation.`rho` is a density matrix of the probe state.  The number of iterations can be 
set via `max_episode` with the default value 1000. `eps` represents the machine epsilon which 
defaults to $10^{-8}$. Three files "pout.npy", "xout.npy", and "y.npy" will be generated 
including the posterior distributions, the estimated values, and the experimental results. 
If `savefile=True`, these files will be generated during the training and "pout.npy" will save
all the posterior distributions, otherwise, the posterior distribution in the final iteration 
will be saved at the end of the program. 

If the dynamics of the system can be described by the master equation, then the dynamics data 
`tspan`, `H`, and `dH` shoule be input, `tspan` is the time length for the evolution, `H` and 
`dH` are multidimensional lists representing the Hamiltonian and its derivatives on the unknown 
parameters to be estimated, they can be generated via
=== "Python"
    ``` py
    H, dH = AdaptiveInput(x, func, dfunc, channel="dynamics")
    ```
=== "Julia"
    ``` jl
    H, dH = AdaptiveInput(x, func, dfunc; channel="dynamics")
    ```
Here `func` and `dfunc` are the function defined by the users which return `H` and `dH`, 
respectively. Futhermore, for the systems with noise and controls, the variables `decay`, 
`Hc`, and `ctrl` should be input. Here `Hc` and `ctrl` are two lists representing the control 
Hamiltonians and the corresponding control coefficients. `decay` contains decay operators 
$(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$
with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...].  

The objective function for adaptive estimation are CFI and $\mathrm{Tr}(W\mathcal{I}^
{-1})$ with $I$ the CFIM. `W` is the weight matrix which defaults to the identity matrix.

If the parameterization is implemented with the Kraus operators, the codes become
=== "Python"
    ``` py
    apt = adaptive(x, p, rho0, savefile=False, max_episode=1000, eps=1e-8)
    apt.kraus(K, dK)               
    apt.CFIM(M=[], W=[]) 
    ```
=== "Julia"
    ``` jl
    adaptive(x, p, rho0, K, dK; savefile=false, max_episode=1000, eps=1e-8, 
             Hc=nothing, ctrl=nothing, decay=nothing, M=nothing, W=nothing)
    ```
and 
=== "Python"
    ``` py
    K, dK = AdaptiveInput(x, func, dfunc, channel="kraus")
    ```
=== "Julia"
    ``` jl
    K, dK = AdaptiveInput(x, func, dfunc; channel="kraus")
    ```
where `K` and `dK` are the Kraus operators and its derivatives on the unknown parameters.

**Example**  
The Hamiltonian of a qubit system is
\begin{align}
H=\frac{B}{2}(\sigma_1\cos{x}+\sigma_3\sin{x}),
\end{align}

where $B$ is the magnetic field in the XZ plane, $x$ is the unknown parameter and $\sigma_{1}$, $\sigma_{3}$ are the Pauli matrices.

The probe state is taken as $|\pm\rangle$. The measurement is 
$\{|\!+\rangle\langle+\!|,|\!-\rangle\langle-\!|\}$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ $(-1)$. In this example, the prior distribution $p(x)$ is uniform on $[0, \pi/2]$.
=== "Python"
    ``` py
    import numpy as np
    from quanestimation import *
    import random
    from itertools import product

    # initial state
    rho0 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
    # free Hamiltonian
    B = 0.5 * np.pi
    sx = np.array([[0.0j, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = np.array([[1.0, 0.0j], [0.0, -1.0]])
    H0_func = lambda x: 0.5 * x[1] * (sx * np.cos(x[0]) + sz * np.sin(x[0]))
    dH_func = lambda x: [0.5 * x[1] * (-sx * np.sin(x[0]) + sz * np.cos(x[0])), \
                         0.5 * (sx * np.cos(x[0]) + sz * np.sin(x[0]))]
    # measurement
    M1 = 0.5 * np.array([[1.0 + 0.0j, 1.0], [1.0, 1.0]])
    M2 = 0.5 * np.array([[1.0 + 0.0j, -1.0], [-1.0, 1.0]])
    M = [M1, M2]
    # dynamics
    tspan = np.linspace(0.0, 1.0, 1000)

    # Bayesian estimation
    x = [np.linspace(0.0, 0.5 * np.pi, 100),
         np.linspace(0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1, 10)]
    p = ((1.0 / (x[0][-1] - x[0][0]))* (1.0 / (x[1][-1] - x[1][0]))* \
          np.ones((len(x[0]), len(x[1]))))
    dp = np.zeros((len(x[0]), len(x[1])))

    rho = [[[] for j in range(len(x[1]))] for i in range(len(x[0]))]
    for i in range(len(x[0])):
        for j in range(len(x[1])):
            x_tp = [x[0][i], x[1][j]]
            H0_tp = H0_func(x_tp)
            dH_tp = dH_func(x_tp)
            dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
            rho_tp, drho_tp = dynamics.expm()
            rho[i][j] = rho_tp[-1]
    # Generation of the experimental results
    np.random.seed(1234)
    y = [0 for i in range(500)]
    res_rand = random.sample(range(0, len(y)), 125)
    for i in range(len(res_rand)):
        y[res_rand[i]] = 1
    pout, xout = Bayes(x, p, rho, y, M=M, savefile=False)

    # adaptive measurement
    p = pout
    H, dH = AdaptiveInput(x, H0_func, dH_func, channel="dynamics")
    apt = adaptive(x, p, rho0, max_episode=10, eps=1e-8)
    apt.dynamics(tspan, H, dH)
    apt.CFIM(M=M, W=[], savefile=False)
    ```
=== "Julia"
    <span style="color:red">(julia code) </span>

---
Berry et al. [[1,2]](#Berry2000) introduced a famous adaptive scheme in phase estimation. The 
phase for the $(n+1)$th round is updated via $\Phi_{n+1}=\Phi_{n}-(-1)^{y^{(n)}}\Delta
\Phi_{n+1}$ with $y^{(n)}$ the experimental result in the $n$th round and $\Delta\Phi_{n+1}$ 
the phase difference generated by the proper algorithms. This adaptive scheme can be performed
in QuanEstimation by
=== "Python"
    ``` py
    apt = adaptMZI(x, p, rho0)
    apt.general()
    apt.online(output="phi")
    ```
=== "Julia"
    <span style="color:red">(julia code) </span>
Here `x`, `p`, and `rho0` are the same with `adaptive`. The output can be set through 
`output="phi"` (default) and `output="dphi"` representing the phase and phase difference, 
respectively. Online and offline strategies are both available in the package and the code for 
calling offline stratege becomes `apt.offline(method="DE", **kwargs)` or 
`apt.offline(method="PSO", **kwargs)`. If `method="DE"`, `**kwargs` is
=== "Python"
    ``` py
    kwargs = {"popsize":10, "DeltaPhi0":[], "max_episode":1000, "c":1.0, 
              "cr":0.5, "seed":1234}
    ```
=== "Julia"
    <span style="color:red">(julia code) </span>

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "popsize"                        | 10                         |
| "DeltaPhi0"                      | [ ]                        |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |
| "seed"                           | 1234                       |

`DeltaPhi0` represents the initial guesses of phase difference. `popsize` and `max_episode` 
are the number of populations and training episodes. `c` and `cr` are DE parameters 
representing the mutation and crossover constants, `seed` is the random seed which can
ensure the reproducibility of results.

If `method="PSO"`, `**kwargs` becomes
=== "Python"
    ``` py
    kwargs = {"particle_num":10, "DeltaPhi0":[], "max_episode":[1000,100], 
          "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
    ```
=== "Julia"
    <span style="color:red">(julia code) </span>

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "particle_num"                   | 10                         |
| "DeltaPhi0"                          | [ ]                        |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |
| "seed"                           | 1234                       |

Here `particle_num` is the number of particles, `max_episode` accepts both integer and 
array with two elements. If it is an integer, for example `max_episode=1000`, it means the 
program will continuously run 1000 episodes. However, if it is an array, for example 
`max_episode=[1000,100]`, the program will run 1000 episodes in total but replace the data 
of all the particles with global best every 100 episodes. `c0`, `c1`, and `c2` are the PSO 
parameters representing the inertia weight, cognitive learning factor and social 
learning factor, respectively. 

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
