# **Adaptive measurement schemes**
In QuanEstimation, the Hamiltonian of the adaptive system should be written as
$H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$ 
the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the 
Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. In this scenario,
the adaptive estimation can be excuted through
=== "Python"
    ``` py
    apt = Adapt(x, p, rho0, method="FOP", savefile=False, 
                max_episode=1000, eps=1e-8)
    apt.dynamics(tspan, H, dH, Hc=[], ctrl=[], decay=[], 
                 dyn_method="expm")               
    apt.CFIM(M=[], W=[]) 
    ```
    where `x` is a list of arrays representing the regime of the parameters for the integral, 
    `p` is an array representing the prior distribution, it is multidimensional for multiparameter
    estimation.`rho0` is the density matrix of the probe state. The number of iterations can be 
    set via `max_episode` with the default value 1000. `eps` represents the machine epsilon which 
    defaults to $10^{-8}$. At the end of the program, three files "pout.npy", "xout.npy", and "y.npy"  
    including the posterior distributions, the estimated values and the experimental results will be 
    generated. The package contains two mothods for updating the tunable parameters. The first one is 
    updating the tunable parameters with a fix optimal point (`mtheod="FOP"`), which is the default 
    method in QuanEstimation. The other is `method="MI"` which means updating the tunable parameters 
    by maximizing the mutual information which is defined as
    
    \begin{equation}
    I(\textbf{u})=\int\mathrm{p}(\textbf{x}) \sum_{y}\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{log}_2 
    \left[\frac{\mathrm{p}(y|\textbf{x},\textbf{u})}{\int\mathrm{p}(\textbf{x})\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{d}\textbf{x}}\right]\mathrm{d}\textbf{x}.
    \end{equation}

    If `savefile=True`, these files will be generated during the 
    training and "pout.npy" will save all the posterior distributions, otherwise, the posterior 
    distribution in the final iteration will be saved. 
=== "Julia"
    ``` jl
    Adapt(x, p, rho0, tspan, H, dH; dyn_method=:Expm, method="FOP", 
          savefile=false, max_episode=1000, eps=1e-8, Hc=missing, 
          ctrl=missing, decay=missing, M=missing, W=missing)
    ```
    where `x` is a list of arrays representing the regime of the parameters for the integral, 
    `p` is an array representing the prior distribution, it is multidimensional for multiparameter
    estimation.`rho0` is the density matrix of the probe state. The number of iterations can be 
    set via `max_episode` with the default value 1000. `eps` represents the machine epsilon which 
    defaults to $10^{-8}$. At the end of the program, three files "pout.csv", "xout.csv", and "y.csv"  
    including the posterior distributions, the estimated values and the experimental results will be 
    generated. The package contains two mothods for updating the tunable parameters. The first one is 
    updating the tunable parameters with a fix optimal point (`mtheod="FOP"`), which is the default 
    method in QuanEstimation. The other is `method="MI"` which means updating the tunable parameters 
    by maximizing the mutual information which is defined as
    
    \begin{equation}
    I(\textbf{u})=\int\mathrm{p}(\textbf{x}) \sum_{y}\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{log}_2 
    \left[\frac{\mathrm{p}(y|\textbf{x},\textbf{u})}{\int\mathrm{p}(\textbf{x})\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{d}\textbf{x}}\right]\mathrm{d}\textbf{x}.
    \end{equation}
    
    If `savefile=true`, these files will be generated during the training and "pout.csv" 
    will save all the posterior distributions, otherwise, the posterior distribution in the final 
    iteration will be saved. 
If the dynamics of the system can be described by the master equation, then the dynamics data 
`tspan`, `H`, and `dH` shoule be input. `tspan` is the time length for the evolution, `H` and 
`dH` are multidimensional lists representing the Hamiltonian and its derivatives with respect to
the unknown parameters to be estimated, they can be generated via
=== "Python"
    ``` py
    H, dH = BayesInput(x, func, dfunc, channel="dynamics")
    ```
=== "Julia"
    ``` jl
    H, dH = BayesInput(x, func, dfunc; channel="dynamics")
    ```
Here `func` and `dfunc` are the functions defined by the users which return `H` and `dH`, 
respectively. Futhermore, for the systems with noise and controls, the variables `decay`, 
`Hc` and `ctrl` should be input. Here `Hc` and `ctrl` are two lists representing the control 
Hamiltonians and the corresponding control coefficients. `decay` contains decay operators 
$(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$
with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...].  

The objective function for adaptive measurement are CFI and $\mathrm{Tr}(W\mathcal{I}^
{-1})$ with $\mathcal{I}$ the CFIM. `W` is the weight matrix which defaults to the identity matrix.

If the parameterization is implemented with the Kraus operators, the codes become
=== "Python"
    ``` py
    apt = Adapt(x, p, rho0, method="FOP", savefile=False,  
                max_episode=1000, eps=1e-8)
    apt.Kraus(K, dK)               
    apt.CFIM(M=[], W=[]) 
    ```
=== "Julia"
    ``` jl
    Adapt(x, p, rho0, K, dK; method="FOP", savefile=false, 
          max_episode=1000, eps=1e-8, Hc=missing, ctrl=missing, 
          decay=missing, M=missing, W=missing)
    ```
and 
=== "Python"
    ``` py
    K, dK = BayesInput(x, func, dfunc, channel="Kraus")
    ```
=== "Julia"
    ``` jl
    K, dK = BayesInput(x, func, dfunc; channel="Kraus")
    ```
where `K` and `dK` are the Kraus operators and its derivatives with respect to the unknown parameters.

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
    H0_func = lambda x: 0.5*B*omega0*(sx*np.cos(x[0])+sz*np.sin(x[0]))
    # derivative of free Hamiltonian in x
    dH_func = lambda x: [0.5*B*omega0*(-sx*np.sin(x[0])+sz*np.cos(x[0]))]
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
        H_tp = H0_func([x[xi]])
        dH_tp = dH_func([x[xi]])
        dynamics = Lindblad(tspan, rho0, H_tp, dH_tp)
        rho_tp, drho_tp = dynamics.expm()
        rho[xi] = rho_tp[-1]
    # Bayesian estimation
    np.random.seed(1234)
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
    apt.CFIM(M=M, W=[])
    ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using Random
    using StatsBase

    # free Hamiltonian
    function H0_func(x)
        return 0.5*B*omega0*(sx*cos(x[1])+sz*sin(x[1]))
    end
    # derivative of free Hamiltonian in x
    function dH_func(x)
        return [0.5*B*omega0*(-sx*sin(x[1])+sz*cos(x[1]))]
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
    # generation of H and dH
    H, dH = QuanEstimation.BayesInput([x], H0_func, dH_func; 
                                      channel="dynamics")
    # adaptive measurement
    QuanEstimation.Adapt([x], pout, rho0, tspan, H, dH; M=M, dyn_method=:Expm, 
                         method="FOP", max_episode=100)
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
    Here `x`, `p`, and `rho0` are the same with `Adapt`. `target="sharpness"` represents the
    target function for calculating the tunable phase is sharpness, and it can also be set as 
    `target="MI"` which means the target function is mutual information. The output can be set 
    through `output="phi"` (default) and `output="dphi"` representing the phase and phase 
    difference, respectively. Online and offline strategies are both available in the package 
    and the code for calling offline stratege becomes `apt.offline(method="DE", **kwargs)` or 
    `apt.offline(method="PSO", **kwargs)`. 
=== "Julia"
    ``` jl
    apt = Adapt_MZI(x, p, rho0)
    online(apt, target=:sharpness, output="phi")
    ```
    Here `x`, `p`, and `rho0` are the same with `Adapt`. `target=:sharpness` represents the
    target function for calculating the tunable phase is sharpness, and it can also be set as 
    `target=:MI` which means the target function is mutual information. The output can be set 
    through `output="phi"` (default) and `output="dphi"` representing the phase and phase 
    difference, respectively. Online and offline strategies are both available in the package 
    and the code for calling offline stratege becomes `alg = QuanEstimation.DE(kwargs...)` 
    (`alg = QuanEstimation.PSO(kwargs...)`) and `offline(apt, alg, seed=seed)`. 
    `seed` is the random seed which can ensure the reproducibility of results.

If the optimization algorithm is PSO, the keywords and the default values are
=== "Python"
    ``` py
    kwargs = {"p_num":10, "deltaphi0":[], "max_episode":[1000,100], 
              "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
    ```
    The keywords and the default values of PSO can be seen in the following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                   | 10                         |
    | "deltaphi0"                      | [ ]                        |
    | "max_episode"                    | [1000,100]                 |
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
    alg = PSO(p_num=10, ini_particle=missing, max_episode=[1000,100], 
              c0=1.0, c1=2.0, c2=2.0)
    ```
    The keywords and the default values of PSO can be seen in the following table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_particle"                   | missing                    |
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
    alg = DE(p_num=10, ini_population=missing, max_episode=1000, 
             c=1.0, cr=0.5)
    ``` 
    The keywords and the default values of DE can be seen in the following table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_population"                 | missing                    |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |

    `ini_population` represents the initial guesses of phase difference. `p_num` and 
    `max_episode` are the number of populations and training episodes. `c` and `cr` are 
    DE parameters representing the mutation and crossover constants.

**Example 9.2**  
In this example, the adaptive measurement shceme is design for the MZI [[3,4]](#Hentschel2010). 
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
        ``` py
        QuanEstimation.online(apt, target=:sharpness, output="phi")
        ```
    === "offline"
        === "DE"
            ``` jl
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            QuanEstimation.offline(apt, alg, target=:sharpness, seed=1234)
            ```
        === "PSO"
            ``` jl
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing,  
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

