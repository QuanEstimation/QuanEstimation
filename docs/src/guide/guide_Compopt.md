# **Comprehensive optimization**
In order to obtain the optimal parameter estimation schemes, it is necessary to
simultaneously optimize the probe state, control, and measurement. The comprehensive 
optimization for the probe state and measurement (SM), the probe state and control (SC), the 
control and measurement (CM) and the probe state, control and measurement (SCM) are proposed
in QuanEstiamtion. In the package, the comprehensive optimization algorithms are particle 
swarm optimization (PSO) [[1]](#Kennedy1995), differential evolution (DE) [[2]](#Storn1997), 
and automatic differentiation (AD) [[3]](#Baydin2018).
=== "Python"
    ``` py
    com = ComprehensiveOpt(savefile=False, method="DE", **kwargs)
    com.dynamics(tspan, H0, dH, Hc=[], ctrl=[], decay=[], ctrl_bound=[], 
                 dyn_method="expm")
    ```
    === "SM"
        ``` py
        com.SM(W=[])
        ```
    === "SC"
        ``` py
        com.SC(W=[], M=[], target="QFIM", LDtype="SLD")
        ```
    === "CM"
        ``` py
        com.CM(rho0, W=[])
        ```
    === "SCM"
        ``` py
        com.SCM(W=[])  
        ```
    Here `savefile` means whether to save all the optimized variables (probe states, control 
    coefficients and measurements). If set `True` then the optimized variables and the values 
    of the objective function obtained in all episodes will be saved during the training, 
    otherwise, the optimized variables in the final episode and the values of the objective 
    function in all episodes will be saved. `method` represents the optimization algorithm used, 
    options are: "PSO", "DE", and "AD". `**kwargs` is the keyword and the default value
    corresponding to the optimization algorithm which will be introduced in detail below.

    If the dynamics of the system can be described by the master equation, then the dynamics data 
    `tspan`, `H0`, and `dH` shoule be input. `tspan` is the time length for the evolution, `H0`
    and `dH` are the free Hamiltonian and its derivatives on the unknown parameters to be 
    estimated. `H0` is a matrix when the free Hamiltonian is time-independent and a list of matrices with the 
    length equal to `tspan` when it is time-dependent. `dH` should be input as
    $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` and `ctrl` are two lists represent the
    control Hamiltonians and the corresponding control coefficients.`decay` contains decay 
    operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, 
    \gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, 
    $\gamma_2$],...]. The default values for `decay`, `Hc` and `ctrl` are empty which means the 
    dynamics is unitary and only governed by the free Hamiltonian. `ctrl_bound`is an array with two 
    elements representing the lower and upper bound of the control coefficients, respectively. The 
    default value of `ctrl_bound=[]` which means the control coefficients are in the regime 
    $[-\infty,\infty]$. `dyn_method="expm"` represents the method for solving the dynamics is 
    matrix exponential, it can also be set as `dyn_method="ode"` which means the dynamics 
    (differential equation) is directly solved with the ODE solvers.

    QuanEstimation contains four comprehensive optimizations which are `com.SM()`, `com.SC()`,
    `com.CM()` and `com.SCM()`. The `target` in `com.SC()` can be set as `target="QFIM"` (default), 
    `target="CFIM` and `target="HCRB` for the corresponding objective functions are QFI 
    ($\mathrm{Tr}(W\mathcal{F}^{-1})$), CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$), and HCRB, 
    respectively. Here $\mathcal{F}$ and $\mathcal{I}$ are the QFIM and CFIM, $W$ corresponds to 
    `W` is the weight matrix which defaults to the identity matrix. If the users set `target="HCRB` 
    for single parameter scenario, the program will exit and print `"Program terminated. In the 
    single-parameter scenario, the HCRB is equivalent to the QFI. Please choose 'QFIM' as the 
    objective function"`. `LDtype` represents the types of the QFIM, it can be set as 
    `LDtype="SLD"` (default), `LDtype="RLD"`, and `LDtype="LLD"`. For the other three scenarios, 
    the objective function is CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$).
=== "Julia"
    === "SM"
        ``` jl
        opt = SMopt(psi=psi, M=M, seed=1234)
        alg = DE(kwargs...)
        dynamics = Lindblad(opt, tspan, H0, dH; Hc=missing, ctrl=missing, 
                            decay=missing, dyn_method=:Expm)  
        obj = CFIM_obj(W=missing)
        run(opt, alg, obj, dynamics; savefile=false)
        ```
    === "SC"
        ``` jl
        opt = SCopt(psi=psi, ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
        alg = DE(kwargs...)
        dynamics = Lindblad(opt, tspan, H0, dH, Hc; decay=missing, 
                            dyn_method=:Expm)
        ```
        === "QFIM"
            ``` jl
            obj = QFIM_obj(W=missing, LDtype=:SLD)
            ```
        === "CFIM"
            ``` jl
            obj = CFIM_obj(W=missing)
            ```
        === "HCRB"
            ``` jl
            obj = HCRB_obj(W=missing)
            ```
        ``` jl
        run(opt, alg, obj, dynamics; savefile=false)
        ```
    === "CM"
        ``` jl
        opt = CMopt(ctrl=ctrl, M=M, ctrl_bound=ctrl_bound, seed=1234)
        alg = DE(kwargs...)
        dynamics = Lindblad(opt, tspan, H0, dH, Hc; decay=missing,
                            dyn_method=:Expm)
        obj = CFIM_obj(W=missing)
        run(opt, alg, obj, dynamics; savefile=false)
        ```
    === "SCM"
        ``` jl
        opt = SCMopt(psi=psi, ctrl=ctrl, M=M, ctrl_bound=ctrl_bound, seed=1234)
        alg = DE(kwargs...)
        dynamics = Lindblad(opt, tspan, H0, dH, Hc; decay=missing, 
                            dyn_method=:Expm)
        obj = CFIM_obj(W=missing)
        run(opt, alg, obj, dynamics; savefile=false)
        ```
    QuanEstimation contains four comprehensive optimizations which are `SMopt()`, `SCopt()`,
    `CMopt()`, and `SCMopt()`. The optimization variables including initial state, control,
    and measurement can be input via `ctrl=ctrl`, `psi=psi`, and `M=M`  for constructing a 
    comprehensive optimization problem. Here, `ctrl` is a list of arrays with the length 
    equal to control Hamiltonians, `psi` is an array representing the state and `M` is
    a list of arrays with the length equal to the dimension of the system which representing 
    the projective measurement basis. Besides, the boundary value of each control 
    coefficients can be input via `ctrl_bound=ctrl_bound` when the optimized variable 
    contains control. `ctrl_bound` is an array with two elements representing the lower and 
    upper bound of the control coefficients, respectively. The default value of `ctrl_bound=
    missing` which means the control coefficients are in the regime $[-\infty,\infty]$.
    `seed` is the random seed which can ensure the reproducibility of results.
    
    The objective function of `SCopt()` can be chosen as `QFIM_obj()` (default), `CFIM_obj()`, 
    and `HCRB_obj()` for the corresponding objective functions are QFI ($\mathrm{Tr}(W\mathcal{F}^
    {-1})$), CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$), and HCRB, respectively. Here $\mathcal{F}$ 
    and $\mathcal{I}$ are the QFIM and CFIM, $W$ corresponds to `W` is the weight matrix which 
    defaults to the identity matrix. If the users set `HCRB_obj()` for single parameter scenario, 
    the program will exit and print `"Program terminated. In the single-parameter scenario, the 
    HCRB is equivalent to the QFI. Please choose 'QFIM_obj()' as the objective function"`. `LDtype` 
    represents the types of the QFIM, it can be set as `LDtype=:SLD` (default), `LDtype=:RLD`, 
    and `LDtype=:LLD`. For the other three scenarios, the objective function is `CFIM_obj()`.
    
    If the dynamics of the system can be described by the master equation, then the dynamics data 
    `tspan`, `H0`, and `dH` shoule be input. `tspan` is the time length for the evolution, `H0`
    and `dH` are the free Hamiltonian and its derivatives on the unknown parameters to be 
    estimated. `H0` is a matrix when the free Hamiltonian is time-independent and a list of matrices with the 
    length equal to `tspan` when it is time-dependent. `dH` should be input as
    $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` and `ctrl` are two lists represent the
    control Hamiltonians and the corresponding control coefficients.`decay` contains decay 
    operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, 
    \gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, 
    $\gamma_2$],...]. `dyn_method=:Expm` represents the method for solving the dynamics is 
    matrix exponential, it can also be set as `dyn_method=:Ode` which means the dynamics 
    (differential equation) is directly solved with the ODE solvers.

    The variable `savefile` means whether to save all the optimized variables (probe states, 
    control coefficients, and measurements). If set `true` then the optimized variables and the 
    values of the objective function obtained in all episodes will be saved during the training, 
    otherwise, the optimized variables in the final episode and the values of the objective 
    function in all episodes will be saved. The algorithm used in QuanEstimation for 
    comprehensive optimizaiton are PSO, DE, and AD. `kwargs...` is the keywords and the default 
    values corresponding to the optimization algorithm which will be introduced in detail below.

---
## **PSO**
The code for comprehensive optimization with PSO is as follows
=== "Python"
    ``` py
    com = ComprehensiveOpt(method="PSO", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], 
              "max_episode":[1000,100], "c0":1.0, "c1":2.0, "c2":2.0, 
              "seed":1234}
    ```
    The keywords and the default values of PSO can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                   | 10                         |
    | "psi0"                           | [ ]                        |
    | "ctrl0"                          | [ ]                        |
    | "measurement0"                   | [ ]                        |
    | "max_episode"                    | [1000,100]                 |
    | "c0"                             | 1.0                        |
    | "c1"                             | 2.0                        |
    | "c2"                             | 2.0                        |
    | "seed"                           | 1234                       |

    `psi0`, `ctrl0`, and `measurement0` in the algorithms represent the initial 
    guesses of states, control coefficients and measurements, respectively, `seed` 
    is the random seed. Here `p_num` is the number of particles, `c0`, 
    `c1`, and `c2` are the PSO parameters representing the inertia weight, cognitive 
    learning factor, and social learning factor, respectively. `max_episode` accepts 
    both integers and arrays with two elements. If it is an integer, for example 
    max_episode=1000, it means the program will continuously run 1000 episodes. 
    However, if it is an array, for example max_episode=[1000,100], the program will 
    run 1000 episodes in total but replace control coefficients of all the particles 
    with global best every 100 episodes. 
=== "Julia"
    ``` jl
    alg = PSO(p_num=10, ini_particle=missing, max_episode=[1000,100], 
              c0=1.0, c1=2.0, c2=2.0)
    ```
    The keywords and the default values of PSO can be seen in the following 
    table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_particle"                   | missing                    |
    | "max_episode"                    | [1000,100]                 |
    | "c0"                             | 1.0                        |
    | "c1"                             | 2.0                        |
    | "c2"                             | 2.0                        |

    `ini_particle`is a tuple contains `psi0`, `ctrl0`, and `measurement0`, which 
    representing the initial guesses of states, control coefficients, and measurements,
    respectively. The input rule of `ini_particle` shoule be `ini_particle=(psi0, 
    measurement0)`(SM), `ini_particle=(psi0, ctrl0)`(SC), `ini_particle=(ctrl0, 
    measurement0)`(CM) and  `ini_particle=(psi0, ctrl0, measurement0)`(SCM).
    Here `p_num` is the number of particles, `c0`, `c1`, and `c2` are the PSO parameters 
    representing the inertia weight, cognitive learning factor, and social learning 
    factor, respectively. `max_episode` accepts both integers and arrays with two 
    elements. If it is an integer, for example max_episode=1000, it means the program 
    will continuously run 1000 episodes. However, if it is an array, for example 
    max_episode=[1000,100], the program will run 1000 episodes in total but replace 
    control coefficients of all the particles with global best every 100 episodes. 

## **DE**
The code for comprehensive optimization with DE is as follows
=== "Python"
    ``` py
    com = ComprehensiveOpt(method="DE", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], 
              "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
    ```
    The keywords and the default values of DE can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                        | 10                         |
    | "psi0"                           | [ ]                        |
    | "ctrl0"                          | [ ]                        |
    | "measurement0"                   | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |
    | "seed"                           | 1234                       |

    `p_num` is the number of populations. Here `max_episode` is 
    an integer representing the number of episodes. `c` and `cr` are constants for 
    mutation and crossover.  
=== "Julia"
    ``` jl
    alg = DE(p_num=10, ini_population=missing, max_episode=1000, 
             c=1.0, cr=0.5)
    ```
    The keywords and the default values of DE can be seen in the following 
    table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_population"                 | missing                    |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |

    Here `max_episode` is an integer representing the number of episodes. `p_num` 
    represents the number of populations. `c` and `cr` are constants for mutation 
    and crossover. `ini_particle`is a tuple contains `psi0`, `ctrl0`, and `measurement0`,
    which representing the initial guesses of states, control coefficients, and 
    measurements, respectively. The input rule of `ini_particle` shoule be 
    `ini_particle=(psi0, measurement0)`(SM), `ini_particle=(psi0, ctrl0)`(SC), 
    `ini_particle=(ctrl0, measurement0)`(CM), and  
    `ini_particle=(psi0, ctrl0, measurement0)`(SCM).
## **AD**
The code for comprehensive optimization with AD is as follows
=== "Python"
    ``` py
    com = ComprehensiveOpt(method="AD", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"Adam":True, "psi0":[], "ctrl0":[], "measurement0":[],
              "max_episode":300, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
    ```
    The keywords and the default values of AD can be seen in the 
    following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "Adam"                           | False                      |
    | "psi0"                           | [ ]                        |
    | "ctrl0"                          | [ ]                        |
    | "measurement0"                   | [ ]                        |
    | "max_episode"                    | 300                        |
    | "epsilon"                        | 0.01                       |
    | "beta1"                          | 0.90                       |
    | "beta2"                          | 0.99                       |

    The optimized variables will update according to the learning rate `"epsilon"` 
    when `Adam=False`. However, If `Adam=True`, Adam algorithm will be used and the 
    Adam parameters include learning rate, the exponential decay rate for the first 
    moment estimates and the second moment estimates can be set by the user via 
    `epsilon`, `beta1` and `beta2`.
=== "Julia"
    ``` jl
    alg = AD(Adam=false, max_episode=300, epsilon=0.01, beta1=0.90, 
            beta2=0.99)
    ```
    The keywords and the default values of AD can be seen in the 
    following table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "Adam"                           | false                      |
    | "max_episode"                    | 300                        |
    | "epsilon"                        | 0.01                       |
    | "beta1"                          | 0.90                       |
    | "beta2"                          | 0.99                       |

    The optimized variables will update according to the learning rate `"epsilon"` 
    when `Adam=false`. However, If `Adam=true`, Adam algorithm will be used and the 
    Adam parameters include learning rate, the exponential decay rate for the first 
    moment estimates, and the second moment estimates can be set by the user via 
    `epsilon`, `beta1`, and `beta2`.

**Example 8.1**  
<a id="example8_1"></a>
A single qubit system whose free evolution Hamiltonian is $H_0 = \frac{1}{2}\omega \sigma_3$ with 
$\omega$ the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by

\begin{align}
\partial_t\rho=-i[H_0, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}\sigma_{-},\rho\}\right),
\end{align}

where $\gamma_{+}$, $\gamma_{-}$ are decay rates and $\sigma_{\pm}=(\sigma_1 \pm \sigma_2)/2$. The control Hamiltonian
\begin{align}
H_\mathrm{c}=u_1(t)\sigma_1+u_2(t)\sigma_2+u_3(t)\sigma_3
\end{align}

with $u_i(t)$ $(i=1,2,3)$ the control field. Here $\sigma_{1}$, $\sigma_{2}$ are also Pauli matrices.

In this case, we consider two types of comprehensive optimization, the first one is optimization of probe state and control (SC), and the other is optimization of probe state, control and measurement (SCM). QFI is taken as the target function for SC and CFI for SCM.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # initial state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    omega = 1.0
    sx = np.array([[0., 1.], [1., 0.]])
	sy = np.array([[0., -1.j], [1.j, 0.]]) 
	sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # control Hamiltonians 
    Hc = [sx,sy,sz]
    # dissipation
    sp = np.array([[0., 1.], [0., 0.]])  
	sm = np.array([[0., 0.], [1., 0.]]) 
	decay = [[sp, 0.], [sm, 0.1]]
    # measurement
    M1 = 0.5*np.array([[1., 1.], [1., 1.]])
	M2 = 0.5*np.array([[1., -1.], [-1., 1.]])
    M = [M1, M2]
    M_num = 2
    # time length for the evolution
    tspan = np.linspace(0., 10., 2500)
    ```
    === "SM"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
            com.SM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
            com.SM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
    === "SC"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            ```
            === "QFIM"
                ``` py
                com.SC(W=[], target="QFIM", LDtype="SLD")
                ```
            === "CFIM"
                ``` py
                com.SC(W=[], M=M, target="CFIM")
                ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            ```
            === "QFIM"
                ``` py
                com.SC(W=[], target="QFIM", LDtype="SLD")
                ```
            === "CFIM"
                ``` py
                com.SC(W=[], M=M, target="CFIM")
                ```
        === "AD"
            ``` py
            # comprehensive optimization algorithm: AD
            AD_paras = {"Adam":False, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":300, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            com = ComprehensiveOpt(savefile=False, method="AD", **AD_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            ```
            === "QFIM"
                ``` py
                com.SC(W=[], target="QFIM", LDtype="SLD")
                ```
            === "CFIM"
                ``` py
                com.SC(W=[], M=M, target="CFIM")
                ```
    === "CM"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            com.CM(rho0)
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            com.CM(rho0)
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
    === "SCM"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            com.SCM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-2.0,2.0], \
                         dyn_method="expm")
            com.SCM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using DelimitedFiles

    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
	sy = [0. -im; im 0.]
	sz = [1. 0.0im; 0. -1.]
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0. 1.; 0. 0.0im]
	sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
	M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    M_num = 2
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    ```
    === "SM"
        ``` jl
        opt = QuanEstimation.SMopt(seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5) 
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M)
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, 
                                               dyn_method=:Expm)
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M) 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, 
                                               dyn_method=:Expm)   
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
    === "SC"
        ``` jl
        opt = QuanEstimation.SCopt(ctrl_bound=[-2.0,2.0], seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
            ```
            === "QFIM"
                ``` jl
                # objective function: QFI
                obj = QuanEstimation.QFIM_obj()
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "CFIM"
                ``` jl
                # objective function: CFI
                obj = QuanEstimation.CFIM_obj(M=M) 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0)
            ```
            === "QFIM"
                ``` jl
                # objective function: QFI
                obj = QuanEstimation.QFIM_obj() 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "CFIM"
                ``` jl
                # objective function: CFI
                obj = QuanEstimation.CFIM_obj(M=M) 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
        === "AD"
            ``` jl
            # comprehensive optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ``` 
            === "QFIM"
                ``` jl
                # objective function: QFI
                obj = QuanEstimation.QFIM_obj() 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "CFIM"
                ``` jl
                # objective function: CFI
                obj = QuanEstimation.CFIM_obj(M=M) 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
    === "CM"
        ``` jl
        opt = QuanEstimation.CMopt(ctrl_bound=[-2.0,2.0], seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)  
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M) 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc,   
                                               decay=decay, dyn_method=:Expm) 
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M) 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc,  
                                               decay=decay, dyn_method=:Expm)  
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
    === "SCM"
        ``` jl
        opt = QuanEstimation.SCMopt(ctrl_bound=[-2.0,2.0], seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5) 
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M) 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                               dyn_method=:Expm)  
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0) 
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M)
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                               dyn_method=:Expm)  
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
**Example 8.2**  
<a id="example8_2"></a>
The Hamiltonian of a controlled system can be written as
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters $\textbf{x}$ and $H_k$ 
represents the $k$th control Hamiltonian with $u_k$ the correspong control coefficient.

In the multiparameter scenario, the dynamics of electron and nuclear coupling in NV$^{-}$ can be expressed as
\begin{align}
\partial_t\rho=-i[H_0+H_{\mathrm{c}},\rho]+\frac{\gamma}{2}(S_3\rho S_3-S^2_3\rho-\rho S^2_3)
\end{align}

with $\gamma$ the dephasing rate. And
\begin{align}
H_0/\hbar=DS^2_3+g_{\mathrm{S}}\vec{B}\cdot\vec{S}+g_{\mathrm{I}}\vec{B}\cdot\vec{I}+\vec{S}^{\,\mathrm{T}}\mathcal{A}\vec{I}
\end{align}

is the free evolution Hamiltonian, where $\vec{S}=(S_1,S_2,S_3)^{\mathrm{T}}$ and $\vec{I}=(I_1,I_2,I_3)^{\mathrm{T}}$ with 
$S_i=s_i\otimes I$ and $I_i=I\otimes \sigma_i$ ($i=1,2,3$) the electron and nuclear operators. $\mathcal{A}=\mathrm{diag}
(A_1,A_1,A_2)$ is the hyperfine tensor with $A_1$ and $A_2$ the axial and transverse magnetic hyperfine coupling coefficients.
The coefficients $g_{\mathrm{S}}=g_\mathrm{e}\mu_\mathrm{B}/\hbar$ and $g_{\mathrm{I}}=g_\mathrm{n}\mu_\mathrm{n}/\hbar$, 
where $g_\mathrm{e}$ ($g_\mathrm{n}$) is the $g$ factor of the electron (nuclear), $\mu_\mathrm{B}$ ($\mu_\mathrm{n}$) is the Bohr (nuclear) magneton, and $\hbar$ is the Plank's constant. $\vec{B}$ is the magnetic field which be estimated. The control Hamiltonian is
\begin{align}
H_{\mathrm{c}}/\hbar=\sum^3_{i=1}\Omega_i(t)S_i
\end{align}

with $\Omega_i(t)$ the time-dependent Rabi frequency.

In this case, the initial state is taken as $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)\otimes|\!\!\uparrow\rangle$, 
where $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)$ is an electron state with $|1\rangle$ $(|\!-\!1\rangle)$ the 
eigenstate of $s_3$ with respect to the eigenvalue $1$ ($-1$). $|\!\!\uparrow\rangle$ is a nuclear state and 
the eigenstate of $\sigma_3$ with respect to the eigenvalue 1. $W$ is set to be identity.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # initial state
    rho0 = np.zeros((6, 6), dtype=np.complex128)
    rho0[0][0], rho0[0][4], rho0[4][0], rho0[4][4] = 0.5, 0.5, 0.5, 0.5
    # free Hamiltonian
    sx = np.array([[0., 1.],[1., 0.]])
	sy = np.array([[0., -1.j],[1.j, 0.]]) 
	sz = np.array([[1., 0.],[0., -1.]])
	ide2 = np.array([[1., 0.],[0., 1.]])
	s1 = np.array([[0., 1., 0.],[1., 0., 1.],[0., 1., 0.]])/np.sqrt(2)
	s2 = np.array([[0., -1.j, 0.],[1.j, 0., -1.j],[0., 1.j, 0.]])/np.sqrt(2)
	s3 = np.array([[1., 0., 0.],[0., 0., 0.],[0., 0., -1.]])
	ide3 = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])
	I1, I2, I3 = np.kron(ide3, sx), np.kron(ide3, sy), np.kron(ide3, sz)
	S1, S2, S3 = np.kron(s1, ide2), np.kron(s2, ide2), np.kron(s3, ide2)
	B1, B2, B3 = 5.0e-4, 5.0e-4, 5.0e-4
    # All numbers are divided by 100 in this example 
    # for better calculation accurancy
    cons = 100
	D = (2*np.pi*2.87*1000)/cons
	gS = (2*np.pi*28.03*1000)/cons
	gI = (2*np.pi*4.32)/cons
	A1 = (2*np.pi*3.65)/cons
	A2 = (2*np.pi*3.03)/cons
	H0 = D*np.kron(np.dot(s3, s3), ide2) + gS*(B1*S1+B2*S2+B3*S3) \
		 + gI*(B1*I1+B2*I2+B3*I3) + A1*(np.kron(s1, sx) + np.kron(s2, sy)) \
		 + A2*np.kron(s3, sz)
    # derivatives of the free Hamiltonian on B1, B2, and B3
    dH = [gS*S1+gI*I1, gS*S2+gI*I2, gS*S3+gI*I3]
    # control Hamiltonians 
    Hc = [S1, S2, S3]
    # dissipation
    decay = [[S3,2*np.pi/cons]]  
    # measurement
    dim = len(rho0)
    M_num = dim
    M = [np.dot(basis(dim, i), basis(dim, i).conj().T) for i in range(dim)]
    # time length for the evolution
    tspan = np.linspace(0., 2., 4000)
    # guessed control coefficients
    cnum = 10
    ctrl0 = -0.2*np.ones((len(Hc), cnum))
    ```
    === "SM"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
            com.SM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
            com.SM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
    === "SC"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl=[ctrl0], \
                         ctrl_bound=[-0.2,0.2], dyn_method="expm")
            ```
            === "QFIM"
                ``` py
                # objective function: tr(WF^{-1})
                com.SC(W=[], target="QFIM", LDtype="SLD")
                ```
            === "CFIM"
                ``` py
                # objective function: tr(WI^{-1})
                com.SC(W=[], M=M, target="CFIM")
                ```
            === "HCRB"
                ``` py
                # objective function: HCRB
                com.SC(W=[], target="HCRB")
                ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl=[ctrl0], \
                         ctrl_bound=[-0.2,0.2], dyn_method="expm")
            ```
            === "QFIM"
                ``` py
                # objective function: tr(WF^{-1})
                com.SC(W=[], target="QFIM", LDtype="SLD")
                ```
            === "CFIM"
                ``` py
                # objective function: tr(WI^{-1})
                com.SC(W=[], M=M, target="CFIM")
                ```
            === "HCRB"
                ``` py
                # objective function: HCRB
                com.SC(W=[], target="HCRB")
                ```
        === "AD"
            ``` py
            # comprehensive optimization algorithm: AD
            AD_paras = {"Adam":False, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":300, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            com = ComprehensiveOpt(savefile=False, method="AD", **AD_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl=[ctrl0], \
                         ctrl_bound=[-0.2,0.2], dyn_method="expm")
            ```
            === "QFIM"
                ``` py
                # objective function: tr(WF^{-1})
                com.SC(W=[], target="QFIM", LDtype="SLD")
                ```
            === "CFIM"
                ``` py
                # objective function: tr(WI^{-1})
                com.SC(W=[], M=M, target="CFIM")
                ```
    === "CM"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-0.2,0.2], \
                         dyn_method="expm")
            com.CM(rho0)
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-0.2,0.2], \  
                         dyn_method="expm")
            com.CM(rho0)
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
    === "SCM"
        === "DE"
            ``` py
            # comprehensive optimization algorithm: DE
            DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                        "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-0.2,0.2], \
                         dyn_method="expm")
            com.SCM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
        === "PSO"
            ``` py
            # comprehensive optimization algorithm: PSO
            PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                         "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                         "c1":2.0, "c2":2.0, "seed":1234}
            com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
            com.dynamics(tspan, H0, dH, Hc=Hc, decay=decay, ctrl_bound=[-0.2,0.2], \
                         dyn_method="expm")
            com.SCM()
            # convert the ".csv" file to the ".npy" file
            M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
            csv2npy_measurements(M_, M_num)
            # load the measurements
            M = np.load("measurements.npy")[-1]
            ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using LinearAlgebra
    using DelimitedFiles

    # initial state
    rho0 = zeros(ComplexF64, 6, 6)
    rho0[1:4:5, 1:4:5] .= 0.5
    # free Hamiltonian
    sx = [0. 1.; 1. 0.]
	sy = [0. -im; im 0.]
	sz = [1. 0.; 0. -1.]
	s1 = [0. 1. 0.; 1. 0. 1.; 0. 1. 0.]/sqrt(2)
	s2 = [0. -im 0.; im 0. -im; 0. im 0.]/sqrt(2)
	s3 = [1. 0. 0.; 0. 0. 0.; 0. 0. -1.]
	Is = I1, I2, I3 = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
	S = S1, S2, S3 = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
	B = B1, B2, B3 = [5.0e-4, 5.0e-4, 5.0e-4]
    # All numbers are divided by 100 in this example 
    # for better calculation accurancy
    cons = 100
	D = (2pi*2.87*1000)/cons
	gS = (2pi*28.03*1000)/cons
	gI = (2pi*4.32)/cons
	A1 = (2pi*3.65)/cons
	A2 = (2pi*3.03)/cons
	H0 = sum([D*kron(s3^2, I(2)), sum(gS*B.*S), sum(gI*B.*Is),
          	  A1*(kron(s1, sx) + kron(s2, sy)), A2*kron(s3, sz)])
    # derivatives of the free Hamiltonian on B1, B2 and B3
    dH = gS*S+gI*Is
    # control Hamiltonians 
    Hc = [S1, S2, S3]
    # dissipation
    decay = [[S3, 2pi/cons]]
    # measurement
    dim = size(rho0, 1)
    M_num = dim
    M = [QuanEstimation.basis(dim, i)*QuanEstimation.basis(dim, i)' 
         for i in 1:dim]
    # time length for the evolution
    tspan = range(0., 2., length=4000)
    # guessed control coefficients
    cnum = 10
    ctrl = -0.2*ones((length(Hc), cnum))
    # guessed measurements
    C = [QuanEstimation.basis(dim, i) for i in 1:dim]
    ```
    === "SM"
        ``` jl
        opt = QuanEstimation.SMopt(seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M) 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, 
                                               dyn_method=:Expm)   
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0) 
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj(M=M) 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, 
                                               dyn_method=:Expm) 
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
    === "SC"
        ``` jl
        opt = QuanEstimation.SCopt(ctrl_bound=[-0.2,0.2], seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
            ```
            === "QFIM"
                ``` jl
                # objective function: tr(WF^{-1})
                obj = QuanEstimation.QFIM_obj() 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "CFIM"
                ``` jl
                # objective function: tr(WI^{-1})
                obj = QuanEstimation.CFIM_obj(M=M) 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "HCRB"
                ``` jl
                # objective function: HCRB
                obj = QuanEstimation.HCRB_obj() 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0) 
            ```
            === "QFIM"
                ``` jl
                # objective function: tr(WF^{-1})
                obj = QuanEstimation.QFIM_obj() 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm)
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "CFIM"
                ``` jl
                # objective function: tr(WI^{-1})
                obj = QuanEstimation.CFIM_obj(M=M) 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm)
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
            === "HCRB"
                ``` jl
                # objective function: HCRB
                obj = QuanEstimation.HCRB_obj() 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm)
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
        === "AD"
            ``` jl
            # comprehensive optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ``` 
            === "QFIM"
                ``` jl
                # objective function: tr(WF^{-1})
                obj = QuanEstimation.QFIM_obj()
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false) 
                ```
            === "CFIM"
                ``` jl
                # objective function: tr(WI^{-1})
                obj = QuanEstimation.CFIM_obj(M=M) 
                # input the dynamics data
                dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                                   dyn_method=:Expm) 
                # run the comprehensive optimization problem
                QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
                ```
    === "CM"
        ``` jl
        opt = QuanEstimation.CMopt(seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc,  
                                               decay=decay, dyn_method=:Expm)    
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj() 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, 
                                               decay=decay, dyn_method=:Expm)  
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
    === "SCM"
        ``` jl
        opt = QuanEstimation.SCMopt(seed=1234)
        ```
        === "DE"
            ``` jl
            # comprehensive optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj() 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                               dyn_method=:Expm)   
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
        === "PSO"
            ``` jl
            # comprehensive optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0)
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj() 
            # input the dynamics data
            dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, 
                                               dyn_method=:Expm)   
            # run the comprehensive optimization problem
            QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
            # convert the flattened data into a list of matrix
            M_ = readdlm("measurements.csv",'\t', Complex{Float64})
            M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
                for j in 1:Int(length(M_[:,1])/M_num)][end]
            ```
For optimization of probe state and measurement, the parameterization can also be implemented 
with the Kraus operators which can be realized by
=== "Python"
    ``` py
    com = ComprehensiveOpt(savefile=False, method="DE", **kwargs)
    com.Kraus(K, dK)
    com.SM(W=[])
    ```
=== "Julia"
    ``` jl
    opt = SMopt(psi=psi, M=M, seed=1234)
    alg = DE(kwargs...)
    dynamics = Kraus(opt, K, dK) 
    obj = CFIM_obj(W=missing)
    run(opt, alg, obj, dynamics; savefile=false)
    ```
where `K` and `dK` are the Kraus operators and its derivatives with respect to the 
unknown parameters.

**Example 8.3**  
The Kraus operators for the amplitude damping channel are

\begin{eqnarray}
K_1 = \left(\begin{array}{cc}
1 & 0  \\
0 & \sqrt{1-\gamma}
\end{array}\right),
K_2 = \left(\begin{array}{cc}
0 & \sqrt{\gamma} \\
0 & 0
\end{array}\right), \nonumber
\end{eqnarray}

where $\gamma$ is the unknown parameter to be estimated which represents the decay 
probability. In this example, the probe state is taken as $|+\rangle\langle+|$ with 
$|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is 
the eigenstate of $\sigma_3$ (Pauli matrix) with respect to the eigenvalue $1$ $(-1)$.

=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # initial state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    # Kraus operators for the amplitude damping channel
    gamma = 0.1
    K1 = np.array([[1., 0.], [0., np.sqrt(1-gamma)]])
    K2 = np.array([[0., np.sqrt(gamma)], [0., 0.]])
    K = [K1, K2]
    # derivatives of Kraus operators on gamma
    dK1 = np.array([[1., 0.], [0., -0.5/np.sqrt(1-gamma)]])
    dK2 = np.array([[0., 0.5/np.sqrt(gamma)], [0., 0.]])
    dK = [[dK1], [dK2]]
    ```
    === "DE"
        ``` py
        # comprehensive optimization algorithm: DE
        DE_paras = {"p_num":10, "psi0":[], "ctrl0":[], "measurement0":[], \
                    "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
        com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
        com.Kraus(K, dK)
        com.SM()
        # convert the ".csv" file to the ".npy" file
        M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
        csv2npy_measurements(M_, 2)
        # load the measurements
        M = np.load("measurements.npy")[-1]
        ```
    === "PSO"
        ``` py
        # comprehensive optimization algorithm: PSO
        PSO_paras = {"p_num":10, "psi0":[], "ctrl0":[], \
                     "measurement0":[], "max_episode":[1000,100], "c0":1.0, \
                     "c1":2.0, "c2":2.0, "seed":1234}
        com = ComprehensiveOpt(savefile=False, method="PSO", **PSO_paras)
        com.Kraus(K, dK)
        com.SM()
        # convert the ".csv" file to the ".npy" file
        M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
        csv2npy_measurements(M_, 2)
        # load the measurements
        M = np.load("measurements.npy")[-1]
        ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using DelimitedFiles

    # initial state
    rho0 = 0.5*ones(2, 2)
    # Kraus operators for the amplitude damping channel
    gamma = 0.1
    K1 = [1. 0.; 0. sqrt(1-gamma)]
    K2 = [0. sqrt(gamma); 0. 0.]
    K = [K1, K2]
    # derivatives of Kraus operators on gamma
    dK1 = [1. 0.; 0. -0.5/sqrt(1-gamma)]
    dK2 = [0. 0.5/sqrt(gamma); 0. 0.]
    dK = [[dK1], [dK2]]
    # comprehensive optimization 
    M_num = 2
    opt = QuanEstimation.SMopt(seed=1234)
    ```
    === "DE"
        ``` jl
        # comprehensive optimization algorithm: DE
        alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
        # objective function: CFI
        obj = QuanEstimation.CFIM_obj() 
        # input the dynamics data
        dynamics = QuanEstimation.Kraus(opt, K, dK)  
        # run the comprehensive optimization problem
        QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
        # convert the flattened data into a list of matrix
        M_ = readdlm("measurements.csv",'\t', Complex{Float64})
        M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
            for j in 1:Int(length(M_[:,1])/M_num)][end]
        ```
    === "PSO"
        ``` jl
        # comprehensive optimization algorithm: PSO
        alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                     c1=2.0, c2=2.0)
        # objective function: CFI
        obj = QuanEstimation.CFIM_obj() 
        # input the dynamics data
        dynamics = QuanEstimation.Kraus(opt, K, dK)   
        # run the comprehensive optimization problem
        QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
        # convert the flattened data into a list of matrix
        M_ = readdlm("measurements.csv",'\t', Complex{Float64})
        M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
            for j in 1:Int(length(M_[:,1])/M_num)][end]
        ```
---
## **Bibliography**
<a id="Kennedy1995">[1]</a>
J. Kennedy and R. Eberhar,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

<a id="Storn1997">[2]</a>
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

<a id="Baydin2018">[3]</a>
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)