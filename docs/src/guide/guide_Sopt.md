# **State optimization**
For state optimization in QuanEstimation, the probe state is expanded as 
$|\psi\rangle=\sum_i c_i|i\rangle$ in a specific basis $\{|i\rangle\}$. Thus, search of the 
optimal probe states is equal to search of the normalized complex coefficients $\{c_i\}$. In 
QuanEstimation, the state optimization algorithms are the automatic differentiation (AD) 
[[1]](#Baydin2018), reverse iterative (RI)[[2]](#Rafal2011) algorithm, particle swarm optimization (PSO) 
[[3]](#Kennedy1995), differential evolution (DE) [[4]](#Storn1997), and Nelder-Mead (NM) 
[[5]](#Nelder1965). 
Call the following codes to perform state optimizaiton
=== "Python"
    ``` py
    state = StateOpt(savefile=False, method="AD", **kwargs)
    state.dynamics(tspan, H0, dH, Hc=[], ctrl=[], decay=[], 
                   dyn_method="expm")
    ```
    === "QFIM"
        ``` py
        state.QFIM(W=[], LDtype="SLD")
        ```
    === "CFIM"
        ``` py
        state.CFIM(M=[], W=[])
        ```
    === "HCRB"
        ``` py
        state.HCRB(W=[])
        ```
    The variable `savefile` means whether to save all the states. If set `False` (default) the 
    states in the final episode and the values of the objective function in all episodes will 
    be saved. If set `True` then the states and the values of the objective function obtained 
    in all episodes will be saved during the training. `method` represents the algorithm used 
    to optimize the states, options are: "AD", "PSO", "DE", and "NM". `**kwargs` contains 
    the keywords and default values corresponding to the optimization algorithm which will be 
    introduced in detail below.

    `tspan` is the time length for the evolution, `H0` and `dH` are the free Hamiltonian and its
    derivatives with respect to the unknown parameters to be estimated. `H0` accepts both matrix 
    (time-independent evolution) and list of matrices (time-dependent evolution) with the length equal to 
    `tspan`. `dH` should be input as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` 
    and `ctrl` are two lists represent the control Hamiltonians and the corresponding control 
    coefficients. `decay` contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the 
    corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule 
    decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. The default values for 
    `decay`, `Hc` and `ctrl` are empty which means the dynamics is unitary and only governed by 
    the free Hamiltonian. `dyn_method="expm"` represents the method for solving the dynamics is 
    matrix exponential, it can also be set as `dyn_method="ode"` which means the dynamics 
    (differential equation) is directly solved with the ODE solvers.

    The objective functions for state optimization can be chosen as QFI $\left[\mathrm{Tr}
    (W\mathcal{F}^{-1})\right]$, CFI $\left[\mathrm{Tr}(W\mathcal{I}^{-1})\right]$, and HCRB, 
    the corresponding codes for them are `state.QFIM()` (default), `state.CFIM()`, and 
    `state.HCRB()`. Here $\mathcal{F}$ and $\mathcal{I}$ are the QFIM and CFIM, $W$ corresponds 
    to `W` represents the weight matrix, the default value for `W` is the identity matrix. If 
    the users call `state.HCRB()` for single parameter scenario, the program will exit and print 
    `"Program terminated. In the single-parameter scenario, the HCRB is equivalent to the QFI. Please 
    choose 'QFIM' as the objective function"`. `LDtype` in `state.QFIM()` represents the types 
    of the QFIM, it can be set as `LDtype="SLD"` (default), `LDtype="RLD"`, and `LDtype="LLD"`. 
    `M` represents a set of positive operator-valued measure (POVM) with default value `[]`. 
    In the package, a set of rank-one symmetric informationally complete POVM (SIC-POVM) is 
    used when `M=[]`.
=== "Julia"
    ``` jl
    opt = StateOpt(psi=psi, seed=1234)
    alg = AD(kwargs...)
    dynamics = Lindblad(opt, tspan, H0, dH; Hc=missing, 
                        ctrl=missing, decay=missing, dyn_method=:Expm)
    ```
    === "QFIM"
        ``` jl
        obj = QFIM_obj(W=missing, LDtype=:SLD)
        ```
    === "CFIM"
        ``` jl
        obj = CFIM_obj(M=missing, W=missing)
        ```
    === "HCRB"
        ``` jl
        obj = HCRB_obj(W=missing)
        ```
    ``` jl
    run(opt, alg, obj, dynamics; savefile=false)
    ```
    The initial state (optimization variable) can be input via `psi=psi` in `StateOpt()` for 
    constructing a state optimization problem. `psi` is an array representing the state. 
    `Lindblad` accepts the dynamics parameters. `tspan` is the time length for the evolution, 
    `H0` and `dH` are the free Hamiltonian and its derivatives with respect to the unknown 
    parameters to be estimated. `H0` accepts both matrix (time-independent evolution) and list of matrices
    (time-dependent evolution) with the length equal to `tspan`. `dH` should be input as 
    $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` and `ctrl` are two lists represent the 
    control Hamiltonians and the corresponding control coefficients. `decay` contains decay 
    operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates 
    $(\gamma_1, \gamma_2, \cdots)$ with the input rule 
    decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. The default values for 
    `decay`, `Hc`, and `ctrl` are `missing` which means the dynamics is unitary and only governed 
    by the free Hamiltonian. `seed` is the random seed which can ensure the reproducibility of 
    results. `dyn_method=:Expm` represents the method for solving the dynamics is 
    matrix exponential, it can also be set as `dyn_method=:Ode` which means the dynamics 
    (differential equation) is directly solved with the ODE solvers.

    The objective functions for state optimization can be set as QFI ($\mathrm{Tr}(W\mathcal{F}^
    {-1})$), CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$), and HCRB, the corresponding codes for them are
    `QFIM_obj()` (default), `CFIM_obj()`, and `HCRB_obj()`. Here $\mathcal{F}$ and 
    $\mathcal{I}$ are the QFIM and CFIM, $W$ corresponds to `W` is the weight matrix which 
    defaults to the identity matrix. If the users call `HCRB_obj()` for single parameter 
    scenario, the program will exit and print `"Program terminated. In the single-parameter scenario, 
    the HCRB is equivalent to the QFI. Please choose 'QFIM_obj()' as the objective function"`.
    `LDtype` in `QFIM_obj()` represents the types of the QFIM, it can be set as `LDtype=:SLD` 
    (default), `LDtype=:RLD` and `LDtype=:LLD`. `M` represents a set of positive operator-valued 
    measure (POVM) with default value `missing`. In the package, a set of rank-one symmetric 
    informationally complete POVM (SIC-POVM) is used when `M=missing`.

    `savefile` means whether to save all the states. If set `false` (default) the states in the 
    final episode and the values of the objective function in all episodes will be saved. If set 
    `true` then the states and the values of the objective function obtained in all episodes will 
    be saved during the training. The algorithm used to optimize the states in QuanEstimation 
    are AD, PSO, DE, DDPG, and NM. `kwargs...` contains the keywords and default values corresponding 
    to the optimization algorithm which will be introduced in detail below.

---
## **AD**
The code for state optimization with AD is as follows
=== "Python"
    ``` py
    state = StateOpt(method="GRAPE", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"Adam":False, "psi0":[], "max_episode":300, "epsilon":0.01, 
              "beta1":0.90, "beta2":0.99}
    ```
    The keywords and the default values of AD can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "Adam"                           | False                      |
    | "psi0"                           | [ ]                        |
    | "max_episode"                    | 300                        |
    | "epsilon"                        | 0.01                       |
    | "beta1"                          | 0.90                       |
    | "beta2"                          | 0.99                       |

    In state optimization, the state will update according to the learning rate `"epsilon"`.
    However, Adam algorithm can be introduced to update the states which can be realized by setting 
    `Adam=True`. In this case, the Adam parameters include learning rate, the exponential decay 
    rate for the first moment estimates and the second moment estimates can be set by the user via 
    `epsilon`, `beta1` and `beta2`. `psi0` is a list representing the initial guesses of states and 
    `max_episode` is the number of episodes.
=== "Julia"
    ``` jl
    alg = AD(Adam=false, max_episode=300, epsilon=0.01, beta1=0.90, 
             beta2=0.99)
    ```
    The keywords and the default values of AD can be seen in the following 
    table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "Adam"                           | false                      |
    | "max_episode"                    | 300                        |
    | "epsilon"                        | 0.01                       |
    | "beta1"                          | 0.90                       |
    | "beta2"                          | 0.99                       |

    In state optimization, the state will update according to the learning rate `"epsilon"`.
    However, Adam algorithm can be introduced to update the states which can be realized by setting 
    `Adam=true`. In this case, the Adam parameters include learning rate, the exponential decay 
    rate for the first moment estimates and the second moment estimates can be set by the user via 
    `epsilon`, `beta1`, and `beta2`.

## **RI**
The code for state optimization with RI is as follows
=== "Python"
    ``` py
    state = StateOpt(method="RI", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"psi0":[], "max_episode":300, "seed":1234}
    ```
    The keywords and the default values of RI can be seen in the following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "psi0"                           | [ ]                        |
    | "max_episode"                    | 300                        |
    | "seed"                           | 1234                       |

    `psi0` is a list representing the initial guesses of states and `max_episode` is the number of episodes. `seed` is the random seed. 
=== "Julia"
    ``` jl
    alg = RI(max_episode=300)
    ```
    The keywords and the default values of RI can be seen in the following 
    table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "max_episode"                    | 300                        |

    `max_episode` is the number of episodes.

## **PSO**
The code for state optimization with PSO is as follows
=== "Python"
    ``` py
    state = StateOpt(method="PSO", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "psi0":[], "max_episode":[1000,100], 
              "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
    ```
    The keywords and the default values of PSO can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                   | 10                         |
    | "psi0"                           | [ ]                        |
    | "max_episode"                    | [1000,100]                 |
    | "c0"                             | 1.0                        |
    | "c1"                             | 2.0                        |
    | "c2"                             | 2.0                        |
    | "seed"                           | 1234                       |

    `p_num` is the number of particles. Here `max_episode` accepts both integers and 
    arrays with two elements. If it is an integer, for example `max_episode=1000`, it means the 
    program will continuously run 1000 episodes. However, if it is an array, for example 
    `max_episode=[1000,100]`, the program will run 1000 episodes in total but replace states of 
    all the particles with global best every 100 episodes. `c0`, `c1`, and `c2` are the PSO 
    parameters representing the inertia weight, cognitive learning factor, and social learning 
    factor, respectively.  `seed` is the random seed.
=== "Julia"
    ``` jl
    alg = PSO(p_num=10, ini_particle=missing,  max_episode=[1000,100], 
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

    `p_num` is the number of particles. `ini_particle` is a tuple representing the initial 
    guesses of states and `max_episode` is the number of episodes. Here `max_episode` accepts 
    both integers and arrays with two elements. If it is an integer, for example `max_episode=1000`, 
    it means the program will continuously run 1000 episodes. However, if it is an array, for example 
    `max_episode=[1000,100]`, the program will run 1000 episodes in total but replace states of 
    all the particles with global best every 100 episodes. `c0`, `c1`, and `c2` are the PSO 
    parameters representing the inertia weight, cognitive learning factor, and social learning 
    factor, respectively. 

## **DE**
The code for state optimization with DE is as follows
=== "Python"
    ``` py
    state = StateOpt(method="DE", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "psi0":[], "max_episode":1000, "c":1.0, 
              "cr":0.5, "seed":1234}
    ```
    The keywords and the default values of DE can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                        | 10                         |
    | "psi0"                           | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |
    | "seed"                           | 1234                       |

    `p_num` represents the number of populations. `c` and `cr` are the mutation constant and 
    crossover constant.
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

    `ini_population` is a tuple representing the initial guesses of states , `c` and `cr` 
    are the mutation constant and crossover constant.

## **NM**
The code for state optimization with NM is as follows
=== "Python"
    ``` py
    state = StateOpt(method="NM", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "psi0":psi0, "max_episode":1000, "ar":1.0, 
              "ae":2.0, "ac":0.5, "as0":0.5, "seed":1234}
    ```
    The keywords and the default values of NM can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                      | 10                         |
    | "psi0"                           | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "ar"                             | 1.0                        |
    | "ae"                             | 2.0                        |
    | "ac"                             | 0.5                        |
    | "as0"                            | 0.5                        |
    | "seed"                           | 1234                       |

    `p_num` represents the number of initial states. `ar`, `ae`, `ac`, and `as0` are 
    constants for reflection, expansion, constraction, and shrink, respectively.
=== "Julia"
    ``` jl
    alg = NM(p_num=10, ini_state=missing, max_episode=1000, ar=1.0, 
             ae=2.0, ac=0.5, as0=0.5)
    ```
    The keywords and the default values of NM can be seen in the following 
    table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                      | 10                         |
    | "ini_state"                      | missing                    |
    | "max_episode"                    | 1000                       |
    | "ar"                             | 1.0                        |
    | "ae"                             | 2.0                        |
    | "ac"                             | 0.5                        |
    | "as0"                            | 0.5                        |

    `ini_state` represents the number of initial states. `ar`, `ae`, `ac`, and `as0` are 
    constants for reflection, expansion, constraction, and shrink, respectively.



**Example 6.1**  
<a id="example6_1"></a>
The Hamiltonian of the Lipkin–Meshkov–Glick (LMG) model is
\begin{align}
H_{\mathrm{LMG}}=-\frac{\lambda}{N}(J_1^2+gJ_2^2)-hJ_3,
\end{align}

where $N$ is the number of spins of the system, $\lambda$ is the spin–spin interaction strength, 
$h$ is the strength of the external field and $g$ is the anisotropic parameter. 
$J_i=\frac{1}{2}\sum_{j=1}^N \sigma_i^{(j)}$ ($i=1,2,3$) is the collective spin operator with 
$\sigma_i^{(j)}$ the $i$th Pauli matrix for the $j$th spin. In single-parameter scenario, we take 
$g$ as the unknown parameter to be estimated. The states are expanded as 
$|\psi\rangle=\sum^J_{m=-J}c_m|J,m\rangle$ with $|J,m\rangle$ the Dicke state and $c_m$ a complex 
coefficient. Here we fixed $J=N/2$. In this example, the probe state is optimized for both noiseless
scenario and collective dephasing noise. The dynamics under collective dephasing can be expressed as
<center> $\partial_t\rho = -i[H_{\mathrm{LMG}},\rho]+\gamma \left(J_3\rho J_3-\frac{1}{2}\left\{\rho, J^2_3\right\}\right)$ </center>
with $\gamma$ the decay rate.

In this case, all searches with different algorithms start from the coherent spin state defined by
$|\theta=\frac{\pi}{2},\phi=\frac{\pi}{2}\rangle=\exp(-\frac{\theta}{2}e^{-i\phi}J_{+}+\frac{\theta}{2}e^{i\phi}J_{-})|J,J\rangle$ with $J_{\pm}=J_1{\pm}iJ_2$. Here, QuTip [[6,7]](#Johansson2012) is applied
for generation of the spin coherent state.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np
    from qutip import *

    # the dimension of the system
    N = 8
    # generation of the coherent spin state
    psi_css = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type="ket").full()
    psi_css = psi_css.reshape(1, -1)[0]
    # guessed state
    psi0 = [psi_css]
    # free Hamiltonian
    Lambda, g, h = 1.0, 0.5, 0.1
    Jx, Jy, Jz = jmat(0.5*N)
    Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
    H0 = -Lambda*(np.dot(Jx, Jx) + g*np.dot(Jy, Jy))/N - h*Jz
    # derivative of the free Hamiltonian on g
    dH = [-Lambda*np.dot(Jy, Jy)/N]
    # dissipation
    decay = [[Jz, 0.1]]
    # time length for the evolution
    tspan = np.linspace(0., 10., 2500)
    ```
    === "AD"
		``` py
        # state optimization algorithm: AD
		AD_paras = {"Adam":False, "psi0":psi0, "max_episode":300, \
                    "epsilon":0.01, "beta1":0.90, "beta2":0.99}
        state = StateOpt(savefile=False, method="AD", **AD_paras)
		```
	=== "PSO"
		``` py
        # state optimization algorithm: PSO
		PSO_paras = {"p_num":10, "psi0":psi0, "max_episode":[1000,100], \
					 "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
		state = StateOpt(savefile=False, method="PSO", **PSO_paras)
		```
	=== "DE"
		``` py
        # state optimization algorithm: DE
		DE_paras = {"p_num":10, "psi0":psi0, "max_episode":1000, "c":1.0, \
				    "cr":0.5, "seed":1234}
		state = StateOpt(savefile=False, method="DE", **DE_paras)
		```
    === "NM"
		``` py
        # state optimization algorithm: NM
		NM_paras = {"p_num":20, "psi0":psi0, "max_episode":1000, \
                    "ar":1.0, "ae":2.0, "ac":0.5, "as0":0.5, "seed":1234}
        state = StateOpt(savefile=False, method="NM", **NM_paras)
		```
        
    ``` py
    # input the dynamics data
    state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
    ```
    === "QFIM"
        ``` py
        # objective function: QFI
        state.QFIM()
        ```
    === "CFIM"
        ``` py
        # objective function: CFI
        state.CFIM()
        ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using Random
    using StableRNGs
    using LinearAlgebra
    using SparseArrays

    # the dimension of the system
    N = 8
    # generation of the coherent spin state
    j, theta, phi = N÷2, 0.5pi, 0.5pi
    Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
    Jm = Jp'
    psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
           QuanEstimation.basis(Int(2*j+1), 1)
    dim = length(psi0)
    # free Hamiltonian
    lambda, g, h = 1.0, 0.5, 0.1
    Jx = 0.5*(Jp + Jm)
    Jy = -0.5im*(Jp - Jm)
    Jz = spdiagm(j:-1:-j)
    H0 = -lambda*(Jx*Jx + g*Jy*Jy) / N - h*Jz
    # derivative of the free Hamiltonian on g
    dH = [-lambda*Jy*Jy/N]
    # dissipation
    decay = [[Jz, 0.1]]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # set the optimization type
    opt = QuanEstimation.StateOpt(psi=psi0, seed=1234) 
    ```
    === "AD"
        ``` jl
        # state optimization algorithm: AD
        alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01, 
                                beta1=0.90, beta2=0.99)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    === "PSO"
        ``` jl
        # state optimization algorithm: PSO
        alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                 c1=2.0, c2=2.0)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    === "DE"
        ``` jl
        # state optimization algorithm: DE
        alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    === "NM"
        ``` jl
        # state optimization algorithm: NM
        alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0, 
                                ae=2.0, ac=0.5, as0=0.5)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    ``` jl
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, 
                                       dyn_method=:Expm) 
    # run the state optimization problem
    QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
    ```
**Example 6.2**  
<a id="example6_2"></a>
In the multiparameter scenario, $g$ and $h$ are chooen to be the unknown parameters to be estimated.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np
    from qutip import *

    # the dimension of the system
    N = 8
    # generation of coherent spin state
    psi_css = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type="ket").full()
    psi_css = psi_css.reshape(1, -1)[0]
    # guessed state
    psi0 = [psi_css]
    # free Hamiltonian
    Lambda, g, h = 1.0, 0.5, 0.1
    Jx, Jy, Jz = jmat(0.5*N)
    Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
    H0 = -Lambda*(np.dot(Jx, Jx) + g*np.dot(Jy, Jy))/N - h*Jz
    # derivatives of the free Hamiltonian on the g and h
    dH = [-Lambda*np.dot(Jy, Jy)/N, -Jz]
    # dissipation
    decay = [[Jz, 0.1]]
    # time length for the evolution
    tspan = np.linspace(0., 10., 2500)
    # weight matrix
    W = np.array([[1/3, 0.], [0., 2/3]])
    ```
    === "AD"
		``` py
        # state optimization algorithm: AD
		AD_paras = {"Adam":False, "psi0":psi0, "max_episode":300, \
                    "epsilon":0.01, "beta1":0.90, "beta2":0.99}
        state = StateOpt(savefile=False, method="AD", **AD_paras)
		```
        ``` py
        # input the dynamics data
        state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
        ```
        === "QFIM"
            ``` py
            # objective function: tr(WF^{-1})
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: tr(WI^{-1})
            state.CFIM()
            ```
        === "HCRB"
            ``` py
            # objective function: HCRB
            state.HCRB()
            ```
	=== "PSO"
		``` py
        # state optimization algorithm: PSO
		PSO_paras = {"p_num":10, "psi0":psi0, "max_episode":[1000,100], \
					 "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
		state = StateOpt(savefile=False, method="PSO", **PSO_paras)
		```
        ``` py
        # input the dynamics data
        state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
        ```
        === "QFIM"
            ``` py
            # objective function: tr(WF^{-1})
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: tr(WI^{-1})
            state.CFIM()
            ```
        === "HCRB"
            ``` py
            # objective function: HCRB
            state.HCRB()
            ```
	=== "DE"
		``` py
        # state optimization algorithm: DE
		DE_paras = {"p_num":10, "psi0":psi0, "max_episode":1000, "c":1.0, \
				    "cr":0.5, "seed":1234}
		state = StateOpt(savefile=False, method="DE", **DE_paras)
		```
        ``` py
        # input the dynamics data
        state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
        ```
        === "QFIM"
            ``` py
            # objective function: tr(WF^{-1})
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: tr(WI^{-1})
            state.CFIM()
            ```
        === "HCRB"
            ``` py
            # objective function: HCRB
            state.HCRB()
            ```
    === "NM"
		``` py
        # state optimization algorithm: NM
		NM_paras = {"p_num":20, "psi0":psi0, "max_episode":1000, \
                    "ar":1.0, "ae":2.0, "ac":0.5, "as0":0.5, "seed":1234}
        state = StateOpt(savefile=False, method="NM", **NM_paras)
		```
        ``` py
        # input the dynamics data
        state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
        ```
        === "QFIM"
            ``` py
            # objective function: tr(WF^{-1})
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: tr(WI^{-1})
            state.CFIM()
            ```
        === "HCRB"
            ``` py
            # objective function: HCRB
            state.HCRB()
            ```
	<!-- === "DDPG"
		``` py
        # state optimization algorithm: DDPG
		DDPG_paras = {"layer_num":4, "layer_dim":250, "max_episode":500, \
		              "seed":1234}
		state = StateOpt(savefile=False, method="DDPG", **DDPG_paras)
		```
        ``` py
        # input the dynamics data
        state.dynamics(tspan, H0, dH, decay=decay, dyn_method="expm")
        ```
        === "QFIM"
            ``` py
            # objective function: tr(WF^{-1})
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: tr(WI^{-1})
            state.CFIM()
            ``` -->
=== "Julia"
    ``` jl
    using QuanEstimation
    using Random
    using StableRNGs
    using LinearAlgebra
    using SparseArrays

    # the dimension of the system
    N = 8
    # generation of the coherent spin state
    j, theta, phi = N÷2, 0.5pi, 0.5pi
    Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
    Jm = Jp'
    psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
               QuanEstimation.basis(Int(2*j+1), 1)
    dim = length(psi0)
    # free Hamiltonian
    lambda, g, h = 1.0, 0.5, 0.1
    Jx = 0.5*(Jp + Jm)
    Jy = -0.5im*(Jp - Jm)
    Jz = spdiagm(j:-1:-j)
    H0 = -lambda*(Jx*Jx + g*Jy*Jy) / N + g * Jy^2 / N - h*Jz
    # derivative of the free Hamiltonian on g
    dH = [-lambda*Jy*Jy/N, -Jz]
    # dissipation
    decay = [[Jz, 0.1]]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # weight matrix
    W = [1/3 0.; 0. 2/3]
    # set the optimization type
    opt = QuanEstimation.StateOpt(psi=psi0, seed=1234)
    ```
    === "AD"
        ``` jl
        # state optimization algorithm: AD
        alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01, 
                                beta1=0.90, beta2=0.99)
        ```
        === "QFIM"
            ``` jl
            # objective function: tr(WF^{-1})
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: tr(WI^{-1})
            obj = QuanEstimation.CFIM_obj()
            ```
    === "PSO"
        ``` jl
        # state optimization algorithm: PSO
        alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                 c1=2.0, c2=2.0)
        ```
        === "QFIM"
            ``` jl
            # objective function: tr(WF^{-1})
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: tr(WI^{-1})
            obj = QuanEstimation.CFIM_obj()
            ```
        === "HCRB"
            ``` jl
            # objective function: HCRB
            obj = QuanEstimation.HCRB_obj()
            ```
    === "DE"
        ``` jl
        # state optimization algorithm: DE
        alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
        ```
        === "QFIM"
            ``` jl
            # objective function: tr(WF^{-1})
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: tr(WI^{-1})
            obj = QuanEstimation.CFIM_obj()
            ```
        === "HCRB"
            ``` jl
            # objective function: HCRB
            obj = QuanEstimation.HCRB_obj()
            ```
    === "NM"
        ``` jl
        # state optimization algorithm: NM
        alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0, 
                                ae=2.0, ac=0.5, as0=0.5)
        ```
        === "QFIM"
            ``` jl
            # objective function: tr(WF^{-1})
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: tr(WI^{-1})
            obj = QuanEstimation.CFIM_obj()
            ```
        === "HCRB"
            ``` jl
            # objective function: HCRB
            obj = QuanEstimation.HCRB_obj()
            ```
    <!-- === "DDPG"
        ``` jl
        # state optimization algorithm: DDPG
        alg = QuanEstimation.DDPG(max_episode=500, layer_num=3, layer_dim=200)
        ```
        === "QFIM"
            ``` jl
            # objective function: tr(WF^{-1})
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: tr(WI^{-1})
            obj = QuanEstimation.CFIM_obj()
            ``` -->
    ``` jl
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, 
                                       dyn_method=:Expm) 
    # run the state optimization problem
    QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
    ```

If the parameterization process is implemented with the Kraus operators, then the corresponding 
codes are
=== "Python"
    ``` py
    state = StateOpt(savefile=False, method="AD", **kwargs)
    state.state.Kraus(K, dK)
    ```
    === "QFIM"
        ``` py
        state.QFIM(W=[], LDtype="SLD")
        ```
    === "CFIM"
        ``` py
        state.CFIM(M=[], W=[])
        ```
    === "HCRB"
        ``` py
        state.HCRB(W=[])
        ```
=== "Julia"
    ``` jl
    opt = StateOpt(psi=psi, seed=1234)
    alg = AD(kwargs...)
    dynamics = Kraus(opt, K, dK)
    ```
    === "QFIM"
        ``` jl
        obj = QFIM_obj(W=missing, LDtype=:SLD)
        ```
    === "CFIM"
        ``` jl
        obj = CFIM_obj(M=missing, W=missing)
        ```
    === "HCRB"
        ``` jl
        obj = HCRB_obj(W=missing)
        ```
    ``` jl
    run(opt, alg, obj, dynamics; savefile=false)
    ```

where `K` and `dK` are the Kraus operators and its derivatives with respect to the 
unknown parameters.

**Example 6.3**  
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

where $\gamma$ is the unknown parameter to be estimated which represents the decay probability.

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
    === "AD"
		``` py
        # state optimization algorithm: AD
		AD_paras = {"Adam":False, "psi0":[], "max_episode":300, \
                    "epsilon":0.01, "beta1":0.90, "beta2":0.99}
        state = StateOpt(savefile=False, method="AD", **AD_paras)
		```
        ``` py
        # input the dynamics data
        state.Kraus(K, dK)
        ```
        === "QFIM"
            ``` py
            # objective function: QFI
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: CFI
            state.CFIM()
            ```
    === "RI"
		``` py
        # state optimization algorithm: RI
		RI_paras = {"psi0":[], "max_episode":300, "seed":1234}
        state = StateOpt(savefile=False, method="RI", **RI_paras)
		```
        ``` py
        # input the dynamics data
        state.Kraus(K, dK)
        ```
        === "QFIM"
            ``` py
            # objective function: QFI
            state.QFIM()
            ```
	=== "PSO"
		``` py
        # state optimization algorithm: PSO
		PSO_paras = {"p_num":10, "psi0":[], "max_episode":[1000,100], \
					 "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
		state = StateOpt(savefile=False, method="PSO", **PSO_paras)
		```
        ``` py
        # input the dynamics data
        state.Kraus(K, dK)
        ```
        === "QFIM"
            ``` py
            # objective function: QFI
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: CFI
            state.CFIM()
            ```
	=== "DE"
		``` py
        # state optimization algorithm: DE
		DE_paras = {"p_num":10, "psi0":[], "max_episode":1000, "c":1.0, \
				    "cr":0.5, "seed":1234}
		state = StateOpt(savefile=False, method="DE", **DE_paras)
		```
        ``` py
        # input the dynamics data
        state.Kraus(K, dK)
        ```
        === "QFIM"
            ``` py
            # objective function: QFI
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: CFI
            state.CFIM()
            ```
    === "NM"
		``` py
        # state optimization algorithm: NM
		NM_paras = {"p_num":20, "psi0":[], "max_episode":1000, \
                    "ar":1.0, "ae":2.0, "ac":0.5, "as0":0.5, "seed":1234}
        state = StateOpt(savefile=False, method="NM", **NM_paras)
		```
        ``` py
        # input the dynamics data
        state.Kraus(K, dK)
        ```
        === "QFIM"
            ``` py
            # objective function: QFI
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: CFI
            state.CFIM()
            ```
	<!-- === "DDPG"
		``` py
        # state optimization algorithm: DDPG
		DDPG_paras = {"layer_num":4, "layer_dim":250, "max_episode":500, \
		              "seed":1234}
		state = StateOpt(savefile=False, method="DDPG", **DDPG_paras)
		```
        ``` py
        # input the dynamics data
        state.Kraus(K, dK)
        ```
        === "QFIM"
            ``` py
            # objective function: QFI
            state.QFIM()
            ```
        === "CFIM"
            ``` py
            # objective function: CFI
            state.CFIM()
            ``` -->
=== "Julia"
    ``` jl
    using QuanEstimation

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
    opt = QuanEstimation.Sopt(seed=1234)
    ```
    === "AD"
        ``` jl
        # state optimization algorithm: AD
        alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01, 
                                beta1=0.90, beta2=0.99)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    === "RI"
        ``` jl
        # state optimization algorithm: RI
        alg = QuanEstimation.RI(max_episode=300)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
    === "PSO"
        ``` jl
        # state optimization algorithm: PSO
        alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
                                 c1=2.0, c2=2.0)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    === "DE"
        ``` jl
        # state optimization algorithm: DE
        alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
        ```
                === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    === "NM"
        ``` jl
        # state optimization algorithm: NM
        alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0, 
                                ae=2.0, ac=0.5, as0=0.5)
        ```        
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ```
    <!-- === "DDPG"
        ``` jl
        # state optimization algorithm: DDPG
        alg = QuanEstimation.DDPG(max_episode=500, layer_num=3, layer_dim=200)
        ```
        === "QFIM"
            ``` jl
            # objective function: QFI
            obj = QuanEstimation.QFIM_obj()
            ```
        === "CFIM"
            ``` jl
            # objective function: CFI
            obj = QuanEstimation.CFIM_obj()
            ``` -->
    ``` jl
    # input the dynamics data
    dynamics = QuanEstimation.Kraus(opt, K, dK)
    # run the state optimization problem
    QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
    ```
---
## **Bibliography**
<a id="Baydin2018">[1]</a>
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)

<a id="Rafal2011">[2]</a>
R. Demkowicz-Dobrzański,
Optimal phase estimation with arbitrary a priori knowledge,
[Phys. Rev. A **83**, 061802(R) (2011).
](https://doi.org/10.1103/PhysRevA.83.061802)

<a id="Kennedy1995">[3]</a>
J. Kennedy and R. Eberhar,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

<a id="Storn1997">[4]</a>
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

<a id="Nelder1965">[5]</a>
J. A. Nelder and R. Mead,
A Simplex Method for Function Minimization,
[Comput. J. **7**, 308–313 (1965).](https://doi.org/10.1093/comjnl/7.4.308)

<a id="Johansson2012">[6]</a>
J. R. Johansson, P. D. Nation, and F. Nori,
QuTiP: An open-source Python framework for the dynamics of open quantum systems,
[Comp. Phys. Comm. **183**, 1760 (2012).](https://doi.org/10.1016/j.cpc.2012.02.021)

<a id="Johansson2013">[7]</a>
J. R. Johansson, P. D. Nation, and F. Nori,
QuTiP 2: A Python framework for the dynamics of open quantum systems,
[Comp. Phys. Comm. **184**, 1234 (2013).](https://doi.org/10.1016/j.cpc.2012.11.019)
