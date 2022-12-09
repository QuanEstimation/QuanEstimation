# **Measurement optimization**
In QuanEstimation, three measurement optimization scenarios are considered. The first one
is to optimize a set of rank-one projective measurement, it can be written in a specific
basis $\{|\phi_i\rangle\}$ with $|\phi_i\rangle=\sum_j C_{ij}|j\rangle$ in the Hilbert space 
as $\{|\phi_i\rangle\langle\phi_i|\}$. In this case, the goal is to search a set of optimal 
coefficients $C_{ij}$. The second scenario is to find the optimal linear combination of 
an input measurement $\{\Pi_j\}$. The third scenario is to find the optimal rotated 
measurement of an input measurement. After rotation, the new measurement is
$\{U\Pi_i U^{\dagger}\}$, where $U=\prod_k \exp(i s_k\lambda_k)$ with $\lambda_k$ a SU($N$) 
generator and $s_k$ a real number in the regime $[0,2\pi]$. In this scenario, the goal is 
to search a set of optimal coefficients $s_k$. Here different algorithms are invoked to 
search the optimal measurement include particle swarm optimization (PSO) [[1]](#Kennedy1995), 
differential evolution (DE) [[2]](#Storn1997), and automatic differentiation (AD) 
[[3]](#Baydin2018). The codes for execute measurement optimization are
=== "Python"
    ``` py
    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, 
                       method="DE", **kwargs)
    m.dynamics(tspan, rho0, H0, dH, Hc=[], ctrl=[], decay=[], \
               dyn_method="expm")
    m.CFIM(W=[])
    ```
    `mtype` represents the type of measurement optimization which defaults to `"projection"`. 
    In this setting, rank-one projective measurement optimization will be performed `minput` 
    will keep empty in this scenario. For the other two measurement optimization scenarios 
    `mtype="input"`. If the users want to find the optimal linear combination of an input 
    measurement, the variable `minput` should be input as `minput=["LC", [Pi1,Pi2,...], m]` 
    with `[Pi1,Pi2,...]` a set of POVM and `m` the number of operators of the output 
    measurement. For finding the optimal linear combination of an input measurement, the 
    variable `minput` becomes `minput=["rotation", [Pi1,Pi2,...]]`. `savefile` means whether 
    to save all the measurements. If set `False` the measurements in the final episode and 
    the values of the objective function in all episodes will be saved, if `savefile=True` 
    the measurements and the values of the objective function obtained in all episodes will 
    be saved during the training. `method` represents the algorithm used to optimize the 
    measurements, options are: "PSO", "DE" and "AD". `**kwargs` is the keyword and default 
    value corresponding to the optimization algorithm which will be introduced in detail 
    below.

    If the dynamics of the system can be described by the master equation, then the dynamics data 
    `tspan`, `rho0`, `H0` and `dH` shoule be input. `tspan` is the time length for the evolution, 
    `rho0` represents the density matrix of the initial state, `H0` and `dH` are the free 
    Hamiltonian and its derivatives on the unknown parameters to be estimated. `H0` is a matrix 
    when the free Hamiltonian is time-independent and a list of matrices with the length equal to 
    `tspan` when it is time-dependent. `dH` should be input as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. 
    `Hc` and `ctrl` are two lists represent the control Hamiltonians and the corresponding control 
    coefficients. `decay` contains decay operators $(\Gamma_1,\Gamma_2, \cdots)$ and the 
    corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule 
    decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. The default values for `decay`, 
    `Hc` and `ctrl` are empty which means the dynamics is unitary and only governed by the free 
    Hamiltonian. `dyn_method="expm"` represents the method for solving the dynamics is 
    matrix exponential, it can also be set as `dyn_method="ode"` which means the dynamics 
    (differential equation) is directly solved with the ODE solvers.

    The objective function for measurement optimization are CFI and $\mathrm{Tr}(W\mathcal{I}^
    {-1})$ with $\mathcal{I}$ the CFIM. $W$ corresponds to `W` in the objective function is the 
    weight matrix which defaults to the identity matrix.

=== "Julia"
    ``` jl
    opt = MeasurementOpt(mtype=:Projection, seed=1234)
    alg = DE(kwargs...)
    dynamics = Lindblad(opt, tspan, rho0, H0, dH; Hc=missing, 
                        ctrl=missing, decay=missing, dyn_method=:Expm)
    obj = CFIM_obj(W=missing)
    run(opt, alg, obj, dynamics; savefile=false)
    ```
    `mtype` represents the type of measurement optimization which defaults to `"Projection"`. 
    In this setting, rank-one projective measurement optimization will be performed. For the other 
    two measurement optimization scenarios this variable becomes `mtype="LC"` and `mtype="Rotation"`. 
    If the users want to find the optimal linear combination of an input measurement, the input 
    rule of `opt` is `MeasurementOpt(mtype=:LC,POVM_basis=[Pi1,Pi2,...],M_num=m)` with 
    `[Pi1,Pi2,...]` a set of POVM and `m` the number of operators of the output measurement. 
    For finding the optimal linear combination of an input measurement, the code `opt` is 
    `MeasurementOpt(mtype=:LC,POVM_basis=[Pi1,Pi2,...])`.  `seed` is the random seed which can 
    ensure the reproducibility of results.
    
    If the dynamics of the system can be described by the master equation, then the dynamics data 
    `tspan`, `rho0`, `H0` and `dH` shoule be input. `tspan` is the time length for the evolution, 
    `rho0` represents the density matrix of the initial state, `H0` and `dH` are the free 
    Hamiltonian and its derivatives on the unknown parameters to be estimated. `H0` is a matrix 
    when the free Hamiltonian is time-independent and a list of matrices with the length equal to 
    `tspan` when it is time-dependent. `dH` should be input as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. 
    `Hc` and `ctrl` are two lists represent the control Hamiltonians and the corresponding control 
    coefficients. `decay` contains decay operators $(\Gamma_1,\Gamma_2, \cdots)$ and the 
    corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule 
    decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. The default values for `decay`, 
    `Hc` and `ctrl` are `missing` which means the dynamics is unitary and only governed by the free 
    Hamiltonian. `dyn_method=:Expm` represents the method for solving the dynamics is 
    matrix exponential, it can also be set as `dyn_method=:Ode` which means the dynamics 
    (differential equation) is directly solved with the ODE solvers.

    The objective function for measurement optimization are CFI and $\mathrm{Tr}(W\mathcal{I}^
    {-1})$ with $\mathcal{I}$ the CFIM. $W$ corresponds to `W` in the objective function is the 
    weight matrix which defaults to the identity matrix. The variable `savefile` means whether to 
    save all the measurements. If set `false` the measurements in the final episode and the values 
    of the objective function in all episodes will be saved, if `savefile=true` the measurements 
    and the values of the objective function obtained in all episodes will be saved during the 
    training. The algorithm used for optimizing the measurements are PSO, DE and AD. `kwargs...` 
    is the keywords and default values corresponding to the optimization algorithm which will be 
    introduced in detail below.

---
## **PSO**
The code for measurement optimization with PSO is as follows
=== "Python"
    ``` py
    m = MeasurementOpt(method="PSO", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "measurement0":[], "max_episode":[1000,100], \ 
              "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
    ```
    The keywords and the default values of PSO can be seen in the following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "measurement0"                   | [ ]                        |
    | "max_episode"                    | [1000,100]                 |
    | "c0"                             | 1.0                        |
    | "c1"                             | 2.0                        |
    | "c2"                             | 2.0                        |
    | "seed"                           | 1234                       |

    `p_num` is the number of particles, `c0`, `c1` and `c2` are the PSO parameters 
    representing the inertia weight, cognitive learning factor and social learning factor, 
    respectively. `max_episode` accepts both integers and arrays with two elements. If it is 
    an integer, for example `max_episode=1000`, it means the program will continuously run 
    1000 episodes. However, if it is an array, for example `max_episode=[1000,100]`, the 
    program will run 1000 episodes in total but replace control coefficients of all the 
    particles with global best every 100 episodes. `seed` is the random seed which can 
    ensure the reproducibility of the results. `measurement0` in the algorithm is a list 
    representing the initial guesses of measurements. In the case of projective measurement 
    optimization, the entry of `measurement0` is a list of arrays with the length equal to 
    the dimension of the system. In the cases of finding the optimal linear combination and 
    the optimal rotation of a given set of measurement, the entry of `measurement0` is a 
    2D-array and array, respectively. Here, an example of generating `measurement0` is given 
    as follows
    
    **Example 7.1**
    === "projection"
        ``` py
        from quanestimation import *
        import numpy as np

        # the dimension of the system
        dim = 6
        # generation of the entry of `measurement0`
        C = [[] for i in range(dim)] 
        for i in range(dim):
            r_ini = 2*np.random.random(dim)-np.ones(dim)
            r = r_ini/np.linalg.norm(r_ini)
            phi = 2*np.pi*np.random.random(dim)
            C[i] = [r[j]*np.exp(1.0j*phi[j]) for j in range(dim)]
        C = np.array(gramschmidt(C))
        measurement0 = [C]
        ```
    === "LC"
        ``` py
        from quanestimation import *
        import numpy as np

        # the dimension of the system
        dim = 6
        # a given set of measurement
        POVM_basis = SIC(dim)
        # the number of operators of the output measurement
        m = 4
        # generation of the entry of `measurement0`
        B = np.array([np.random.random(len(POVM_basis)) for i in range(m)])
        measurement0 = [B]
        ```
    === "rotation"
        ``` py
        from quanestimation import *
        import numpy as np

        # the dimension of the system
        dim = 6
        # a given set of measurement
        POVM_basis = SIC(dim)
        # generation of the entry of `measurement0`
        s = np.random.random(dim**2)
        measurement0 = [s]
        ```
    In this algorithm, the length of `measurement0` should be less than or equal to `p_num`.
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

    `p_num` is the number of particles, `c0`, `c1`, and `c2` are the PSO parameters representing 
    the inertia weight, cognitive learning factor, and social learning factor, respectively. 
    `max_episode` accepts both integers and arrays with two elements. If it is an integer, 
    for example `max_episode=1000`, it means the program will continuously run 1000 episodes. 
    However, if it is an array, for example `max_episode=[1000,100]`, the program will run 1000 
    episodes in total but replace control coefficients of all the particles with global best 
    every 100 episodes. `ini_particle` in the algorithm is a list representing the initial guesses 
    of measurements, In the case of projective measurement optimization, the entry of `ini_particle` 
    is a list of arrays with the length equal to the dimension of the system. In the cases of 
    finding the optimal linear combination and the optimal rotation of a given set of measurement, 
    the entry of `ini_particle` is a 2D-array and array, respectively. Here, an example of 
    generating `ini_particle` is given as follows

    **Example 7.1**
    === "Projection"
        ``` jl
        using QuanEstimation

        # the dimension of the system
        dim = 6
        # generation of the entry of `measurement0`
        C = [ComplexF64[] for _ in 1:dim]
		for i in 1:dim
			r_ini = 2*rand(dim) - ones(dim)
			r = r_ini/norm(r_ini)
			ϕ = 2pi*rand(dim)
			C[i] = [r*exp(im*ϕ) for (r,ϕ) in zip(r,ϕ)] 
		end
        C = QuanEstimation.gramschmidt(C)
        measurement0 = ([C],)
        ```
    === "LC"
        ``` jl
        using QuanEstimation

        # the dimension of the system
        dim = 6
        # a given set of measurement
        POVM_basis = QuanEstimation.SIC(dim)
        # the number of operators of the output measurement
        m = 4
        # generation of the entry of `measurement0`
        B = [rand(length(POVM_basis)) for _ in 1:m]
        measurement0 = ([B],)
        ```
    === "Rotation"
        ``` jl
        using QuanEstimation

        # the dimension of the system
        dim = 6
        # a given set of measurement
        POVM_basis = QuanEstimation.SIC(dim)
        # generation of the entry of `measurement0`
        s = rand(dim^2)
        measurement0 = ([s],)
        ```
    In this algorithm, the length of `measurement0` should be less than or equal to `p_num`.

## **DE**
The code for measurement optimization with DE is as follows
=== "Python"
    ``` py
    m = MeasurementOpt(method="DE", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"p_num":10, "measurement0":[], "max_episode":1000, \
              "c":1.0, "cr":0.5, "seed":1234}
    ```
    The keywords and the default values of DE can be seen in the following table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                        | 10                         |
    | "measurement0"                   | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |
    | "seed"                           | 1234                       |

    `p_num` represents the number of populations, `c` and `cr` are the mutation constant and the 
    crossover constant. Here `max_episode` is an integer representing the number of episodes, 
    the variable `measurement0` is the same with `measurement0` in PSO.
=== "Julia"
    ``` jl
    alg = DE(p_num=10, ini_population=missing, max_episode=1000, 
             c=1.0, cr=0.5)
    ```
        The keywords and the default values of DE can be seen in the following table

    | $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "p_num"                          | 10                         |
    | "ini_population"                 | [ ]                        |
    | "max_episode"                    | 1000                       |
    | "c"                              | 1.0                        |
    | "cr"                             | 0.5                        |

    `p_num` represents the number of populations, `c` and `cr` are the mutation constant 
    and the crossover constant. Here `max_episode` is an integer representing the number of 
    episodes, the variable `ini_population` is the same with `ini_particle` in PSO.

## **AD**
The code for measurement optimization with AD is as follows
=== "Python"
    ``` py
    com = MeasurementOpt(method="AD", **kwargs)
    ```
    where `kwargs` is of the form
    ``` py
    kwargs = {"Adam":False, "measurement0":[], "max_episode":300, \ 
              "epsilon":0.01, "beta1":0.90, "beta2":0.99}
    ```
    The keywords and the default values of AD can be seen in the following 
    table

    | $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
    | :----------:                     | :----------:               |
    | "Adam"                           | False                      |
    | "measurement0"                   | [ ]                        |
    | "max_episode"                    | 300                        |
    | "epsilon"                        | 0.01                       |
    | "beta1"                          | 0.90                       |
    | "beta2"                          | 0.99                       |

    The measurements will update according to the learning rate `"epsilon"` for `Adam=False`, 
    However,  Adam algorithm can be introduced to update the measurements which can be realized by 
    setting `Adam=True`. In this case, the Adam parameters include learning rate, the exponential 
    decay rate for the first moment estimates, and the second moment estimates can be set by the 
    user via `epsilon`, `beta1`, and `beta2`. The input rule of `measurement0` is the same with 
    that in PSO, but the length of `measurement0` is equal to one.
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

    The measurements will update according to the learning rate `"epsilon"` for `Adam=False`, 
    However,  Adam algorithm can be introduced to update the measurements which can be realized by 
    setting `Adam=true`. In this case, the Adam parameters include learning rate, the exponential 
    decay rate for the first moment estimates, and the second moment estimates can be set by the 
    user via `epsilon`, `beta1`, and `beta2`.

**Example 7.2**  
<a id="example7_2"></a>
A single qubit system whose dynamics is governed by

\begin{align}
\partial_t\rho=-i[H, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}
\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}
\sigma_{-},\rho\}\right),
\end{align}


where $H = \frac{1}{2}\omega \sigma_3$ is the free Hamiltonian with $\omega$ the frequency, 
$\sigma_{\pm}=(\sigma_1 \pm \sigma_2)/2$ and $\gamma_{+}$, $\gamma_{-}$ are decay rates.
Here $\sigma_{i}$ for $(i=1,2,3)$ is the  Pauli matrix.

In this case, the probe state is taken as $\frac{1}{\sqrt{2}}(|0\rangle +|1\rangle)$, 
$|0\rangle$ $(|1\rangle)$ is the eigenstate of $\sigma_3$ with respect to the eigenvalue 
$1$ $(-1)$. 
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
    # dissipation
    sp = np.array([[0., 1.], [0., 0.]])  
	sm = np.array([[0., 0.], [1., 0.]]) 
	decay = [[sp, 0.], [sm, 0.1]]
    # generation of a set of POVM basis
    dim = np.shape(rho0)[0]
    POVM_basis = SIC(dim)
    # time length for the evolution
    tspan = np.linspace(0., 10., 2500)
    ```
    === "projection"
        === "DE"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, \
                               method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], \
                         "max_episode":[1000,100], "c0":1.0, \
					    "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, \
                               method="PSO", **PSO_paras)
		    ```
    === "LC"
        === "DE"
		    ``` py
            M_num = 2
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
				        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], \
                               savefile=False, method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            M_num = 2
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], "max_episode":[1000,100], \
					     "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], \
                               savefile=False, method="PSO", **PSO_paras)
		    ```
        === "AD"
		    ``` py
            M_num = 2
            # measurement optimization algorithm: AD
		    AD_paras = {"Adam":False, "measurement0":[], "max_episode":300, \
                        "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], \
                               savefile=False, method="AD", **AD_paras)
		    ```
    === "rotation"
        === "DE"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], \
                         "max_episode":[1000,100], "c0":1.0, \
					     "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="PSO", **PSO_paras)
		    ```
        === "AD"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: AD
		    AD_paras = {"Adam":False, "measurement0":[], "max_episode":300, \
                        "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="AD", **AD_paras)
		    ```
    ``` py
    # input the dynamics data
    m.dynamics(tspan, rho0, H0, dH, decay=decay, dyn_method="expm")
    # objective function: CFI
    m.CFIM()
    # convert the ".csv" file to the ".npy" file
    M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
    csv2npy_measurements(M_, M_num)
    # load the measurements
    M = np.load("measurements.npy")[-1]
    ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using Random
    using StableRNGs
    using LinearAlgebra
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
    # dissipation
    sp = [0. 1.; 0. 0.0im]
	sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.], [sm, 0.1]]
    # generation of a set of POVM basis
    dim = size(rho0, 1)
    POVM_basis = QuanEstimation.SIC(dim)
    M_num = dim
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    ```
    === "Projection"
        ``` jl
        opt = QuanEstimation.MeasurementOpt(mtype=:Projection, seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
    === "LC"
        ``` jl
        opt = QuanEstimation.MeasurementOpt(mtype=:LC, 
                                            POVM_basis=POVM_basis, M_num=M_num, 
                                            seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
        === "AD"
            ``` jl
            # measurement optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ```
    === "Rotation"
        ``` jl
        opt = QuanEstimation.MeasurementOpt(mtype=:Rotation, 
                                            POVM_basis=POVM_basis, seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
        === "AD"
            ``` jl
            # measurement optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ```
    ``` jl
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan ,rho0, H0, dH, 
                                       decay=decay, dyn_method=:Expm)
    # objective function: CFI
    obj = QuanEstimation.CFIM_obj()
    # run the measurement optimization problem
    QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

    # convert the flattened data into a list of matrix
    M_ = readdlm("measurements.csv",'\t', Complex{Float64})
    M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
        for j in 1:Int(length(M_[:,1])/M_num)][end]
    ```

**Example 7.3**  
<a id="example7_3"></a>
In the multiparameter scenario, the dynamics of electron and nuclear coupling in NV$^{-}$ can be expressed as
\begin{align}
\partial_t\rho=-i[H_0,\rho]+\frac{\gamma}{2}(S_3\rho S_3-S^2_3\rho-\rho S^2_3)
\end{align}

with $\gamma$ the dephasing rate. And
\begin{align}
H_0/\hbar=DS^2_3+g_{\mathrm{S}}\vec{B}\cdot\vec{S}+g_{\mathrm{I}}\vec{B}\cdot\vec{I}+\vec{S}^{\,\mathrm{T}}\mathcal{A}\vec{I}
\end{align}

is the free evolution Hamiltonian, where $\vec{S}=(S_1,S_2,S_3)^{\mathrm{T}}$ and $\vec{I}=(I_1,I_2,I_3)^{\mathrm{T}}$ 
with $S_i=s_i\otimes I$ and $I_i=I\otimes \sigma_i$ $(i=1,2,3)$ the electron and nuclear operators. 
$s_1, s_2, s_3$ are spin-1 operators with 

\begin{eqnarray}
s_1 = \frac{1}{\sqrt{2}}\left(\begin{array}{ccc}
0 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 0
\end{array}\right),
s_2 = \frac{1}{\sqrt{2}}\left(\begin{array}{ccc}
0 & -i & 0\\
i & 0 & -i\\
0 & i & 0
\end{array}\right), \nonumber
\end{eqnarray}

and $s_3=\mathrm{diag}(1,0,-1)$ and $\sigma_i (i=1,2,3)$ is Pauli matrix. $\mathcal{A}=\mathrm{diag}
(A_1,A_1,A_2)$ is the hyperfine tensor with $A_1$ and $A_2$ the axial and transverse magnetic hyperfine coupling coefficients.
The coefficients $g_{\mathrm{S}}=g_\mathrm{e}\mu_\mathrm{B}/\hbar$ and $g_{\mathrm{I}}=g_\mathrm{n}\mu_\mathrm{n}/\hbar$, 
where $g_\mathrm{e}$ ($g_\mathrm{n}$) is the $g$ factor of the electron (nuclear), $\mu_\mathrm{B}$ ($\mu_\mathrm{n}$) is 
the Bohr (nuclear) magneton and $\hbar$ is the Plank's constant. $\vec{B}$ is the magnetic field which be estimated.

In this case,the initial state is taken as $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)\otimes|\!\!\uparrow\rangle$, 
where $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)$ is an electron state with $|1\rangle$ ($|\!-\!1\rangle$) the 
eigenstate of $s_3$ with respect to the eigenvalue $1$ ($-1$). $|\!\!\uparrow\rangle$ is a nuclear state and 
the eigenstate of $\sigma_3$ with respect to the eigenvalue 1. $W$ is set to be $I$.

Here three types of measurement optimization are considerd, projective measurement, linear combination of a given set of positive operator-valued measure (POVM) and optimal rotated measurement of an input measurement.
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
    # generation of a set of POVM basis
    dim = len(rho0)
    POVM_basis = [np.dot(basis(dim, i), basis(dim, i).conj().T) \
                  for i in range(dim)]
    # time length for the evolution
    tspan = np.linspace(0., 2., 4000)
    ```
    === "projection"
        === "DE"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, \
                               method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], \
                         "max_episode":[1000,100], "c0":1.0, \
					    "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, \
                               method="PSO", **PSO_paras)
		    ```
    === "LC"
        === "DE"
		    ``` py
            M_num = 4
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
				        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], \
                               savefile=False, method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            M_num = 4
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], "max_episode":[1000,100], \
					     "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], \
                               savefile=False, method="PSO", **PSO_paras)
		    ```
        === "AD"
		    ``` py
            M_num = 4
            # measurement optimization algorithm: AD
		    AD_paras = {"Adam":False, "measurement0":[], "max_episode":300, \
                        "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, M_num], \
                               savefile=False, method="AD", **AD_paras)
		    ```
    === "rotation"
        === "DE"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], \
                         "max_episode":[1000,100], "c0":1.0, \
					     "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="PSO", **PSO_paras)
		    ```
        === "AD"
		    ``` py
            M_num = dim
            # measurement optimization algorithm: AD
		    AD_paras = {"Adam":False, "measurement0":[], "max_episode":300, \
                        "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="AD", **AD_paras)
		    ```
    ``` py
    # input the dynamics data
    m.dynamics(tspan, rho0, H0, dH, decay=decay, dyn_method="expm")
    # objective function: tr(WI^{-1})
    m.CFIM()
    # convert the ".csv" file to the ".npy" file
    M_ = np.loadtxt("measurements.csv", dtype=np.complex128)
    csv2npy_measurements(M_, M_num)
    # load the measurements
    M = np.load("measurements.npy")[-1]
    ```
=== "Julia"
    ``` jl
    using QuanEstimation
    using Random
    using LinearAlgebra
    using DelimitedFiles

    # initial state
    rho0 = zeros(ComplexF64, 6, 6)
    rho0[1:4:5, 1:4:5] .= 0.5
    # Hamiltonian
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
    # generation of a set of POVM basis
    dim = size(rho0, 1)
    POVM_basis = [QuanEstimation.basis(dim, i)*QuanEstimation.basis(dim, i)' 
                  for i in 1:dim]
    # time length for the evolution
    tspan = range(0., 2., length=4000)
    ```
    === "Projection"
        ``` jl
        M_num = dim
        opt = QuanEstimation.MeasurementOpt(mtype=:Projection, seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
    === "LC"
        ``` jl
        M_num = 4
        opt = QuanEstimation.MeasurementOpt(mtype=:LC, 
                                            POVM_basis=POVM_basis, M_num=M_num, 
                                            seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
        === "AD"
            ``` jl
            # measurement optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ```
    === "Rotation"
        ``` jl
        M_num = dim
        opt = QuanEstimation.MeasurementOpt(mtype=:Rotation, 
                                            POVM_basis=POVM_basis, seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
        === "AD"
            ``` jl
            # measurement optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ```
    ``` jl
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan ,rho0, H0, dH, 
                                       decay=decay, dyn_method=:Expm)
    # objective function: CFI
    obj = QuanEstimation.CFIM_obj()
    # run the measurement optimization problem
    QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
    M_ = readdlm("measurements.csv",'\t', Complex{Float64})
    M = [[reshape(M_[i,:], dim, dim) for i in 1:M_num] 
        for j in 1:Int(length(M_[:,1])/M_num)][end]
    ```
If the parameterization process is implemented with the Kraus operators, then the corresponding 
parameters should be input via

=== "Python"
    ``` py
    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, 
                       method="DE", **kwargs)
    m.Kraus(K, dK)
    m.CFIM(W=[])
    ```
=== "Julia"
    ``` jl
    opt = MeasurementOpt(mtype=:Projection, seed=1234)
    alg = DE(kwargs...)
    dynamics = Kraus(opt, K, dK)
    obj = CFIM_obj(W=missing)
    run(opt, alg, obj, dynamics; savefile=false)
    ```
where `K` and `dK` are the Kraus operators and its derivatives with respect to the 
unknown parameters.

**Example 7.4**  
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
    dK1 = np.array([[1., 0.], [0.0, -0.5/np.sqrt(1-gamma)]])
    dK2 = np.array([[0., 0.5/np.sqrt(gamma)], [0., 0.]])
    dK = [[dK1], [dK2]]
    # measurement
    POVM_basis = SIC(len(rho0))
    M_num = 2
    ```
    === "projection"
        === "DE"
		    ``` py
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, \
                               method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], \
                         "max_episode":[1000,100], "c0":1.0, \
					    "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="projection", minput=[], savefile=False, \
                               method="PSO", **PSO_paras)
		    ```
    === "LC"
        === "DE"
		    ``` py
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
				        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, 2], \
                               savefile=False, method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], "max_episode":[1000,100], \
					     "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, 2], \
                               savefile=False, method="PSO", **PSO_paras)
		    ```
        === "AD"
		    ``` py
            # measurement optimization algorithm: AD
		    AD_paras = {"Adam":False, "measurement0":[], "max_episode":300, \
                        "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            m = MeasurementOpt(mtype="input", minput=["LC", POVM_basis, 2], \
                               savefile=False, method="AD", **AD_paras)
		    ```
    === "rotation"
        === "DE"
		    ``` py
            # measurement optimization algorithm: DE
		    DE_paras = {"p_num":10, "measurement0":[], "max_episode":1000, \
                        "c":1.0, "cr":0.5, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="DE", **DE_paras)
		    ```
        === "PSO"
		    ``` py
            # measurement optimization algorithm: PSO
		    PSO_paras = {"p_num":10, "measurement0":[], \
                         "max_episode":[1000,100], "c0":1.0, \
					     "c1":2.0, "c2":2.0, "seed":1234}
		    m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="PSO", **PSO_paras)
		    ```
        === "AD"
		    ``` py
            # measurement optimization algorithm: AD
		    AD_paras = {"Adam":False, "measurement0":[], "max_episode":300, \
                        "epsilon":0.01, "beta1":0.90, "beta2":0.99}
            m = MeasurementOpt(mtype="input", minput=["rotation", POVM_basis], \
                               savefile=False, method="AD", **AD_paras)
		    ```
    ``` py
    # input the dynamics data
    m.Kraus(rho0, K, dK)
    # objective function: CFI
    m.CFIM()
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
    # Kraus operators for the amplitude damping channel
    gamma = 0.1
    K1 = [1. 0.; 0. sqrt(1-gamma)]
    K2 = [0. sqrt(gamma); 0. 0.]
    K = [K1, K2]
    # derivatives of Kraus operators on gamma
    dK1 = [1. 0.; 0. -0.5/sqrt(1-gamma)]
    dK2 = [0. 0.5/sqrt(gamma); 0. 0.]
    dK = [[dK1], [dK2]]
    # measurement
    dim = size(rho0,1)
    POVM_basis = QuanEstimation.SIC(dim)
    M_num = 2
    ```
    === "Projection"
        ``` jl
        opt = QuanEstimation.MeasurementOpt(mtype=:Projection, seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
    === "LC"
        ``` jl
        opt = QuanEstimation.MeasurementOpt(mtype=:LC, 
                                            POVM_basis=POVM_basis, M_num=2, 
                                            seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
        === "AD"
            ``` jl
            # measurement optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ```
    === "Rotation"
        ``` jl
        opt = QuanEstimation.MeasurementOpt(mtype=:Rotation, 
                                            POVM_basis=POVM_basis, seed=1234)
        ```
        === "DE"
            ``` jl
            # measurement optimization algorithm: DE
            alg = QuanEstimation.DE(p_num=10, ini_population=missing, 
                                    max_episode=1000, c=1.0, cr=0.5)
            ```
        === "PSO"
            ``` jl
            # measurement optimization algorithm: PSO
            alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, 
                                     max_episode=[1000,100], c0=1.0, c1=2.0, 
                                     c2=2.0)
            ```
        === "AD"
            ``` jl
            # measurement optimization algorithm: AD
            alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
                                    beta1=0.90, beta2=0.99)
            ```
    ``` jl
    # input the dynamics data
    dynamics = QuanEstimation.Kraus(opt, rho0, K, dK)
    # objective function: CFI
    obj = QuanEstimation.CFIM_obj()
    # run the measurement optimization problem
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

