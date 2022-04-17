# **Overview**
To import QuanEstimation, the users can call the code
=== "Python"
    ``` py
    from quanestimation import *
    ```
=== "Julia"
    ``` jl
    using quanestimation
    ```
    
---
## **Dynamics**
In QuanEstimation, two types of parameterization processes are considered. The first one is the 
master equation of the form

\begin{align}
\partial_t\rho &=\mathcal{L}\rho \nonumber \\
&=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
\end{align}

where $\rho$ is the evolved density matrix, H is the Hamiltonian of the system, $\Gamma_i$ and 
$\gamma_i$ are the $i\mathrm{th}$ decay operator and the corresponding decay rate. Numerically, 
the evolved state at $j$th time interval is obtained by $\rho_j=e^{\Delta t\mathcal{L}}
\rho_{j-1}$ with $\Delta t$ the time interval. The derivatives of $\rho_j$ on $\textbf{x}$ is 
calculated as
<center> $\partial_{\textbf{x}}\rho_j =\Delta t(\partial_{\textbf{x}}\mathcal{L})\rho_j
+e^{\Delta t \mathcal{L}}(\partial_{\textbf{x}}\rho_{j-1}),$ </center> <br>
where $\rho_{j-1}$ is the evolved density matrix at $(j-1)$th time interval.

The evolved density matrix $\rho$ and its derivatives ($\partial_{\textbf{x}}\rho$) on 
$\textbf{x}$  can be calculated via the codes
=== "Python"
    ``` py
    dynamics = Lindblad(tspan, rho0, H0, dH, decay=[], Hc=[], ctrl=[])
    rho, drho = dynamics.expm()
    ```
=== "Julia"
    ``` jl
    rho, drho = expm(tspan, rho0, H0, dH, decay=missing, Hc=missing, 
                     ctrl=missing)
    ```
Here `tspan` is the time length for the evolution, `rho0` represents the density matrix of the
probe state, `H0` and `dH` are the free Hamiltonian and its derivatives on the unknown 
parameters to be estimated. `H0` is a matrix when the free Hamiltonian is time-independent and 
a list with the length equal to `tspan` when it is time-dependent. `dH` should be input as 
$[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `decay` contains decay operators 
$(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ 
with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. `Hc` and 
`ctrl` are two lists represent the control Hamiltonians and the corresponding control 
coefficients. The default values for `decay`, `Hc` and `ctrl` are empty which means the 
dynamics is unitary and only governed by the free Hamiltonian.

The output (`rho` and `drho`) of this class by calling `dynamics.expm()` are two lists with 
the length equal to `tspan`. Here `rho` represents the density matrix and `drho` is the 
corresponding derivatives on all the parameters, the $i$th entry of `drho` is 
$[\partial_a{\rho},\partial_b{\rho},\cdots].$

**Example**  
In this example, the free evolution Hamiltonian of a single qubit system is $H_0=\frac{1}{2}
\omega_0 \sigma_3$ with $\omega_0$ the frequency and $\sigma_3$ a Pauli matrix. 
The dynamics of the system is governed by
\begin{align}
\partial_t\rho=-i[H_0, \rho],
\end{align}

where $\rho$ is the parameterized density matrix. The probe state is taken as $|+\rangle\langle+|$ 
with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is the 
eigenstate of $\sigma_3$ (Pauli matrix) with respect to the eigenvalue $1$ $(-1)$.

=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # initial state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    omega0 = 1.0
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5*omega0*sz
    # derivative of the free Hamiltonian on omega0
    dH = [0.5*sz]
    # time length for the evolution
    tspan = np.linspace(0., 10., 2500)
    # dynamics
    dynamics = Lindblad(tspan, rho0, H0, dH)
    rho, drho = dynamics.expm()
    ```
=== "Julia"
    ``` jl
    using QuanEstimation

    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega0 = 1.0
    sz = [1. 0.0im; 0. -1.]
	H0 = 0.5*omega0*sz
    # derivative of the free Hamiltonian on omega0
    dH = [0.5*sz]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # dynamics
    rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH)
    ```
If the parameterization process is implemented with the Kraus operators
\begin{align}  
\rho=\sum_i K_i\rho_0K_i^{\dagger},
\end{align}

where $K_i$ is a Kraus operator satisfying $\sum_{i}K^{\dagger}_i K_i=I$ with $I$ 
the identity operator. $\rho$ and $\partial_{\textbf{x}}\rho$ can be solved by
=== "Python"
    ``` py
    rho, drho = Kraus(K, dK, rho0)
    ```
=== "Julia"
    ``` jl
    Kraus = Kraus(K, dK, rho0)
    rho, drho = evolve(Kraus)
    ```
where `K` and `dK` are the Kraus operators and its derivatives on the unknown parameters.

**Example**  
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

where $\gamma$ is the decay probability. The parameterized density matrix can be calculated
via $\rho=\sum_i K_i\rho_0K_i^{\dagger}$ and corresponding derivatives on the unknown
parameters are $\partial_{\textbf{x}}\rho=\sum_i \partial_{\textbf{x}}K_i\rho_0K_i^{\dagger}
+ K_i\rho_0\partial_{\textbf{x}}K_i^{\dagger}$ with $\rho_0$ the probe state. In this example,
the probe state is taken as $|+\rangle\langle+|$ with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+
|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is the eigenstate of $\sigma_3$ (Pauli matrix) with 
respect to the eigenvalue $1$ $(-1)$.

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
    # parameterization process
    rho, drho = Kraus(K, dK, rho0)
    ```
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
    # parameterization process
    Kraus = QuanEstimation.Kraus(K, dK, rho0)
    rho, drho = QuanEstimation.evolve(Kraus)
    ```

---
## **Metrological resources**
The metrological resources that QuanEstimation can calculate are spin squeezing and the 
minimum time to reach the given target. The spin squeezing can be calculated via the function: 
=== "Python"
    ``` py
    SpinSqueezing(rho, basis="Dicke", output="KU")
    ```
=== "Julia"
    ``` jl
    SpinSqueezing(rho; basis="Dicke", output="KU")
    ```
`rho` represents the density matrix of the state. In this function, the basis of the state can 
be Dicke basis or the original basis of each spin, which can be adjusted by setting 
`basis="Dicke"` or `basis="Pauli"`. The variable `output` represents the type of spin squeezing 
calculation. `output="KU"` represents the spin squeezing defined by Kitagawa and Ueda 
[[1]](#Kitagawa1993) and `output="WBIMH"` calculates the spin squeezing defined by Wineland 
et al. [[2]](#Wineland1992).

**Example**  
In this example, QuTip [[3,4]](#Johansson2012) is used to generate spin coherent state.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np
    from qutip import spin_coherent
    
    # generation of spin coherent state with QuTip
    j = 2
    theta = 0.5*np.pi
    phi = 0.5*np.pi
    rho_CSS = spin_coherent(j, theta, phi, type='dm').full()
    xi = SpinSqueezing(rho_CSS, basis="Dicke", output="KU")
    ```
=== "Julia"
    ``` jl
    using QuanEstimation

    rho = [0.25 -0.35355339im -0.25; 
           0.35355339im 0.5 -0.35355339im;
           -0.25 0.35355339im 0.25]
    xi = QuanEstimation.SpinSqueezing(rho; basis="Dicke", output="KU")
    ```

Calculation of the time to reach a given precision limit with
=== "Python"
    ``` py
    TargetTime(f, tspan, func, *args, **kwargs)
    ```
=== "Julia"
    ``` jl
    TargetTime(f, tspan, func, args...; kwargs...)
    ```
where `f` is the given value of the objective function and `tspan` is the time length for the 
evolution. `func` represents the function for calculating the objective function, `*args` and 
`**kwargs` are the corresponding input parameters and the keyword arguments.

**Example**  
In this example, the free evolution Hamiltonian of a single qubit system is $H_0=\frac{1}{2}
\omega_0 \sigma_3$ with $\omega_0$ the frequency and $\sigma_3$ a Pauli matrix. 
The dynamics of the system is governed by
\begin{align}
\partial_t\rho=-i[H_0, \rho],
\end{align}

where $\rho$ is the parameterized density matrix. The probe state is taken as $|+\rangle\langle+|$ 
with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is the 
eigenstate of $\sigma_3$ (Pauli matrix) with respect to the eigenvalue $1$ $(-1)$.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # initial state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    omega0 = 1.0
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5*omega0*sz
    # derivative of the free Hamiltonian on omega0
    dH = [0.5*sz]
    # time length for the evolution
    tspan = np.linspace(0., 50., 2000)
    # dynamics
    dynamics = Lindblad(tspan, rho0, H0, dH)
    rho, drho = dynamics.expm()
    # the value of the objective function
    f = 20.0
    t = TargetTime(f, tspan, QFIM, rho, drho)
    ```
=== "Julia"
    ``` jl
    using QuanEstimation

    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega0 = 1.0
    sx = [0. 1.; 1. 0.0im]
	sy = [0. -im; im 0.]
	sz = [1. 0.0im; 0. -1.]
	H0 = 0.5*omega0*sz
    # derivative of the free Hamiltonian on omega0
    dH = [0.5*sz]
    # time length for the evolution
    tspan = range(0., 50., length=2000)
    # dynamics
    rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH)
    # the value of the objective function
    f = 20
    t = QuanEstimaion.TargetTime(f, tspan, QuanEstimation.QFIM, rho, drho)
    ```

---
## **Bibliography**
<a id="Kitagawa1993">[1]</a> 
M. Kitagawa and M. Ueda, Squeezed spin states, 
[Phys. Rev. A **47**, 5138 (1993).](https://doi.org/10.1103/PhysRevA.47.5138)

<a id="Wineland1992">[2]</a>
D. J. Wineland, J. J. Bollinger, W. M. Itano, F. L. Moore, and D. J. Heinzen, 
Spin squeezing and reduced quantum noise in spectroscopy, 
[Phys. Rev. A **46**, R6797(R) (1992).](https://doi.org/10.1103/PhysRevA.46.R6797)

<a id="Johansson2012">[3]</a>
J. R. Johansson, P. D. Nation, and F. Nori,
QuTiP: An open-source Python framework for the dynamics of open quantum systems,
[Comp. Phys. Comm. **183**, 1760 (2012).](https://doi.org/10.1016/j.cpc.2012.02.021)

<a id="Johansson2013">[4]</a>
J. R. Johansson, P. D. Nation, and F. Nori,
QuTiP 2: A Python framework for the dynamics of open quantum systems,
[Comp. Phys. Comm. **184**, 1234 (2013).](https://doi.org/10.1016/j.cpc.2012.11.019)
