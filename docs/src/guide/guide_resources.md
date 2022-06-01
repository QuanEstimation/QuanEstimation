# **Metrological resources**
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

**Example 4.1**  
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
    using SparseArrays

    # generation of the coherent spin state
    j, theta, phi = 2, 0.5pi, 0.5pi
    Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
    Jm = Jp'
    psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
           QuanEstimation.basis(Int(2*j+1), 1)
    rho = psi0*psi0'
    xi = QuanEstimation.SpinSqueezing(rho; basis="Dicke", output="KU")
    ```
Calculation of the minimum time to reach a given precision limit with
=== "Python"
    ``` py
    TargetTime(f, tspan, func, *args, **kwargs)
    ```
    where `f` is the given value of the objective function and `tspan` is the time length for the 
    evolution. `func` represents the function for calculating the objective function, `*args` and 
    `**kwargs` are the corresponding input parameters and the keyword arguments.
=== "Julia"
    ``` jl
    TargetTime(f, tspan, func, args...; kwargs...)
    ```
    where `f` is the given value of the objective function and `tspan` is the time length for the 
    evolution. `func` represents the function for calculating the objective function, `args...` 
    and `kwargs...` are the corresponding input parameters and the keyword arguments.

**Example 4.2**  
In this example, the free evolution Hamiltonian of a single qubit system is $H_0=\frac{1}{2}
\omega \sigma_3$ with $\omega$ the frequency and $\sigma_3$ a Pauli matrix. 
The dynamics of the system is governed by
\begin{align}
\partial_t\rho=-i[H_0, \rho],
\end{align}

where $\rho$ is the parameterized density matrix. The probe state is taken as $|+\rangle\langle+|$ 
with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is the 
eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ $(-1)$.
=== "Python"
    ``` py
    from quanestimation import *
    import numpy as np

    # initial state
    rho0 = 0.5*np.array([[1., 1.], [1., 1.]])
    # free Hamiltonian
    omega = 1.0
    sz = np.array([[1., 0.], [0., -1.]])
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
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
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
	sy = [0. -im; im 0.]
	sz = [1. 0.0im; 0. -1.]
	H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # time length for the evolution
    tspan = range(0., 50., length=2000)
    # dynamics
    rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH)
    drho = [drho[i][1] for i in 1:2000]
    # the value of the objective function
    f = 20
    t = QuanEstimation.TargetTime(f, tspan, QuanEstimation.QFIM, rho, drho)
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
