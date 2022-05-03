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

**Example 2.1**  
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
    rho, drho = Kraus(rho0, K, dK)
    ```
=== "Julia"
    ``` jl
    Kraus = Kraus(rho0, K, dK)
    rho, drho = evolve(Kraus)
    ```
where `K` and `dK` are the Kraus operators and its derivatives on the unknown parameters.

**Example 2.2**  
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
    rho, drho = Kraus(rho0, K, dK)
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
    Kraus = QuanEstimation.Kraus(rho0, K, dK)
    rho, drho = QuanEstimation.evolve(Kraus)
    ```

