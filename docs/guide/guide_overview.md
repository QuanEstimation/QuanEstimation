# **Overview**
QuanEstimation is a Python-Julia based open-source toolkit for quantum parameter estimation.
It can be used to calculate several well-used metrological tools and optimize probe state, control, 
measurement. The users can call the code
``` py
from quanestimation import *
```
to import QuanEstimation in python, and call
``` jl
using quanestimation
```
in Julia.

---

## **Dynamics**
In QuanEstimation, two types of parameterization of a state are considered. The first one is the 
master equation of the form

\begin{align}
\partial_t\rho &=\mathcal{L}\rho \nonumber \\
&=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
\end{align}

where $\rho$ is the evolved density matrix, H is the Hamiltonian of the system, $\Gamma_i$ and 
$\gamma_i$ are the $i\mathrm{th}$ decay operator and corresponding decay rate. Numerically, 
the evolved state at $j$th time interval is obtained by $\rho_j=e^{\Delta t\mathcal{L}}\rho_{j-1}$ 
with $\Delta t$ the time interval. The derivatives of $\rho_j$ on $\textbf{x}$ is calculated as
<center> $\partial_{\textbf{x}}\rho_j =\Delta t(\partial_{\textbf{x}}\mathcal{L})\rho_j
+e^{\Delta t \mathcal{L}}(\partial_{\textbf{x}}\rho_{j-1}),$ </center> <br>
where $\rho_{j-1}$ is the evolved density matrix at $(j-1)$th time interval.

The evolved density matrix $\rho$ and its derivatives on $\textbf{x}$ ($\partial_{\textbf{x}}\rho$) can be 
calculated via the codes
``` py
dynamics = Lindblad(tspan, rho0, H0, dH, decay=[], Hc=[], ctrl=[])
rho, drho = dynamics.expm()
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
Here `tspan` is the time length for the evolution, `rho0` represents the density matrix of the initial 
state, `H0` and `dH` are the free Hamiltonian and its derivatives on the unknown parameters to be estimated. `H0` is a matrix when the free Hamiltonian is time-independent and a list with the length equal to `tspan` 
when it is time-dependent. `dH` should be input as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `decay` 
contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, 
\gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. 
`Hc` and `ctrl` are two lists represent the control Hamiltonians and the corresponding control coefficients. 
The default values for `decay`, `Hc` and `ctrl` are empty which means the dynamics is unitary 
and only governed by the free Hamiltonian.

The output (`rho` and `drho`) of this class by calling `dynamics.expm()` are two lists with 
the length equal to `tspan`. Here `rho` represents the density matrix and `drho` is the corresponding 
derivatives on all the parameters. The $i$th entry of `drho` is $[\partial_a{\rho},\partial_b{\rho},\cdots].$

The other is to parameterize a state by the Kraus operators
\begin{align}  
\rho=\sum_i K_i\rho_0K_i^{\dagger},
\end{align}

where $K_i$ is a Kraus operator satisfying $\sum_{i}K^{\dagger}_i K_i=I$ with $I$ 
the identity operator. $\rho$ and $\partial_{\textbf{x}}\rho$ can be solved by
``` py
rho, drho = kraus(rho0, K, dK)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `K` and `dK` are the Kraus operators and its derivatives on the unknown parameters.

---

## **Metrological resources**
The metrological resources that QuanEstimation can calculate are spin squeezing and the time 
to reach the given target. The spin squeezing can be calculated via
``` py
SpinSqueezing(rho, basis="Dicke", output="KU")
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
`rho` represents the density matrix of the state. In this function, the basis of the state can 
be Dicke basis or the original basis of each spin, which can be adjusted by setting `basis="Dicke"` 
or `basis="Pauli"`. The variable `output` represents the type of spin squeezing calculation. 
`output="KU"` represents the spin squeezing defined by Kitagawa and Ueda <span style="color:red">(cite) </span>
and `output="WBIMH"` returns the spin squeezing defined by Wineland et al <span style="color:red">(cite) </span>.

Calculation of the time to reach a given precision limit with
``` py
TargetTime(f,tspan, func, *args, **kwargs)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `f` is the given value of the objective function and `tspan` is the time length for the evolution. 
`func` represents the function for calculating the objective function, `*args` and `**kwargs` are 
the corresponding input parameters and the keyword arguments.

---
