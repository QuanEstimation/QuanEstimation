---
header-includes:
  - \usepackage{caption}
---

# **Control optimization**
The Hamiltonian of a controlled system can be written as
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters $\textbf{x}$ 
and $H_k$ represents the $k$th control Hamiltonian with $u_k$ the correspong control 
coefficient. In QuanEstimation, different algorithms are invoked to update the optimal 
control coefficients. The control optimization algorithms are gradient ascent pulse 
engineering (GRAPE) [[1,2,3]](#Khaneja2005), GRAPE algorithm based on the automatic 
differentiation (auto-GRAPE) [[4]](#Baydin2018), particle swarm optimization (PSO) 
[[5]](#Kennedy1995), differential evolution (DE) [[6]](#Storn1997) and deep deterministic 
policy gradients (DDPG) [[7]](#Lillicrap2015). The codes for control optimization are
```py
control = ControlOpt(savefile=False, method="auto-GRAPE", **kwargs)
control.dynamics(tspan, rho0, H0, dH, Hc, decay=[], ctrl_bound=[])
control.QFIM(W=[], LDtype="SLD")
control.CFIM(M=[], W=[])
control.HCRB(W=[])
```
In QuanEstimation, the optimization codes are executed in Julia and the data will be saved in
the `.csv` file. The variable `savefile` indicates whether to save all the control 
coefficients and its default value is `False` which means the control coefficients for the 
final episode and the values of the objective function in all episodes will be saved in 
"controls.csv" and "f.csv", respectively. If set `True` then the control coefficients and the 
values of the objective function in all episodes will be saved during the training. The package
contains five control optimization algorithms which can be set via `method`. `**kwargs` is the
keyword and default value corresponding to the optimization algorithm which will be 
introduced in detail below.

After calling `control = ControlOpt()`, the dynamics parameters shoule be input. Here `tspan` 
is the time length for the evolution and `rho0` represents the density matrix of the initial state. `H0` and `dH` are the free Hamiltonian and its derivatives on the unknown 
parameters to be estimated. `H0` is a matrix when the free Hamiltonian is time-independent and 
a list with the length equal to `tspan` when it is time-dependent. `dH` should be input as 
$[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` is a list representing the control 
Hamiltonians. `decay` contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the 
corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule 
decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. The default values for 
`decay`, `Hc` and `ctrl` are empty which means the dynamics is unitary and only governed by 
the free Hamiltonian. The package can be used to optimize bounded control problems by setting 
lower and upper bounds of the control coefficients via `ctrl_bound`, which is an array with 
two elements representing the lower and upper bound of the control coefficients, respectively. 
The default value of `ctrl_bound=[]` which means the control coefficients are in the regime 
$[-\infty,\infty]$.

The code is `control.QFIM()` for the objective functions are QFI and $\mathrm{Tr}(W\mathcal{F}
^{-1})$, `control.CFIM()` for CFI and $\mathrm{Tr}(W\mathcal{I}^{-1})$ and `control.HCRB()` for 
HCRB. Here $F$ and $I$ are the QFIM and CFIM, $W$ corresponds to `W` is the weight matrix which 
defaults to the identity matrix. If the users call `control.HCRB()` for single parameter 
scenario, the program will exit and print `"Program exit. In single parameter scenario, HCRB is 
equivalent to QFI. Please choose QFIM as the target function"`. `LDtype` in `state.QFIM()` 
represents the types of the QFIM, it can be set as `LDtype=SLD` (default), `LDtype=RLD`, and 
`LDtype=LLD`. `M` in `control.CFIM()` represents a set of positive operator-valued measure 
(POVM) with default value `[]` which means a set of rank-one symmetric informationally complete 
POVM (SIC-POVM) is used.

---
## **GRAPE and auto-GRAPE**
The codes for control optimization with GRAPE and auto-GRAPE are as follows
``` py
control = ControlOpt(method="GRAPE", **kwargs)
```
``` py
control = ControlOpt(method="auto-GRAPE", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"Adam":True, "ctrl0":[], "max_episode":300, "epsilon":0.01, 
          "beta1":0.90, "beta2":0.99}
```
The keywords and the default values of GRAPE and auto-GRAPE can be seen in the following table

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "Adam"                           | True                       |
| "ctrl0"                          | [ ]                        |
| "max_episode"                    | 300                        |
| "epsilon"                        | 0.01                       |
| "beta1"                          | 0.90                       |
| "beta2"                          | 0.99                       |

Adam algorithm can be introduced to update the control coefficients when using GRAPE and 
auto-GRAPE for control optimization, which can be realized by setting `Adam=True`. In this 
case, the Adam parameters include learning rate, the exponential decay rate for the first 
moment estimates and the second moment estimates can be set by the users via `epsilon`, `beta1`
and `beta2`. If `Adam=False`, the control coefficients will update according to the learning 
rate `"epsilon"`. `ctrl0` is a list representing the initial guesses of control coefficients 
and `max_episode` is the the number of episodes.

## **PSO**
The code for control optimization with PSO is as follows
``` py
control = ControlOpt(method="PSO", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"particle_num":10, "ctrl0":[], "max_episode":[1000,100], 
          "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "particle_num"                   | 10                         |
| "ctrl0"                          | [ ]                        |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |
| "seed"                           | 1234                       |

Here `particle_num` is the the number of particles. `c0`, `c1` and `c2` are the PSO parameters 
representing the inertia weight, cognitive learning factor and social learning factor, 
respectively. `max_episode` accepts both integers and arrays with two elements. If it is an 
integer, for example `max_episode=1000`, it means the program will continuously run 1000 
episodes. However, if it is an array, for example `max_episode=[1000,100]`, the program will 
run 1000 episodes in total but replace control coefficients of all the particles with global 
best every 100 episodes. `seed` is the random seed which can ensure the reproducibility of results.

## **DE**
The code for control optimization with DE is as follows
``` py
control = ControlOpt(method="DE", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"popsize":10, "ctrl0":[], "max_episode":1000, "c":1.0, 
          "cr":0.5, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "popsize"                        | 10                         |
| "ctrl0"                          | [ ]                        |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |
| "seed"                           | 1234                       |

`popsize` and `max_episode` represent the number of populations and episodes. `c` and `cr` 
are the mutation constant and the crossover constant.

## **DDPG**
The code for control optimization with DDPG is as follows
``` py
control = ControlOpt(method="DDPG", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"layer_num":3, "layer_dim":200, "max_episode":1000, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "ctrl0"                          | [ ]                        |
| "max_episode"                    | 1000                       |
| "layer_num"                      | 3                          |
| "layer_dim"                      | 200                        |
| "seed"                           | 1234                       |

`layer_num` and `layer_dim` represent the number of layers (include the input and output layer) 
and the number of neurons in the hidden layer.

**Example 1**  
The Hamiltonian of a controlled system can be written as
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters $\textbf{x}$ and $H_k$ 
represents the $k$th control Hamiltonian with $u_k$ the corresponding control coefficient.

In this example, the free evolution Hamiltonian of a single qubit system is $H_0 = \frac{1}{2}\omega_0 \sigma_3$ with 
$\omega_0$ the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by
\begin{align}
\partial_t\rho=-i[H_0, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}\sigma_{-},\rho\}\right),
\end{align}

where $\gamma_{+}$, $\gamma_{-}$ are decay rates and $\sigma_{\pm}=(\sigma_1 \pm \sigma_2)/2$. The control Hamiltonian 
\begin{align}
H_\mathrm{c}=u_1(t)\sigma_1+u_2(t)\sigma_2+u_3(t)\sigma_3.
\end{align}

Here $\sigma_{1}$, $\sigma_{2}$ are also Pauli matrices. The probe state is taken as $|+\rangle$ and the measurement for CFI is $\{|+\rangle\langle+|, |-\rangle\langle-|\}$ with
$|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$. Here $|0\rangle(|1\rangle)$ is the eigenstate of $\sigma_3$ with respect to the eigenvalue $1(-1)$.

``` py
from quanestimation import *
import numpy as np

# initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
# free Hamiltonian
omega0 = 1.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0 = 0.5*omega0*sz
# derivative of the free Hamiltonian on omega0
dH = [0.5*sz]
# control Hamiltonians 
Hc = [sx,sy,sz]
# dissipation
sp = np.array([[0., 1.],[0., 0.]])  
sm = np.array([[0., 0.],[1., 0.]]) 
decay = [[sp, 0.0],[sm, 0.1]]
# measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
M = [M1, M2]
# time length for the evolution
tspan = np.linspace(0., 10.0, 2500)
# guessed control coefficients
cnum = len(tspan)-1
ctrl0 = [np.array([np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)])]
```
``` py
# Control algorithm: auto-GRAPE
GRAPE_paras = {"Adam":True, "ctrl0":ctrl0, "max_episode":300, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
control = ControlOpt(savefile=False, method="auto-GRAPE", **GRAPE_paras)
control.dynamics(tspan, rho0, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
# choose QFIM as the objective function
control.QFIM()
# choose CFIM as the objective function
control.CFIM(M=M)
```

**Example 2**  
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
\end{array}\right)\!\!, \nonumber
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

``` py
from quanestimation import *
import numpy as np

# initial state
rho0 = np.zeros((6,6), dtype=np.complex128)
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
cons = 100
D = (2*np.pi*2.87*1000)/cons
gS = (2*np.pi*28.03*1000)/cons
gI = (2*np.pi*4.32)/cons
A1 = (2*np.pi*3.65)/cons
A2 = (2*np.pi*3.03)/cons
H0 = D*np.kron(np.dot(s3, s3), ide2)+gS*(B1*S1+B2*S2+B3*S3)+gI*(B1*I1+B2*I2+B3*I3)+\
     + A1*(np.kron(s1, sx)+np.kron(s2, sy)) + A2*np.kron(s3, sz)
# derivatives of the free Hamiltonian on B1, B2 and B3
dH = [gS*S1+gI*I1, gS*S2+gI*I2, gS*S3+gI*I3]
# control Hamiltonians 
Hc = [S1, S2, S3]
# dissipation
decay = [[S3,2*np.pi/cons]]
# generation of a set of POVM basis
dim = len(rho0)
povm_basis = []
for i in range(dim):
    M_tp = np.dot(basis(dim, i), basis(dim, i).conj().T)
    povm_basis.append(M_tp)
# time length for the evolution
tspan = np.linspace(0.0, 2.0, 4000)
```
``` py
# Control algorithm: auto-GRAPE
GRAPE_paras = {"Adam":True, "ctrl0":ctrl0, "max_episode":300, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
control = ControlOpt(savefile=False, method="auto-GRAPE", **GRAPE_paras)
control.dynamics(tspan, rho0, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
# choose QFIM as the objective function
control.QFIM()
# choose CFIM as the objective function
control.CFIM(M=M)
```

---
## **Minimum parameterization time optimization**
Search of the minimum time to reach a given value of the objective function.
``` py
control.mintime(f, W=[], M=[], method="binary", target="QFIM", LDtype="SLD")
```
`f` is the given value of the objective function. In the package, two methods for searching 
the minimum time are provided which are logarithmic search and forward search from the 
beginning of time. It can be realized by setting `method=binary` (default) and 
`method=forward`. `target` represents the objective function for searching the minimum time, 
the users can choose QFI ($\mathrm{Tr}(WF^{-1})$), CFI ($\mathrm{Tr}(WI^{-1})$), and HCRB for 
the objective functions. If `target="QFIM"`, the types for the logarithmic derivatives can be 
set via `LDtype`.

---
## **Bibliography**
<a id="Khaneja2005">[1]</a>
N. Khaneja, T. Reiss, C. Hehlet, T. Schulte-Herbruggen, and S. J. Glaser,
Optimal control of coupled spin dynamics: Design of NMR pulse sequences by gradient 
ascent algorithms,
[J. Magn. Reson. **172**, 296 (2005).](https://doi.org/10.1016/j.jmr.2004.11.004)

<a id="Liu2017a">[2]</a>
J. Liu and H. Yuan,
Quantum parameter estimation with optimal control,
[Phys. Rev. A **96**, 012117 (2017).](https://doi.org/10.1103/PhysRevA.96.012117)

<a id="Liu2017b">[3]</a>
J. Liu and H. Yuan,
Control-enhanced multiparameter quantum estimation,
[Phys. Rev. A **96**, 042114 (2017).](https://doi.org/10.1103/PhysRevA.96.042114)

<a id="Baydin2018">[4]</a>
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)

<a id="Kennedy1995">[5]</a>
J. Kennedy and R. Eberhar,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

<a id="Storn1997">[6]</a>
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

<a id="Lillicrap2015">[7]</a>
T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, 
and D. Wierstra,
Continuous control with deep reinforcement learning,
[arXiv:1509.02971.](https://arxiv.org/abs/1509.02971)

