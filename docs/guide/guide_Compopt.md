# **Comprehensive optimization**
In order to obtain the optimal parameter estimation schemes, it is necessary to
simultaneously optimize the probe state, control and measurement. The comprehensive 
optimization for the probe state and measurement (SM), the probe state and control (SC), the 
control and measurement (CM) and the probe state, control and measurement (SCM) are proposed
in QuanEstiamtion. In the package, the comprehensive optimization algorithms are particle 
swarm optimization (PSO) [[1]](#Kennedy1995), differential evolution (DE) [[2]](#Storn1997), 
and automatic differentiation (AD) [[3]](#Baydin2018).

``` py
com = ComprehensiveOpt(savefile=False, method="DE", **kwargs)
com.dynamics(tspan, H0, dH, Hc=[], ctrl=[], decay=[], ctrl_bound=[])
com.SM(W=[])
com.SC(W=[], M=[], target="QFIM", LDtype="SLD")
com.CM(rho0, W=[])
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
`tspan`, `H`, and `dH` shoule be input. `tspan` is the time length for the evolution, `H0` and `dH` are the free Hamiltonian and its derivatives on the unknown 
parameters to be estimated. `H0` is a matrix when the free Hamiltonian is time-independent
and a list with the length equal to `tspan` when it is time-dependent. `dH` should be input
as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` and `ctrl` are two lists represent the
control Hamiltonians and the corresponding control coefficients.`decay` contains decay 
operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, 
\gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, 
$\gamma_2$],...]. The default values for `decay`, `Hc` and `ctrl` are empty which means the 
dynamics is unitary and only governed by the free Hamiltonian. `ctrl_bound`is an array with two 
elements representing the lower and upper bound of the control coefficients, respectively. The 
default value of `ctrl_bound=[]` which means the control coefficients are in the regime 
$[-\infty,\infty]$.

QuanEstimation contains four comprehensive optimizations which are `com.SM()`, `com.SC()`,
`com.CM()` and `com.SCM()`. The code in `com.SC()` can be set as `target="QFIM"` (default), 
`target="CFIM` and `target="HCRB` for the objective functions are QFI ($\mathrm{Tr}(W\mathcal{F}
^{-1})$), CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$) and HCRB, respectively. Here $F$ and $I$ are 
the QFIM and CFIM, $W$ corresponds to `W` is the weight matrix which defaults to the identity 
matrix. If the users set `target="HCRB` for single parameter scenario, the program will exit 
and print `"Program exit. In single parameter scenario, HCRB is equivalent to QFI. Please 
choose QFIM as the target function"`. `LDtype` represents the types of the QFIM, it can be 
set as `LDtype=SLD` (default), `LDtype=RLD` and `LDtype=LLD`.

For optimization of probe state and measurement, the parameterization can also be implemented 
with the Kraus operators which can be realized by
``` py
com = ComprehensiveOpt(savefile=False, method="DE", **kwargs)
com.kraus(K, dK)
com.SM(W=[])  
```
where `K` and `dK` are the Kraus operators and its derivatives on the unknown parameters.



---
## **PSO**
The code for comprehensive optimization with PSO is as follows
``` py
com = ComprehensiveOpt(method="PSO", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"particle_num":10, "psi0":[], "ctrl0":[], "measurement0":[], 
          "max_episode":[1000,100], "c0":1.0, "c1":2.0, "c2":2.0, 
          "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "particle_num"                   | 10                         |
| "psi0"                           | [ ]                        |
| "ctrl0"                          | [ ]                        |
| "measurement0"                   | [ ]                        |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |
| "seed"                           | 1234                       |

`psi0`, `ctrl0` and `measurement0` in the algorithms represent the initial guesses of states, 
control coefficients and measurements, respectively, `seed` is the random seed. Here 
`particle_num` is the the number of particles, `c0`, `c1`, and `c2` are the PSO parameters 
representing the inertia weight, cognitive learning factor and social learning factor, 
respectively. `max_episode` accepts both integers and arrays with two elements. If it is an 
integer, for example max_episode=1000, it means the program will continuously run 1000 
episodes. However, if it is an array, for example max_episode=[1000,100], the program will run 
1000 episodes in total but replace control coefficients of all the particles with global best 
every 100 episodes. 

## **DE**
The code for comprehensive optimization with DE is as follows
``` py
com = ComprehensiveOpt(method="DE", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"popsize":10, "psi0":[], "ctrl0":[], "measurement0":[], 
          "max_episode":1000, "c":1.0, "cr":0.5, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "popsize"                        | 10                         |
| "psi0"                           | [ ]                        |
| "ctrl0"                          | [ ]                        |
| "measurement0"                   | [ ]                        |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |
| "seed"                           | 1234                       |

Here `max_episode` is the number of episodes, it is an integer. `popsize` represents the number 
of populations. `c` and `cr` are constants for mutation and crossover. 

## **AD**
The code for comprehensive optimization with AD is as follows
``` py
com = ComprehensiveOpt(method="AD", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"Adam":True, "psi0":[], "ctrl0":[], "measurement0":[],
         "max_episode":300, "epsilon":0.01, "beta1":0.90, "beta2":0.99}
```
The keywords and the default values of GRAPE and auto-GRAPE can be seen in the following 
table

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

The optimized variables will update according to the learning rate `"epsilon"` when 
`Adam=False`. However, If `Adam=True` Adam algorithm will be used and the Adam parameters 
include learning rate, the exponential decay rate for the first moment estimates and the second 
moment estimates can be set by the user via `epsilon`, `beta1` and `beta2`.

**Example 1**  
A single qubit system whose free evolution Hamiltonian is $H_0 = \frac{1}{2}\omega_0 \sigma_3$ with 
$\omega_0$ the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by
\begin{align}
\partial_t\rho=-i[H_0, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}\sigma_{-},\rho\}\right),
\end{align}

where $\gamma_{+}$, $\gamma_{-}$ are decay rates and $\sigma_{\pm}=(\sigma_1 \pm \sigma_2)/2$. The control Hamiltonian
\begin{align}
H_\mathrm{c}=u_1(t)\sigma_1+u_2(t)\sigma_2+u_3(t)\sigma_3
\end{align}

with $u_i(t)$ $(i=1,2,3)$ the control field. Here $\sigma_{1}$, $\sigma_{2}$ are also Pauli matrices.

In this case, we consider two types of comprehensive optimization, the first one is optimization of probe state and control (SC), and the other is optimization of probe state, control and measurement (SCM). QFI is taken as the target function for SC and CFI for SCM.

``` py
from quanestimation import *
import numpy as np

# free Hamiltonian
omega0 = 1.0
sx = np.array([[0., 1.],[1., 0.0j]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.0j],[0., -1.]])
H0 = 0.5*omega0*sz
# derivative of the free Hamiltonian on omega0
dH = [0.5*sz]
# control Hamiltonians 
Hc = [sx,sy,sz]
# dissipation
sp = np.array([[0., 1.],[0., 0.0j]])  
sm = np.array([[0., 0.0j],[1., 0.]]) 
decay = [[sp, 0.0],[sm, 0.1]]
# measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
M = [M1, M2]
# time length for the evolution
tspan = np.linspace(0., 20.0, 5000)
# guessed control coefficients
cnum = len(tspan)-1
ctrl0 = [np.array([np.zeros(cnum), np.zeros(cnum), np.zeros(cnum)])]
```
``` py
# Comprehensive optimization algorithm: DE
DE_paras = {"popsize":10, "psi0":[], "ctrl0":ctrl0, "measurement0":[], "max_episode":100, "c":1.0, "cr":0.5, "seed":1234}
com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
# comprehensive optimization for state and control (SC)
com.SC(W=[], target="QFIM", LDtype="SLD")
```
**Example 2**
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
where $g_\mathrm{e}$ ($g_\mathrm{n}$) is the $g$ factor of the electron (nuclear), $\mu_\mathrm{B}$ ($\mu_\mathrm{n}$) is the Bohr (nuclear) magneton and $\hbar$ is the Plank's constant. $\vec{B}$ is the magnetic field which be estimated. The control Hamiltonian is
\begin{align}
H_{\mathrm{c}}/\hbar=\sum^3_{i=1}\Omega_i(t)S_i
\end{align}

with $\Omega_i(t)$ the time-dependent Rabi frequency.

In this case, the initial state is taken as $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)\otimes|\!\!\uparrow\rangle$, 
where $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)$ is an electron state with $|1\rangle$ $(|\!-\!1\rangle)$ the 
eigenstate of $s_3$ with respect to the eigenvalue $1$ ($-1$). $|\!\!\uparrow\rangle$ is a nuclear state and 
the eigenstate of $\sigma_3$ with respect to the eigenvalue 1. $W$ is set to be $I$.

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
# measurement
M = []
for i in range(len(rho0)):
    M_tp = np.dot(basis(len(rho0), i), basis(len(rho0), i).conj().T)
    M.append(M_tp)
# time length for the evolution
tspan = np.linspace(0.0, 2.0, 4000)
# guessed control coefficients
cnum = 10
np.random.seed(1234)
ini_1 = np.zeros((len(Hc), cnum))
ini_2 = 0.2*np.ones((len(Hc), cnum))
ini_3 = -0.2*np.ones((len(Hc), cnum))
ini_4 = np.array([np.linspace(-0.2,0.2,cnum) for i in range(len(Hc))])
ini_5 = np.array([np.linspace(-0.2,0.0,cnum) for i in range(len(Hc))])
ini_6 = np.array([np.linspace(0,0.2,cnum) for i in range(len(Hc))])
ini_7 = -0.2*np.ones((len(Hc), cnum))+0.01*np.random.random((len(Hc), cnum))
ini_8 = -0.2*np.ones((len(Hc), cnum))+0.01*np.random.random((len(Hc), cnum))
ini_9 = -0.2*np.ones((len(Hc), cnum))+0.05*np.random.random((len(Hc), cnum))
ini_10 = -0.2*np.ones((len(Hc), cnum))+0.05*np.random.random((len(Hc), cnum))
ctrl0 = [ini_1, ini_2, ini_3, ini_4, ini_5, ini_6, ini_7, ini_8, ini_9, ini_10]
```
``` py
# Comprehensive optimization algorithm: DE
DE_paras = {"popsize":10, "psi0":[], "ctrl0":ctrl0, "measurement0":[], "max_episode":100, "c":1.0, "cr":0.5, "seed":1234}
com = ComprehensiveOpt(savefile=False, method="DE", **DE_paras)
com.dynamics(tspan, H0, dH, Hc, decay=decay, ctrl_bound=[-2.0, 2.0])
# comprehensive optimization for state and control (SC)
com.SC(W=[], target="QFIM", LDtype="SLD")
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