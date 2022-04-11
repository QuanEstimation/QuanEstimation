# **State optimization**
For state optimization in QuanEstimation, the probe state is expanded as 
$|\psi\rangle=\sum_i c_i|i\rangle$ in a specific basis $\{|i\rangle\}$. Search of the optimal 
probe states is equal to search of the normalized complex coefficients $\{c_i\}$. In 
QuanEstimation, the state optimization algorithms are automatic differentiation (AD) [[1]]
(#Baydin2018), particle swarm optimization (PSO) [[2]](#Kennedy1995), differential evolution 
(DE) [[3]](#Storn1997), Nelder-Mead (NM) [[4]](#Nelder1965), and deep deterministic policy 
gradients (DDPG) [[5]](#Lillicrap2015). The following codes can be used to perform state
optimizaiton

``` py
state = StateOpt(savefile=False, method="AD", **kwargs)
state.dynamics(tspan, H0, dH, Hc=[], ctrl=[], decay=[])
state.QFIM(W=[], LDtype="SLD")
state.CFIM(M=[], W=[])
state.HCRB(W=[])
```
`savefile` means whether to save all the states. If set `False` (default) the states in the 
final episode and the values of the objective function in all episodes will be saved. If set 
`True` then the states and the values of the objective function obtained in all episodes will 
be saved during the training. `method` represents the algorithm used to optimize the states, 
options are: "AD", "PSO", "DE", "DDPG", and "NM". `**kwargs` contains the keyword and 
default value corresponding to the optimization algorithm which will be introduced in 
detail below.

`tspan` is the time length for the evolution, `H0` and `dH` are the free Hamiltonian and its
derivatives on the unknown parameters to be estimated. `H0` accepts both matrix 
(time-independent evolution) and list (time-dependent evolution) with the length equal to 
`tspan`. `dH` should be input as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` 
and `ctrl` are two lists represent the control Hamiltonians and the corresponding control 
coefficients. `decay` contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the 
corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule 
decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. The default values for 
`decay`, `Hc` and `ctrl` are empty which means the dynamics is unitary and only governed by 
the free Hamiltonian. 

The code is `state.QFIM()` for the objective functions are QFI and $\mathrm{Tr}(W\mathcal{F}
^{-1})$, `state.CFIM()` for CFI and $\mathrm{Tr}(W\mathcal{I}^{-1})$ and `state.HCRB()` for 
HCRB. Here $F$ and $I$ are the QFIM and CFIM, $W$ corresponds to `W` is the weight matrix which 
defaults to the identity matrix. If the users call `state.HCRB()` for single parameter 
scenario, the program will exit and print `"Program exit. In single parameter scenario, HCRB is 
equivalent to QFI. Please choose QFIM as the target function"`. `LDtype` in `state.QFIM()` 
represents the types of the QFIM, it can be set as `LDtype=SLD` (default), `LDtype=RLD`, and 
`LDtype=LLD`. `M` represents a set of positive operator-valued measure (POVM) with default 
value `[]`. In the package, a set of rank-one symmetric informationally complete POVM 
(SIC-POVM) is used when `M=[]`.

If the parameterization is implemented with the Kraus operators, the code `state.dynamics()`
becomes
``` py
state.kraus(K, dK)
```
where `K` and `dK` are the Kraus operators and its derivatives on the unknown parameters.

---
## **AD**
The code for state optimization with AD is as follows
``` py
state = StateOpt(method="GRAPE", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"Adam":False, "psi0":[], "max_episode":300, "epsilon":0.01, 
          "beta1":0.90, "beta2":0.99}
```
The keywords and the default values of GRAPE and auto-GRAPE can be seen in the following 
table

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "Adam"                           | False                      |
| "psi0"                           | []                         |
| "max_episode"                    | 300                        |
| "epsilon"                        | 0.01                       |
| "beta1"                          | 0.90                       |
| "beta2"                          | 0.99                       |

In state optimization, he state will update according to the learning rate `"epsilon"`.
However, Adam algorithm can be introduced to update the states which can be realized by setting 
`Adam=True`. In this case, the Adam parameters include learning rate, the exponential decay 
rate for the first moment estimates and the second moment estimates can be set by the user via 
`epsilon`, `beta1` and `beta2`. `psi0` is a list representing the initial guesses of states and `max_episode` is the the number of episodes.

## **PSO**
The code for state optimization with PSO is as follows
``` py
state = StateOpt(method="PSO", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"particle_num":10, "psi0":[], "max_episode":[1000,100], 
          "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "particle_num"                   | 10                         |
| "psi0"                           | []                         |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |
| "seed"                           | 1234                       |

`particle_num` is the the number of particles. Here `max_episode` accepts both integers and 
arrays with two elements. If it is an integer, for example `max_episode=1000`, it means the 
program will continuously run 1000 episodes. However, if it is an array, for example 
`max_episode=[1000,100]`, the program will run 1000 episodes in total but replace states of 
all the particles with global best every 100 episodes. `c0`, `c1`, and `c2` are the PSO 
parameters representing the inertia weight, cognitive learning factor and social learning 
factor, respectively.  `seed` is the random seed.

## **DE**
The code for state optimization with DE is as follows
``` py
state = StateOpt(method="DE", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"popsize":10, "psi0":[], "max_episode":1000, "c":1.0, 
          "cr":0.5, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "popsize"                        | 10                         |
| "psi0"                           | []                         |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |
| "seed"                           | 1234                       |

`popsize` represents the number of populations. `c` and `cr` are the mutation constant and 
crossover constant.

## **NM**
The code for state optimization with NM is as follows
``` py
state = StateOpt(method="NM", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"state_num":10, "psi0":psi0, "max_episode":1000, "ar":1.0, 
          "ae":2.0, "ac":0.5, "as0":0.5, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "state_num"                      | 10                         |
| "psi0"                           | []                         |
| "max_episode"                    | 1000                       |
| "ar"                             | 1.0                        |
| "ae"                             | 2.0                        |
| "ac"                             | 0.5                        |
| "as0"                            | 0.5                        |
| "seed"                           | 1234                       |

`state_num` represents the number of initial states. `ar`, `ae`, `ac`, and `as0` are 
constants for reflection, expansion, constraction, and shrink, respectively.

## **DDPG**
The code for state optimization with DDPG is as follows
``` py
state = StateOpt(method="DDPG", **kwargs)
```
where `kwargs` is of the form
``` py
kwargs = {"layer_num":3, "layer_dim":200, "max_episode":1000, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "psi0"                           | []                         |
| "max_episode"                    | 1000                       |
| "layer_num"                      | 3                          |
| "layer_dim"                      | 200                        |
| "seed"                           | 1234                       |

`layer_num` and `layer_dim` represent the number of layers (include the input and output layer) 
and the number of neurons in the hidden layer.

**Example 1**  
The Hamiltonian of the Lipkin–Meshkov–Glick (LMG) model is
\begin{align}
H_{\mathrm{LMG}}=-\frac{\lambda}{N}(J_1^2+gJ_2^2)-hJ_3,
\end{align}

where $N$ is the number of spins of the system, $\lambda$ is the spin–spin interaction strength, $h$ is the strength of the 
external field and $g$ is the anisotropic parameter. $J_i=\frac{1}{2}\sum_{j=1}^N \sigma_i^{(j)}$ ($i=1,2,3$) is the collective spin operator with $\sigma_i^{(j)}$ the $i$th Pauli matrix for the $j$th spin. In single-parameter scenario, we take $g$ as the unknown parameter to be estimated. The states are expanded as 
$|\psi\rangle=\sum^J_{m=-J}c_m|J,m\rangle$ with $|J,m\rangle$ the Dicke state and $c_m$ a complex coefficient. Here we fixed 
$J=N/2$. In this example, the probe state is optimized for both noiseless scenario and collective dephasing noise. The dynamics under collective dephasing can be expressed as
<center> $\partial_t\rho = -i[H_{\mathrm{LMG}},\rho]+\gamma \left(J_3\rho J_3-\frac{1}{2}\left\{\rho, J^2_3\right\}\right)$ </center>
with $\gamma$ the decay rate.

In this case, all searches with different algorithms start from the coherent spin state defined by
$|\theta=\frac{\pi}{2},\phi=\frac{\pi}{2}\rangle=\exp(-\frac{\theta}{2}e^{-i\phi}J_{+}+\frac{\theta}{2}e^{i\phi}J_{-})|J,J\rangle$ with $J_{\pm}=J_1{\pm}iJ_2$.

``` py
from quanestimation import *
import numpy as np
from qutip import *

N = 8
# generation of coherent spin state
psi_css = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type="ket").full()
psi_css = psi_css.reshape(1, -1)[0]
# guessed state
psi0 = [psi_css]
# free Hamiltonian
Lambda = 1.0
g = 0.5
h = 0.1
Jx, Jy, Jz = jmat(0.5 * N)
Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
H0 = -Lambda*(np.dot(Jx, Jx) + g*np.dot(Jy, Jy))/N - h*Jz
# derivative of the free Hamiltonian on g
dH = [-Lambda*np.dot(Jy, Jy)/N]
# dissipation
decay = [[Jz, 0.1]]
# time length for the evolution
tspan = np.linspace(0.0, 10.0, 2500)
```
``` py
# State optimization algorithm: PSO
PSO_paras = {"particle_num":10, "psi0":psi0, "max_episode":[1000, 100], "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
state = StateOpt(savefile=False, method="PSO", **PSO_paras)
state.dynamics(tspan, H0, dH, decay=decay)
# choose QFIM as the objective function
state.QFIM()
```
 
**Example 2**  
In multi-parameter scenario, $g$ and $h$ are set to be the unknown parameters to be estimated.
``` py
from quanestimation import *
import numpy as np
from qutip import *

N = 8
# generation of coherent spin state
psi_css = spin_coherent(0.5*N, 0.5*np.pi, 0.5*np.pi, type="ket").full()
psi_css = psi_css.reshape(1, -1)[0]
# guessed state
psi0 = [psi_css]
# free Hamiltonian
Lambda = 1.0
g = 0.5
h = 0.1
Jx, Jy, Jz = jmat(0.5 * N)
Jx, Jy, Jz = Jx.full(), Jy.full(), Jz.full()
H0 = -Lambda*(np.dot(Jx, Jx) + g*np.dot(Jy, Jy))/N - h*Jz
# derivatives of the free Hamiltonian on the g and h
dH = [-Lambda * np.dot(Jy, Jy)/N, -Jz]
# dissipation
decay = [[Jz, 0.1]]
# time length for the evolution
tspan = np.linspace(0.0, 10.0, 2500)
# weight matrix
W = np.array([[1/3, 0.0], [0.0, 2/3]])
```
``` py
# State optimization algorithm: PSO
PSO_paras = {"particle_num":10, "psi0":psi0, "max_episode":[1000, 100], "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
state = StateOpt(savefile=False, method="PSO", **PSO_paras)
state.dynamics(tspan, H0, dH, decay=decay)
# choose QFIM as the objective function
state.QFIM()
```

---
## **Bibliography**
<a id="Baydin2018">[1]</a>
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)

<a id="Kennedy1995">[2]</a>
J. Kennedy and R. Eberhar,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

<a id="Storn1997">[3]</a>
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

<a id="Nelder1965">[4]</a>
J. A. Nelder and R. Mead,
A Simplex Method for Function Minimization,
[Comput. J. **7**, 308–313 (1965).](https://doi.org/10.1093/comjnl/7.4.308)

<a id="Lillicrap2015">[5]</a>
T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, 
and D. Wierstra,
Continuous control with deep reinforcement learning,
[arXiv:1509.02971.](https://arxiv.org/abs/1509.02971)

