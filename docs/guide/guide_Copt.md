---
header-includes:
  - \usepackage{caption}
---

# **Control optimization**
The Hamiltonian of a controlled system can be written as
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters $\textbf{x}$ and $H_k$ 
represents the $k$th control Hamiltonian with $u_k$ the correspong control coefficient. In QuanEstimation, 
different algorithms are invoked to calculate the optimal control coefficients. The control optimization 
algorithms are include gradient ascent pulse engineering (GRAPE), auto-GRAPE, particle swarm optimization
(PSO), differential evolution (DE) and deep deterministic policy gradients (DDPG). 
```py
control = ControlOpt(savefile=False, method="auto-GRAPE", **kwargs)
control.dynamics(tspan, rho0, H0, dH, Hc, decay=[], ctrl_bound=[])
control.QFIM(W=[], LDtype="SLD")
control.CFIM(M=[], W=[])
control.HCRB(W=[])
```
Here `savefile` means whether to save all the control coefficients. If set `True` then the control 
coefficients and the values of the objective function obtained in all episodes will be saved during the 
training. If set `False` the control coefficients in the final episode and the values of the objective 
function in all episodes will be saved. `method` represents the control algorithm used to optimize the 
control coefficients, options are: "GRAPE", "auto-GRAPE", "PSO", "DE", and "DDPG". `**kwargs` is the keyword
and default value corresponding to the optimization algorithm which will be introduced in detail below.

`tspan` is the time length for the evolution, `rho0` represents the density matrix of the initial 
state, `H0` and `dH` are the free Hamiltonian and its derivatives on the unknown parameters to be estimated. 
`H0` is a matrix when the free Hamiltonian is time-independent and a list with the length equal to `tspan` 
when it is time-dependent. `dH` should be input as $[\partial_a{H_0}, \partial_b{H_0}, \cdots]$. `Hc` and
`ctrl` are two lists represent the control Hamiltonians and the corresponding control coefficients.`decay` 
contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, 
\gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...].
The default values for `decay`, `Hc` and `ctrl` are empty which means the dynamics is unitary 
and only governed by the free Hamiltonian. The package can be used to optimize bounded control problems by
setting lower and upper bounds of the control coefficients via `ctrl_bound`. It is an array with two 
elements representing the lower and upper bound of the control coefficients, respectively. The default value
of `ctrl_bound=[]` which means the control coefficients are in the regime $[-\infty,\infty]$.

There are two objective functions to choose in QuanEstimation which are QFI, CFI for single parameter 
estimation. If the users call `control.HCRB()` for single parameter estimation, the program will print
`"Program exit. In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the target 
function for control optimization"`. For multiparameter estimation, the package contains three objective 
functions which are $\mathrm{Tr}(W\mathcal{F}^{-1})$, $\mathrm{Tr}(W\mathcal{I}^{-1})$, and HCRB.
They can be choose via the codes `control.QFIM()`, `control.CFIM()`, and `control.HCRB()`. In the objective 
functions, `W` is the weight matrix which defaults to the identity matrix. `LDtype` in `control.QFIM()`
represents the types of the QFIM, it can be set as `LDtype=SLD` (default), `LDtype=RLD`
and `LDtype=LLD`. `M` represents a set of positive operator-valued measure (POVM) with default value `[]`. 
In the package, a set of rank-one symmetric informationally complete POVM (SIC-POVM) is loaded when `M=[]`.

## GRAPE and auto-GRAPE
The codes for control optimization with GRAPE and auto-GRAPE are as follows
``` py
control = ControlOpt(method="GRAPE", **kwargs)
```
``` py
control = ControlOpt(method="auto-GRAPE", **kwargs)
```
where `kwargs` can be input as
``` py
kwargs = {"Adam":True, "ctrl0":[], "max_episode":300, "epsilon":0.01, 
          "beta1":0.90, "beta2":0.99}
```
The keywords and the default values of GRAPE and auto-GRAPE can be seen in the following table

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "Adam"                           | True                       |
| "ctrl0"                          | []                         |
| "max_episode"                    | 300                        |
| "epsilon"                        | 0.01                       |
| "beta1"                          | 0.90                       |
| "beta2"                          | 0.99                       |

Adam algorithm can be introduced to update the control coefficients when using GRAPE and auto-GRAPE for 
control optimization, which can be realized by setting `Adam=True`. In this case, the Adam parameters 
include learning rate, the exponential decay rate for the first moment estimates and the second moment 
estimates can be set by the user via `epsilon`, `beta1` and `beta2`. If `Adam=False`, the control
coefficients will update according to the learning rate `"epsilon"`. `ctrl0` is a list representing 
the initial guesses of control coefficients and `max_episode` is the the number of episodes.

## PSO
The codes for control optimization with PSO are as follows
``` py
control = ControlOpt(method="PSO", **kwargs)
```
where `kwargs` can be input as
``` py
kwargs = {"particle_num":10, "ctrl0":[], "max_episode":[1000,100], 
          "c0":1.0, "c1":2.0, "c2":2.0, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "particle_num"                   | 10                         |
| "ctrl0"                          | []                         |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |
| "seed"                           | 1234                       |

Here `particle_num` is the the number of particles and `seed` ia the random seed. `c0`, `c1`, and `c2` 
are the PSO parameters representing the inertia weight, cognitive learning factor and social learning factor,
respectively. `max_episode` accepts both integers and arrays with two elements. If it is an integer, for 
example max_episode=1000, it means the program will continuously run 1000 episodes. However, if it is 
an array, for example max_episode=[1000,100], the program will run 1000 episodes in total but replace 
control coefficients of all the particles with global best every 100 episodes.

## DE
The codes for control optimization with DE are as follows
``` py
control = ControlOpt(method="DE", **kwargs)
```
where `kwargs` can be input as
``` py
kwargs = {"popsize":10, "ctrl0":[], "max_episode":1000, "c":1.0, 
          "cr":0.5, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "popsize"                        | 10                         |
| "ctrl0"                          | []                         |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |
| "seed"                           | 1234                       |

`popsize` represents the number of populations. `c` is the mutation constant and `cr` is the 
crossover constant.

## DDPG
The codes for control optimization with DDPG are as follows
``` py
control = ControlOpt(method="DDPG", **kwargs)
```
where `kwargs` can be input as
``` py
kwargs = {"layer_num":3, "layer_dim":200, "max_episode":1000, "seed":1234}
```

| $~~~~~~~~~~$**kwargs$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "ctrl0"                          | []                         |
| "max_episode"                    | 1000                       |
| "layer_num"                      | 3                          |
| "layer_dim"                      | 200                        |
| "seed"                           | 1234                       |

`layer_num` is the number of layers (include the input and output layer) and `layer_dim` is the number of neurons in the hidden layer.

---
