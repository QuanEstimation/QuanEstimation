# **Quantum metrological tools**
QuanEstimation can be used to calculate several well-used metrological tools including Quantum Cramér-Rao 
bounds, Holevo Cramér-Rao bound, Bayesian Cramér-Rao bounds, Quantum Ziv-Zakai bound and perform Bayesian 
estimation.

## **Quantum Cramér-Rao bounds**
In quantum metrology, quantum Cramér-Rao bounds are well used metrological tools for parameter 
estimation. It can be expressed as
\begin{align}
\mathrm{cov}\left(\hat{\textbf{x}}, \{\Pi_y\}\right) \geq \frac{1}{n}\mathcal{I}^{-1}\left(\{\Pi_y\}
\right) \geq \frac{1}{n} \mathcal{F}^{-1},
\end{align}

where $\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\sum_y\mathrm{Tr}(\rho\Pi_y)(\hat{\textbf{x}}
-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}$ is the covariance matrix for the unknown parameters 
$\hat{\textbf{x}}=(\hat{x}_0,\hat{x}_1,\dots)^{\mathrm{T}}$. $\{\Pi_y\}$ is a set of positive 
operator-valued measure (POVM) and $\rho$ represents the parameterized density matrix. $n$ is the
repetition of the experiment, $\mathcal{I}$ and $\mathcal{F}$ are the classical Fisher information 
matrix (CFIM) and the quantum Fisher information matrix (QFIM), respectively. The $ab$th entry of CFIM 
is defined by
\begin{align}
\mathcal{I}_{ab}=\sum_y\frac{1}{p(y|\textbf{x})}[\partial_a p(y|\textbf{x})][\partial_b 
p(y|\textbf{x})]
\end{align}

with $\{p(y|\textbf{x})=\mathrm{Tr}(\rho\Pi_y)\}$. The most well-used type of the QFIM is SLD-based QFIM
of the form
\begin{align}
\mathcal{F}_{ab}=\frac{1}{2}\mathrm{Tr}(\rho\{L_a, L_b\})
\end{align}

with $\mathcal{F}_{ab}$ the $ab$th entry of $\mathcal{F}$ and $L_{a}(L_{b})$ the symmetric logarithmic
derivative (SLD) operator for $x_{a}(x_b)$. The SLD operator is determined by
\begin{align}
\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)
\end{align}

and the calculation of SLD is via the equation
\begin{align}
\langle\lambda_i|L_{a}|\lambda_j\rangle=\frac{2\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle}
{\lambda_i+\lambda_j}
\end{align}

for $\lambda_i (\lambda_j)\neq 0$ and for $\lambda_i=\lambda_j=0$, it is set to be zero.

Besides, there are right logarithmic derivative (RLD) and left logarithmic derivative (LLD) defined by
$\partial_{a}\rho=\rho \mathcal{R}_a$ and $\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho$ with the  corresponding QFIM  $\mathcal{F}_{ab}=\mathrm{Tr}(\rho \mathcal{R}_a \mathcal{R}^{\dagger}_b)$. The RLD
and LLD operators are calculated as

\begin{align}
\langle\lambda_i| \mathcal{R}_{a} |\lambda_j\rangle
&= \frac{1}{\lambda_i}\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle,~~\lambda_i\neq 0; \\
\langle\lambda_i| \mathcal{R}_{a}^{\dagger} |\lambda_j\rangle
&= \frac{1}{\lambda_j}\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle,~~\lambda_j\neq 0.
\end{align}

In QuanEstimation, three types of the logarithmic derivatives can be solved by calling the codes
``` py
SLD(rho, drho, rep="original", eps=1e-8)
```
``` py
RLD(rho, drho, rep="original", eps=1e-8)
```
``` py
LLD(rho, drho, rep="original", eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `rho` and `drho` are the density matrix of the state and its derivatives on the unknown parameters
to be estimated. `drho` should be input as $[\partial_a{\rho}, \partial_b{\rho}, \cdots]$.
For single parameter estimation (the length of `drho` is equal to one), the output is a matrix and for 
multiparameter estimation (the length of `drho` is more than one), it returns a list. There are two 
output choices for the logarithmic derivatives basis which can be setting through `rep`. The default
basis (`rep="original"`) of the logarithmic derivatives is the same with `rho`, another choice is the 
`rep="eigen"` which means the logarithmic derivatives are written in the eigenspace of `rho`. `eps` 
represents the machine epsilon which defaults to $10^{-8}$.

In QuanEstimation, the QFI and QFIM can be calculated via
``` py
QFIM(rho, drho, LDtype="SLD", exportLD=False, eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
`LDtype` represents the types of QFI (QFIM) can be set. Options are `LDtype=SLD` (default), `LDtype=RLD`
and `LDtype=LLD`. This function will return QFI (QFIM) if `exportLD=False` and if the users set 
`exportLD=True`, it will return logarithmic derivatives other than QFI (QFIM).

If the parameterization of a state is via the Kraus operators, the QFI (QFIM) can be calculated by calling 
the function
``` py
QFIM_Kraus(rho0, K, dK, LDtype="SLD", exportLD=False, eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `K` and `dK` are the Kraus operators and the derivatives on the unknown parameters to be estimated.

The FI (FIM) for a set of probabilities `p` can be calculated as
``` py
FIM(p, dp, eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `dp` is the derivatives of probabilities `p` on the unknown parameters, it is a list.

In quantum metrology, the CFI (CFIM) are solved by
``` py
CFIM(rho, drho, M=[], eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
Here `M` represents a set of positive operator-valued measure (POVM) with default value `[]`. In this 
function, a set of rank-one symmetric informationally complete POVM (SIC-POVM) is load when `M=[]`. 
SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded 
from [http://www.physics.umb.edu/Research/QBism/solutions.html](http://www.physics.umb.edu/Research/QBism/solutions.html).

In Bloch representation, the SLD based QFI (QFIM) is calculated by
``` py
QFIM_Bloch(r, dr, eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
`r` and `dr` are the parameterized Bloch vector and its derivatives of on the unknown parameters to 
be estimated.

In QuanEstimation, it can also calculte the SLD based QFI (QFIM) with Gaussian states. 
$\textbf{R}=(q_1,p_1,q_2,p_2,\dots)^{\mathrm{T}}$ with $q_i=(a_i+a^{\dagger}_i)/\sqrt{2}$ and 
$p_i=(a_i-a^{\dagger}_i)/(i\sqrt{2})$ represents a vector of quadrature operators.
``` py
QFIM_Gauss(R, dR, D, dD)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
is used to calculate the SLD based QFI (QFIM) with Gaussian states. The variable `R` is expected value
($\mathrm{Tr}(\rho\textbf{R})$) of $\textbf{R}$, it is an array epresenting the first-order moment.
`dR` is a list of derivatives of `R` on the unknown parameters with $i$th entry $\partial_{\textbf{x}}
\langle[\textbf{R}]_i\rangle$. `D` and `dD` represent the second-order moment matrix with the $ij$th entry
$D_{ij}=\langle [\textbf{R}]_i [\textbf{R}]_j\rangle$ and its derivatives of on the unknown parameters. 
`dD` is a list.

**Example**  
The Hamiltonian of a single qubit system is $H = \frac{1}{2}\omega_0 \sigma_3$ with 
$\omega_0$ the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by
\begin{align}
\partial_t\rho=-i[H, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}\sigma_{+},
\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}\sigma_{-},\rho\}\right),
\end{align}

where $\sigma_{\pm}=\frac{1}{2}(\sigma_1 \pm \sigma_2)$ with $\sigma_{1}$, $\sigma_{2}$ Pauli matrices 
and $\gamma_{+}$, $\gamma_{-}$ are decay rates. The probe state is taken as $|+\rangle$ and the 
measurement for CFI is $\{|+\rangle\langle+|, |-\rangle\langle-|\}$ with
$|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$. Here $|0\rangle$ and $|1\rangle$ are the
eigenstates of $\sigma_3$ with respect to the eigenvalues $1$ and $-1$.
``` py
from quanestimation import *
import numpy as np

# initial state
rho0 = 0.5*np.array([[1.,1.],[1.,1.]])
# free Hamiltonian
omega0 = 1.0
sz = np.array([[1., 0.],[0., -1.]])
H0 = 0.5*omega0*sz
# derivative of the free Hamiltonian on omega0
dH = [0.5*sz]
# dissipation
sp = np.array([[0., 1.],[0., 0.]])
sm = np.array([[0., 0.],[1., 0.]])
decay = [[sp, 0.0],[sm, 0.1]]
# measurement
M1 = 0.5*np.array([[1.,1.],[1.,1.]])
M2 = 0.5*np.array([[1.,-1.],[-1.,1.]])
M = [M1, M2]
# time length for the evolution
tspan = np.linspace(0.0, 50.0, 2000)
# dynamics
dynamics = Lindblad(tspan, rho0, H0, dH, decay)
rho, drho = dynamics.expm()
# calculation of CFI and QFI
I, F = [], []
for ti in range(1,2000):
    # CFI
    I_tp = CFIM(rho[ti], drho[ti], M=M)
    I.append(I_tp)
    # QFI
    F_tp = QFIM(rho[ti], drho[ti])
    F.append(F_tp)
```
<span style="color:red">(julia example) </span>
``` jl
julia example 
```
---

## **Holevo Cramér-Rao bound**
Holevo Cramér-Rao bound (HCRB) is of the form
\begin{align}
\mathrm{Tr}(W\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\}))\geq \min_{\textbf{X},V} \mathrm{Tr}(WV)
\end{align}
where $W$ is the weight matrix and $V\geq Z(\textbf{X})$ with $[Z(\textbf{X})]_{ab}=\mathrm{Tr}
(\rho X_a X_b)$. $\textbf{X}=[X_0,X_1,\cdots]$ with $X_i:=\sum_y (\hat{x}_i(y)-x_i)\Pi_y$. The HCRB can
be calculated via semidefinite programming as

\begin{align}
& \min_{\textbf{X},V}~\mathrm{Tr}(WV),  \nonumber \\
& \mathrm{subject}~\mathrm{to}
\begin{cases}
\left(\begin{array}{cc}
V & \Lambda^{\mathrm{T}}R^{\dagger} \\
R\Lambda & I\\
\end{array}\right)\geq 0, \\
\sum_i[\Lambda]_{ai}\mathrm{Tr}(\lambda_i\partial_b\rho)=\delta_{ab}.
\end{cases}
\end{align}

$X_i$ is expanded in a specific basis $\{\lambda_i\}$ as $X_i=\sum_j [\Lambda]_{ij}\lambda_j$, 
the Hermitian matrix $Z(\textbf{X})$ satisfies $Z(\textbf{X})=\Lambda^{\mathrm{T}}R^{\dagger}R\Lambda$.
In QuanEstimation, the HCRB can be solved by
``` py
HCRB(rho, drho, W, eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `rho` and `drho` are the density matrix of the state and its derivatives on the unknown parameters
to be estimated, respectively. `W` represents the weight matrix and `eps` is the machine epsilon with
default value $10^{-8}$.

**Example**  
The Hamiltonian of a two-qubit system with $XX$ coupling is 
\begin{align}
H=\omega_1\sigma_3^{(1)}+\omega_2\sigma_3^{(2)}+g\sigma_1^{(1)}\sigma_1^{(2)},
\end{align}

where $\omega_1$, $\omega_2$ are the frequencies of the first and second qubit, $\sigma_i^{(1)}=
\sigma_i\otimes I$ and $\sigma_i^{(2)}=I\otimes\sigma_i$ for $i=1,2,3$. $\sigma_1$, $\sigma_2$, $\sigma_3$ 
are Pauli matrices and $I$ denotes the identity matrix. The dynamics is described by the master equation 
\begin{align}
\partial_t\rho=-i[H, \rho]+\sum_{i=1,2}\gamma_i\left(\sigma_3^{(i)}\rho\sigma_3^{(i)}-\rho\right)
\end{align}

with $\gamma_i$ the decay rate for the $i$th qubit.

The probe state is taken as $\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$ and the weight matrix is set to be
identity. The measurement for $\mathrm{Tr}(W\mathcal{I^{-1}})$ is $\{\Pi_1$, $\Pi_2$, $I-\Pi_1-\Pi_2\}$ 
with $\Pi_1=0.85|00\rangle\langle 00|$ and $\Pi_2=0.1|\!+\!+\rangle\langle+\!+\!|$. Here 
$|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of 
$\sigma_3$ with respect to the eigenvalue $1$ ($-1$).

``` py 
from quanestimation import *
import numpy as np

# initial state
psi0 = np.array([1.0, 0.0, 0.0, 1.0])/np.sqrt(2)
rho0 = np.dot(psi0.reshape(-1,1), psi0.reshape(1,-1).conj())
# free Hamiltonian
omega1, omega2, g = 1.0, 1.0, 0.1
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
ide = np.array([[1.,0.],[0.,1.]])   
H0 = omega1*np.kron(sz, ide)+omega2*np.kron(ide, sz)+g*np.kron(sx, sx)
# derivatives of the free Hamiltonian on omega2 and g
dH = [np.kron(ide, sz), np.kron(sx, sx)] 
# dissipation
decay = [[np.kron(sz,ide), 0.05], [np.kron(ide,sz), 0.05]]
# measurement
m1 = np.array([0., 0., 0., 1.])
M1 = 0.85*np.dot(m1.reshape(-1,1), m1.reshape(1,-1).conj())
M2 = 0.1*np.array([[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.]])
M = [M1, M2, np.identity(4)-M1-M2]
# time length for the evolution
tspan = np.linspace(0.0, 10.0, 1000)
# dynamics
dynamics = Lindblad(tspan, rho0, H0, dH, decay)
rho, drho = dynamics.expm()
# weight matrix
W = [[1.0, 0.0],[0.0, 1.0]]
# calculation of CFIM, QFIM and HCRB
F, I, f = [], [], []
for ti in range(1, 1000):
    # CFIM
    I_tp = CFIM(rho[ti], drho[ti], M=M)
    I.append(I_tp)
    # QFIM
    F_tp = QFIM(rho[ti], drho[ti])
    F.append(F_tp)
    # HCRB
    f_tp = HCRB(rho[ti], drho[ti], W, eps=1e-6)
    f.append(f_tp)
```
<span style="color:red">(julia example) </span>
``` jl
julia example
```

---

## **Bayesian Cramér-Rao bounds**
The Bayesion version of the CFI (CFIM) and QFI (QFIM) can be calculated by <br>
<center> $\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}$ </center> <br>
and <br>
<center> $\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x},$</center> <br>
where $p(\textbf{x})$ is the prior distribution, $\mathcal{I}$ and $\mathcal{F}$ are CFI (CFIM) and QFI
(QFIM) of all types, respectively.

In QuanEstimation, BCFI (BCFIM) and BQFI (BQFIM) can be solved via
``` py
BCFIM(x, p, rho, drho, M=[], eps=1e-8)
```
``` py
BQFIM(x, p, rho, drho, LDtype="SLD", eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `x` represents the regimes of the parameters for the integral, it should be input as a list of arrays. 
`p` is an array representing the prior distribution. The input varibles `rho` and `drho` are two 
multidimensional lists with the dimensions as `x`. For example, for three parameter ($x_0, x_1, x_2$) 
estimation, the $ijk$th entry of `rho` and `drho` are $\rho$ and $[\partial_0\rho, \partial_1\rho, 
\partial_2\rho]$ with respect to the values $[x_0]_i$, $[x_1]_j$ and $[x_2]_k$, respectively.`LDtype` 
represents the types of QFI (QFIM) can be set. Options are `LDtype=SLD` (default), `LDtype=RLD` and
`LDtype=LLD`. `M` represents a set of positive operator-valued measure (POVM) with default value `[]`. 
In QuanEstimation, a set of rank-one symmetric informationally complete POVM (SIC-POVM) is load when `M=[]`. 
SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded 
from [http://www.physics.umb.edu/Research/QBism/solutions.html](http://www.physics.umb.edu/Research/QBism/solutions.html).

In the Bayesian scenarios, the covariance matrix with a prior distribution $p(\textbf{x})$ is defined as
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}(\rho\Pi_y)
(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}\mathrm{d}\textbf{x}
\end{align}

where $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$ are the unknown parameters to be estimated and the integral 
$\int\mathrm{d}\textbf{x}:=\iiint\mathrm{d}x_0\mathrm{d}x_1\cdots$. $\{\Pi_y\}$ is a set of POVM and $\rho$ 
represents the parameterized density matrix. The two types of Bayesian Cramér-Rao bound (BCRB) are calculated
in this package, the first one is 
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})\left(B\mathcal{I}^{-1}B
+\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x},
\end{align}

where $\textbf{b}$ and $\textbf{b}'$ are the vectors of biase and its derivatives on parameters. $B$ is a 
diagonal matrix with the $i$th entry $B_{ii}=1+[\textbf{b}']_{i}$ and $\mathcal{I}$ is the CFIM. The second 
one is
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{I}_{\mathrm{Bayes}}^{-1}\,
\mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x},
\end{align}

where $\mathcal{B}=\int p(\textbf{x})B\mathrm{d}\textbf{x}$ is the average of $B$ and 
$\mathcal{I}_{\mathrm{Bayes}}$ is the average of the CFIM.

Two types of Bayesian Quantum Cramér-Rao bound (BCRB) are calculated, the first one is 
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq\int p(\textbf{x})\left(B\mathcal{F}^{-1}B
+\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x},
\end{align}
        
where $\textbf{b}$ and $\textbf{b}'$ are the vectors of biases and its derivatives on $\textbf{x}$. $B$ is 
a diagonal matrix with the $i$th entry $B_{ii}=1+[\textbf{b}']_{i}$ and $\mathcal{F}$ is the QFIM for all 
types. The second one is
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{F}_{\mathrm{Bayes}}^{-1}\,
\mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x},
\end{align}

where $\mathcal{B}=\int p(\textbf{x})B\mathrm{d}\textbf{x}$ is the average of $B$ and 
$\mathcal{F}_{\mathrm{Bayes}}$ is the average of the QFIM.

In QuanEstimation, the BCRB and BQCRB are calculated via
``` py
BCRB(x, p, rho, drho, M=[], b=[], db=[], btype=1, eps=1e-8)
```
``` py
BQCRB(x, p, rho, drho, b=[], db=[], btype=1, LDtype="SLD", eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `b` and `db` are the vectors of biases and its derivatives on the unknown parameters. For unbiased 
estimates, `b` and `db` are set to `[]` which are the default values in the package. In QuanEstimation,
two types of BCRB and BQCRB are calculated, the user can choose via the variable `btype`. For single 
parameter estimation, <span style="color:red">Ref </span> calculate the optimal biased bound based on the 
first type of the BQCRB, it can be realized numerically
``` py
OBB(x, p, dp, rho, drho, d2rho, LDtype="SLD", eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
`d2rho` is a list representing the second order derivatives of `rho` on `x`.

Van Trees in 1968 <span style="color:red">Ref </span> provide a well used Bayesian version of Cramér-Rao 
bound known as Van Trees bound (VTB) and the quantum version (QVTB) provided by Tsang, Wiseman and Caves.
Two types of VTB are contained in QuanEstimation, the first one is 
\begin{align}        
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})\left(\mathcal{I}_p
+\mathcal{I}\right)^{-1}\mathrm{d}\textbf{x},
\end{align}

where the entry of $\mathcal{I}_{p}$ is defined by$[\mathcal{I}_{p}]_{ab}=[\partial_a \ln p(\textbf{x})]
[\partial_b \ln p(\textbf{x})]$ and $\mathcal{I}$ represents the CFIM. The second one is      
<center> $\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \left(\mathcal{I}_{\mathrm{prior}}
+\mathcal{I}_{\mathrm{Bayes}}\right)^{-1},$ </center>  

where $\mathcal{I}_{\mathrm{prior}}=\int p(\textbf{x})\mathcal{I}_{p}\mathrm{d}\textbf{x}$ 
is the CFIM for $p(\textbf{x})$ and $\mathcal{I}_{\mathrm{Bayes}}$ is the average of the CFIM.

Besides, the package can also calculate two types of QVTB, the first one is  
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})\left(\mathcal{I}_p
+\mathcal{F}\right)^{-1}\mathrm{d}\textbf{x},
\end{align}

where the entry of $\mathcal{I}_{p}$ is defined by $[\mathcal{I}_{p}]_{ab}=[\partial_a \ln p(\textbf{x})]
[\partial_b \ln p(\textbf{x})]$ and $\mathcal{F}$ is the QFIM for all types. The second one is
<center> $\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \left(\mathcal{I}_{\mathrm{prior}}
+\mathcal{F}_{\mathrm{Bayes}}\right)^{-1},$ </center> 

where $\mathcal{I}_{\mathrm{prior}}=\int p(\textbf{x})\mathcal{I}_{p}\mathrm{d}\textbf{x}$ is 
the CFIM for $p(\textbf{x})$ and $\mathcal{F}_{\mathrm{Bayes}}$ is the average of the QFIM.

The functions to calculate the VTB and QVTB are
``` py
VTB(x, p, dp, rho, drho, M=[], btype=1, eps=1e-8)
```
``` py
QVTB(x, p, dp, rho, drho, btype=1, LDtype="SLD", eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```

## **Quantum Ziv-Zakai bound**
The expression of Quantum Ziv-Zakai bound (QZZB) with a prior distribution p(x) in a finite regime 
$[\alpha,\beta]$ is

\begin{eqnarray}
\mathrm{var}(\hat{x},\{\Pi_y\}) &\geq & \frac{1}{2}\int_0^\infty \mathrm{d}\tau\tau
\mathcal{V}\int_{-\infty}^{\infty} \mathrm{d}x\min\!\left\{p(x), p(x+\tau)\right\} \nonumber \\
& & \times\left(1-\frac{1}{2}||\rho(x)-\rho(x+\tau)||\right),
\end{eqnarray}

where $||\cdot||$ represents the trace norm and $\mathcal{V}$ is the "valley-filling" 
operator satisfying $\mathcal{V}f(\tau)=\max_{h\geq 0}f(\tau+h)$. $\rho(x)$ is the 
parameterized density matrix. 

In QuanEstimation, the QZZB can be calculated via the function:
``` py
QZZB(x, p, rho, eps=1e-8)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `x` is a list of array representing the regime of the parameter for the integral, `p` is an array 
representing the prior distribution and `rho` is a multidimensional list representing the density matrix.
`eps` is the machine epsilon with default value $10^{-8}$.

---

**Example**  
The Hamiltonian of a qubit system under a magnetic field $B$ in the XZ plane is
\begin{align}
H=\frac{B}{2}(\sigma_1\cos{x}+\sigma_3\sin{x})
\end{align}

with $x$ the unknown parameter and $\sigma_{1}$, $\sigma_{3}$ Pauli matrices. The probe state is taken as 
$\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$ with $|0\rangle$ and $|1\rangle$ the eigenvstates of $\sigma_3$ 
with respect to the eigenvalues $1$ and $-1$. The measurement for classical bounds is a set of rank-one 
symmetric informationally complete positive operator-valued measure (SIC-POVM).

Take the Gaussian prior distribution $p(x)=\frac{1}{c\eta\sqrt{2\pi}}\exp\left({-\frac{(x-\mu)^2}{2\eta^2}}
\right)$ on $[-\pi/2, \pi/2]$ with $\mu$ and $\eta$ the expectation and standard deviation, respectively. 
Here $c=\frac{1}{2}\big[\mathrm{erf}(\frac{\pi-2\mu}{2\sqrt{2}\eta})+\mathrm{erf}(\frac{\pi+2\mu}
{2\sqrt{2}\eta})\big]$ is the normalized coefficient with $\mathrm{erf}(x):=\frac{2}{\sqrt{\pi}}\int^x_0 
e^{-t^2}\mathrm{d}t$ the error function.
``` py
from quanestimation import *
import numpy as np
from scipy.integrate import simps

# initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
# free Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0_func = lambda x: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
# derivative of the free Hamiltonian on x
dH_func = lambda x: [0.5*B*(-sx*np.sin(x)+sz*np.cos(x))]
# prior distribution
x = np.linspace(-0.5*np.pi, 0.5*np.pi, 100)
mu, eta = 0.0, 0.2
p_func = lambda x, mu, eta: np.exp(-(x-mu)**2/(2*eta**2))\
                            /(eta*np.sqrt(2*np.pi))
dp_func = lambda x, mu, eta: -(x-mu)*np.exp(-(x-mu)**2/(2*eta**2))\
                             /(eta**3*np.sqrt(2*np.pi))
p_tp = [p_func(x[i], mu, eta) for i in range(len(x))]
dp_tp = [dp_func(x[i], mu, eta) for i in range(len(x))]
# normalization of the distribution
c = simps(p_tp, x)
p, dp = p_tp/c, dp_tp/c
# time length for the evolution
tspan = np.linspace(0., 1.0, 1000)
# dynamics
rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128)\
       for i in range(len(x))]
drho = [[np.zeros((len(rho0), len(rho0)), dtype=np.complex128)]\
         for i in range(len(x))]
for i in range(len(x)):
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    dynamics = Lindblad(tspan, rho0, H0_tp, dH_tp)
    rho_tp, drho_tp = dynamics.expm()
    rho[i] = rho_tp[-1]
    drho[i] = drho_tp[-1]

# Classical Bayesian bounds
f_BCRB1 = BCRB([x], p, rho, drho, M=[], btype=1)
f_BCRB2 = BCRB([x], p, rho, drho, M=[], btype=2)
f_VTB1 = VTB([x], p, dp, rho, drho, M=[], btype=1)
f_VTB2 = VTB([x], p, dp, rho, drho, M=[], btype=2)

# Quantum Bayesian bounds
f_BQCRB1 = BQCRB([x], p, rho, drho, btype=1)
f_BQCRB2 = BQCRB([x], p, rho, drho, btype=2)
f_QVTB1 = QVTB([x], p, dp, rho, drho, btype=1)
f_QVTB2 = QVTB([x], p, dp, rho, drho, btype=2)
f_QZZB = QZZB([x], p, rho)
```
<span style="color:red">(julia example) </span>
``` jl
julia example
```

---

## **Bayesian estimation**
In QuanEstimation, two types of Bayesian estimation are considered including maximum a posteriori 
estimation (MAP) and maximum likelihood estimation (MLE). In Bayesian estimation, the prior distribution 
is updated as
\begin{align}
p(\textbf{x}|y)=\frac{p(y|\textbf{x})p(\textbf{x})}{\int p(y|\textbf{x})p(\textbf{x})\mathrm{d}\textbf{x}}
\end{align}

with $p(\textbf{x})$ the current prior distribution and $y$ the outcome of the experiment. In practice, the 
prior distribution is replaced with $p(\textbf{x}|y)$ and the estimated value of $\textbf{x}$ can be 
evaluated by
``` py
Bayes(x, p, rho, y, M=[], savefile=False)
```
``` py
MLE(x, rho, y, M=[], savefile=False)
```
<span style="color:red">(julia code) </span>
``` jl
julia code 
```
where `x` is a list of arrays representing the regimes of the parameters for the integral and `p` is an array 
representing the prior distribution. For multiparameter estimation, `p` is multidimensional. The input varible 
`rho` and is a multidimensional list with the dimensions as `x` representing the parameterized density matrix. 
`M` contains a set of positive operator-valued measure (POVM). In QuanEstimation, a set of rank-one symmetric 
informationally complete POVM (SIC-POVM) is load when `M=[]`. SIC-POVM is calculated by the Weyl-Heisenberg 
covariant SIC-POVM fiducial state which can be downloaded from 
[http://www.physics.umb.edu/Research/QBism/solutions.html](http://www.physics.umb.edu/Research/QBism/solutions.html). 
`savefile` means whether to save all the posterior distributions (likelihood functions). If set `True` then 
two files "pout.npy" ("Lout.npy") and "xout.npy" will be generated including the likelihood functions and 
the estimated values in the iterations. If set `False` the likelihood function in the final iteration and 
the estimated values in all iterations will be saved in "pout.npy" ("Lout.npy") and "xout.npy". 

**Example**  
The Hamiltonian of a qubit system is 
\begin{align}
H=\frac{B}{2}(\sigma_1\cos{x}+\sigma_3\sin{x}),
\end{align}

where $B$ is the magnetic field in the XZ plane, $x$ is the unknown parameter and $\sigma_{1}$, $\sigma_{3}$ 
are the Pauli matrices. The probe state is taken as $|\pm\rangle$. The measurement is 
$\{|\!+\rangle\langle+\!|,|\!-\rangle\langle-\!|\}$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|
1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ 
$(-1)$. In this example, the prior distribution $p(x)$ is uniform on $[0, \pi/2]$.
``` py
from quanestimation import *
import numpy as np
import random

# initial state
rho0 = 0.5*np.array([[1., 1.],[1., 1.]])
# free Hamiltonian
B = np.pi/2.0
sx = np.array([[0., 1.],[1., 0.]])
sy = np.array([[0., -1.j],[1.j, 0.]]) 
sz = np.array([[1., 0.],[0., -1.]])
H0_func = lambda x: 0.5*B*(sx*np.cos(x)+sz*np.sin(x))
# derivative of the free Hamiltonian on x
dH_func = lambda x: [0.5*B*(-sx*np.sin(x)+sz*np.cos(x))]
# measurement
M1 = 0.5*np.array([[1., 1.],[1., 1.]])
M2 = 0.5*np.array([[1.,-1.],[-1., 1.]])
M = [M1, M2]
# prior distribution
x = np.linspace(0.0, 0.5*np.pi, 1000)
p = (1.0/(x[-1]-x[0]))*np.ones(len(x))
# time length for the evolution
tspan = np.linspace(0., 1.0, 1000)
# dynamics
rho = [np.zeros((len(rho0), len(rho0)), dtype=np.complex128) for i in range(len(x))]
for i in range(len(x)):
    H0 = H0_func(x[i])
    dH = dH_func(x[i])
    dynamics = Lindblad(tspan, rho0, H0, dH)
    rho_tp, drho_tp = dynamics.expm()
    rho[i] = rho_tp[-1]

# Generation of the experimental results
y = [0 for i in range(500)]
res_rand = random.sample(range(0,len(y)), 125)
for i in range(len(res_rand)):
    y[res_rand[i]] = 1

# Maximum a posteriori estimation
pout, xout = Bayes([x], p, rho, y, M=M, savefile=False)

# Maximum likelihood estimation
Lout, xout = MLE([x], rho, y, M=M, savefile=False)
```
<span style="color:red">(julia example) </span>
``` jl
julia example
```

---
