This section documents the Python classes in the Python-Julia package.

## **Dynamics**
::: quanestimation.Lindblad
::: quanestimation.Kraus
::: quanestimation.QubitDephasing

## **Control Waveforms**

Predefined control coefficient functions for constructing controlled Hamiltonians.
Each returns a callable object that evaluates the waveform at time `t`.

::: quanestimation.ZeroCTRL
::: quanestimation.LinearCTRL
::: quanestimation.SineCTRL
::: quanestimation.SawCTRL
::: quanestimation.TriangleCTRL
::: quanestimation.GaussianCTRL
::: quanestimation.GaussianEdgeCTRL

## **NV Magnetometer**

::: quanestimation.NVMagnetometerScheme

## **Control Optimization**
The Hamiltonian of a controlled system can be written as
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters
$\textbf{x}$ and $H_k$ represents the $k$th control Hamiltonian with $u_k$ the
corresponding control coefficient. In QuanEstimation, different algorithms are invoked to
update the optimal control coefficients. The control optimization algorithms are
gradient ascent pulse engineering (GRAPE), GRAPE algorithm based on the automatic
differentiation (auto-GRAPE), particle swarm optimization (PSO), and
differential evolution (DE).

### **Base**
::: quanestimation.ControlSystem

### **Control optimization with GRAPE and auto-GRAPE**
::: quanestimation.GRAPE_Copt

### **Control Optimization with PSO**
::: quanestimation.PSO_Copt

### **Control Optimization DE**
::: quanestimation.DE_Copt

---

## **State Optimization**
The probe state is expanded as $|\psi\rangle=\sum_i c_i|i\rangle$ in a specific
basis, i.e., $\{|i\rangle\}$. In state optimization, the search of the
optimal probe states is equal to search of the normalized complex coefficients
$\{c_i\}$. In QuanEstimation, the state optimization algorithms are
automatic differentiation (AD), random iteration (RI), particle swarm
optimization (PSO), differential evolution (DE), and Nelder-Mead (NM).

### **Base**
::: quanestimation.StateSystem

### **State optimization with AD**
::: quanestimation.AD_Sopt

### **State optimization with RI**
::: quanestimation.RI_Sopt

### **State Optimization with PSO**
::: quanestimation.PSO_Sopt

### **State Optimization DE**
::: quanestimation.DE_Sopt

---

## **Measurement Optimization**
In QuanEstimation, three measurement optimization scenarios are considered. The first one
is to optimize a set of rank-one projective measurement, it can be written in a specific
basis $\{|\phi_i\rangle\}$ with $|\phi_i\rangle=\sum_j C_{ij}|j\rangle$ in the Hilbert space
as $\{|\phi_i\rangle\langle\phi_i|\}$. In this case, the goal is to search a set of optimal
coefficients $C_{ij}$. The second scenario is to find the optimal linear combination of
an input measurement $\{\Pi_j\}$. The third scenario is to find the optimal rotated
measurement of an input measurement. After rotation, the new measurement is
$\{U\Pi_i U^{\dagger}\}$, where $U=\prod_k \exp(i s_k\lambda_k)$ with $\lambda_k$ a SU($N$)
generator and $s_k$ a real number in the regime $[0,2\pi]$. In this scenario, the goal is
to search a set of optimal coefficients $s_k$. The available algorithms include
particle swarm optimization (PSO) [[1]](#Kennedy1995), differential evolution (DE)
[[2]](#Storn1997), and automatic differentiation (AD) [[3]](#Baydin2018).

### **Base**
::: quanestimation.MeasurementSystem

### **Measurement optimization with AD**
::: quanestimation.AD_Mopt

## **Measurement Optimization with PSO**
::: quanestimation.PSO_Mopt

### **Measurement Optimization with DE**
::: quanestimation.DE_Mopt

---

## **Comprehensive Optimization**
In order to obtain the optimal parameter estimation schemes, it is necessary to
simultaneously optimize the probe state, control and measurement. The
comprehensive optimization for the probe state and measurement (SM), the probe
state and control (SC), the control and measurement (CM) and the probe state,
control and measurement (SCM) are proposed for this. In QuanEstimation, the
comprehensive optimization algorithms are particle swarm optimization (PSO),
differential evolution (DE), and automatic differentiation (AD).

### **Base**
::: quanestimation.ComprehensiveSystem

### **Comprehensive optimization with AD**
::: quanestimation.AD_Compopt

### **Comprehensive Optimization with PSO**
::: quanestimation.PSO_Compopt

### **Comprehensive Optimization with DE**
::: quanestimation.DE_Compopt

---

## **Adaptive measurement schemes**
In QuanEstimation, the Hamiltonian of the adaptive system should be written as
$H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$
the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the
Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$.
### **Adaptive measurement**
::: quanestimation.Adapt
::: quanestimation.Adapt_MZI

---

## **References**
<a id="Kennedy1995"></a>**[1]** J. Kennedy and R. Eberhart, Particle swarm optimization, 1995.
<a id="Storn1997"></a>**[2]** R. Storn and K. Price, Differential Evolution—A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces, 1997.
<a id="Baydin2018"></a>**[3]** A. G. Baydin et al., Automatic differentiation in machine learning: a survey, 2018.
