This part contains the methods and structs in Julia that are called by the Python-Julia packagea and the full Julia package.

##  **`QuanEstimation.AD`** &mdash; *Method*.



```julia
AD(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)
```

Optimization algorithm: AD.

  * `max_episode`: The number of episodes.
  * `epsilon`: Learning rate.
  * `beta1`: The exponential decay rate for the first moment estimates.
  * `beta2`: The exponential decay rate for the second moment estimates.
  * `Adam`: Whether or not to use Adam for updating control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L75-L85' class='documenter-source'>source</a><br>

##  **`QuanEstimation.CFIM_obj`** &mdash; *Method*.



```julia
CFIM_obj(;M=missing, W=missing, eps=GLOBAL_EPS)
```

Choose CFI [$\mathrm{Tr}(WI^{-1})$] as the objective function with $W$ the weight matrix and $I$ the CFIM.

  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/AsymptoticBound.jl#L34-L42' class='documenter-source'>source</a><br>

##  **`QuanEstimation.CMopt`** &mdash; *Type*.



```julia
CMopt(ctrl=missing, M=missing, ctrl_bound=[-Inf, Inf], seed=1234)
```

Control and measurement optimization.

  * `ctrl`: Guessed control coefficients.
  * `M`: Guessed projective measurement (a set of basis)
  * `ctrl_bound`: Lower and upper bounds of the control coefficients.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L127-L136' class='documenter-source'>source</a><br>

##  **`QuanEstimation.ControlOpt`** &mdash; *Method*.



```julia
ControlOpt(ctrl=missing, ctrl_bound=[-Inf, Inf], seed=1234)
```

Control optimization.

  * `ctrl`: Guessed control coefficients.
  * `ctrl_bound`: Lower and upper bounds of the control coefficients.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L16-L24' class='documenter-source'>source</a><br>

<!-- ##  **`QuanEstimation.DDPG`** &mdash; *Method*.



```julia
DDPG(;max_episode::Int=500, layer_num::Int=3, layer_dim::Int=200, seed::Number=1234)
``` -->

Optimization algorithm: DE.

  * `max_episode`: The number of populations.
  * `layer_num`: The number of layers (include the input and output layer).
  * `layer_dim`: The number of neurons in the hidden layer.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L146-L155' class='documenter-source'>source</a><br>

##  **`QuanEstimation.DE`** &mdash; *Method*.



```julia
DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0, cr::Number=0.5, seed::Number=1234)
```

Optimization algorithm: DE.

  * `max_episode`: The number of populations.
  * `p_num`: The number of particles.
  * `ini_population`: Initial guesses of the optimization variables.
  * `c`: Mutation constant.
  * `cr`: Crossover constant.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L122-L132' class='documenter-source'>source</a><br>

##  **`QuanEstimation.GRAPE`** &mdash; *Method*.



```julia
GRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)
```

Control optimization algorithm: GRAPE.

  * `max_episode`: The number of episodes.
  * `epsilon`: Learning rate.
  * `beta1`: The exponential decay rate for the first moment estimates.
  * `beta2`: The exponential decay rate for the second moment estimates.
  * `Adam`: Whether or not to use Adam for updating control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L20-L30' class='documenter-source'>source</a><br>

##  **`QuanEstimation.HCRB_obj`** &mdash; *Method*.



```julia
HCRB_obj(;W=missing, eps=GLOBAL_EPS)
```

Choose HCRB as the objective function. 

  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/AsymptoticBound.jl#L45-L52' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector)
```

The parameterization of a state is $\rho=\sum_i K_i\rho_0K_i^{\dagger}$ with $\rho$ the evolved density matrix and $K_i$ the Kraus operator.

  * `ρ0`: Initial state (density matrix).
  * `K`: Kraus operators.
  * `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Kraus/KrausData.jl#L16-L24' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(ψ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector)
```

The parameterization of a state is $\psi\rangle=\sum_i K_i|\psi_0\rangle$ with $\psi$ the evolved state and $K_i$ the Kraus operator.

  * `ψ0`: Initial state (ket).
  * `K`: Kraus operators.
  * `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Kraus/KrausData.jl#L27-L35' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(opt::AbstractMopt, ρ₀::AbstractMatrix, K, dK; eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Kraus operators for the measurement optimization. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Kraus/KrausWrapper.jl#L25-L30' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(opt::CompOpt, K, dK; eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Kraus operators for the comprehensive optimization. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Kraus/KrausWrapper.jl#L41-L46' class='documenter-source'>source</a><br>

##  **`QuanEstimation.LLD`** &mdash; *Method*.



```julia
LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the left logarrithmic derivatives (LLDs). The LLD operator is defined as $\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho$, where ρ is the parameterized density matrix.    

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the LLD operator. Options can be: "original" (default) and "eigen".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L146-L155' class='documenter-source'>source</a><br>

##  **`QuanEstimation.LLD`** &mdash; *Method*.



```julia
LLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L160-L165' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::AbstractMopt, tspan, ρ₀, H0, dH; Hc=missing, ctrl=missing, decay=missing, dyn_method=:Expm, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the measurement optimization.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L203-L208' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::ControlMeasurementOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, dyn_method=:Expm, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on control and measurement.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L361-L366' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::ControlOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, dyn_method=:Expm, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the control optimization.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L2-L7' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateControlMeasurementOpt, tspan, H0, dH, Hc; decay=missing, dyn_method=:Expm, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state, control and measurement.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L507-L512' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateControlOpt, tspan, H0, dH, Hc; decay=missing, dyn_method=:Expm, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state and control.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L298-L303' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateMeasurementOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing, dyn_method=:Expm)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state and measurement.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L419-L424' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing, dyn_method=:Expm, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the state optimization.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladWrapper.jl#L59-L64' class='documenter-source'>source</a><br>

##  **`QuanEstimation.NM`** &mdash; *Method*.



```julia
NM(;max_episode::Int=1000, p_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5, seed::Number=1234)
```

State optimization algorithm: NM.

  * `max_episode`: The number of populations.
  * `p_num`: The number of the input states.
  * `nelder_mead`: Initial guesses of the optimization variables.
  * `ar`: Reflection constant.
  * `ae`: Expansion constant.
  * `ac`: Constraction constant.
  * `as0`: Shrink constant.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L169-L181' class='documenter-source'>source</a><br>

##  **`QuanEstimation.PSO`** &mdash; *Method*.



```julia
PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=missing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0, seed::Number=1234)
```

Optimization algorithm: PSO.

  * `max_episode`: The number of episodes, it accepts both integer and array with two elements.
  * `p_num`: The number of particles.
  * `ini_particle`: Initial guesses of the optimization variables.
  * `c0`: The damping factor that assists convergence, also known as inertia weight.
  * `c1`: The exploitation weight that attracts the particle to its best previous position, also known as cognitive learning factor.
  * `c2`: The exploitation weight that attracts the particle to the best position in the neighborhood, also known as social learning factor.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L99-L110' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QFIM_obj`** &mdash; *Method*.



```julia
QFIM_obj(;W=missing, eps=GLOBAL_EPS, LDtype::Symbol=:SLD)
```

Choose QFI [$\mathrm{Tr}(WF^{-1})$] as the objective function with $W$ the weight matrix and $F$ the QFIM.

  * `W`: Weight matrix.
  * `eps`: Machine epsilon.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/AsymptoticBound.jl#L23-L31' class='documenter-source'>source</a><br>

##  **`QuanEstimation.RI`** &mdash; *Method*.



```julia
RI(;max_episode::Int=300, seed::Number=1234)
```

State optimization algorithm: RI.

  * `max_episode`: The number of episodes.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L188-L194' class='documenter-source'>source</a><br>

##  **`QuanEstimation.RLD`** &mdash; *Method*.



```julia
RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the right logarrithmic derivatives (RLDs). The RLD operator is defined as  $\partial_{a}\rho=\rho \mathcal{R}_a$, where $\rho$ is the parameterized density matrix.  

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the RLD operator. Options can be: "original" (default) and "eigen".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L89-L99' class='documenter-source'>source</a><br>

##  **`QuanEstimation.RLD`** &mdash; *Method*.



```julia
RLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L104-L109' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SCMopt`** &mdash; *Type*.



```julia
SCMopt(psi=missing, ctrl=missing, M=missing, ctrl_bound=[-Inf, Inf], seed=1234)
```

State, control and measurement optimization.

  * `psi`: Guessed probe state.
  * `ctrl`: Guessed control coefficients.
  * `M`: Guessed projective measurement (a set of basis).
  * `ctrl_bound`:  Lower and upper bounds of the control coefficients.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L167-L177' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SCopt`** &mdash; *Type*.



```julia
SCopt(psi=missing, ctrl=missing, ctrl_bound=[-Inf, Inf], seed=1234)
```

State and control optimization.

  * `psi`: Guessed probe state.
  * `ctrl`: Guessed control coefficients.
  * `ctrl_bound`: Lower and upper bounds of the control coefficients.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L106-L115' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SLD`** &mdash; *Method*.



```julia
SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the symmetric logarrithmic derivatives (SLDs). The SLD operator $L_a$ is defined  as$\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)$, where $\rho$ is the parameterized density matrix. 

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the SLD operator. Options can be: "original" (default) and "eigen" .
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L7-L17' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SLD`** &mdash; *Method*.



```julia
SLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L22-L27' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SMopt`** &mdash; *Type*.



```julia
SMopt(psi=missing, M=missing, seed=1234)
```

State and control optimization.

  * `psi`: Guessed probe state.
  * `M`: Guessed projective measurement (a set of basis).
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L146-L154' class='documenter-source'>source</a><br>

##  **`QuanEstimation.StateOpt`** &mdash; *Method*.



```julia
StateOpt(psi=missing, seed=1234)
```

State optimization.

  * `psi`: Guessed probe state.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L35-L42' class='documenter-source'>source</a><br>

##  **`QuanEstimation.autoGRAPE`** &mdash; *Method*.



```julia
autoGRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)
```

Control optimization algorithm: auto-GRAPE.

  * `max_episode`: The number of episodes.
  * `epsilon`: Learning rate.
  * `beta1`: The exponential decay rate for the first moment estimates.
  * `beta2`: The exponential decay rate for the second moment estimates.
  * `Adam`: Whether or not to use Adam for updating control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Algorithm/Algorithm.jl#L48-L58' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Adapt`** &mdash; *Method*.



```julia
Adapt(x::AbstractVector, p, rho0::AbstractMatrix, tspan, H, dH; dyn_method=:Expm, method="FOP", savefile=false, max_episode::Int=1000, eps::Float64=1e-8, Hc=missing, ctrl=missing, decay=missing, M=missing, W=missing)
```

In QuanEstimation, the Hamiltonian of the adaptive system should be written as $H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$ the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the  Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. 

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho0`: Density matrix.
  * `tspan`: The experimental results obtained in practice.
  * `H`: Free Hamiltonian with respect to the values in x.
  * `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated.
  * `dyn_method`: Setting the method for solving the Lindblad dynamics. Options are: "expm" and "ode".
  * `method`: Choose the method for updating the tunable parameters (u). Options are: "FOP" and "MI".
  * `savefile`: Whether or not to save all the posterior distributions.
  * `max_episode`: The number of episodes.
  * `eps`: Machine epsilon.
  * `Hc`: Control Hamiltonians.
  * `ctrl`: Control coefficients.
  * `decay`: Decay operators and the corresponding decay rates.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `W`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/AdaptiveScheme.jl#L1-L25' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Adapt`** &mdash; *Method*.



```julia
Adapt(x::AbstractVector, p, rho0::AbstractMatrix, K, dK; method="FOP", savefile=false, max_episode::Int=1000, eps::Float64=1e-8, M=missing, W=missing)
```

In QuanEstimation, the Hamiltonian of the adaptive system should be written as $H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$ the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the  Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. 

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho0`: Density matrix.
  * `K`: Kraus operator(s) with respect to the values in x.
  * `dK`: Derivatives of the Kraus operator(s) with respect to the unknown parameters to be estimated.
  * `method`: Choose the method for updating the tunable parameters (u). Options are: "FOP" and "MI".
  * `savefile`: Whether or not to save all the posterior distributions.
  * `max_episode`: The number of episodes.
  * `eps`: Machine epsilon.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `W`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/AdaptiveScheme.jl#L169-L188' class='documenter-source'>source</a><br>

##  **`QuanEstimation.BCB`** &mdash; *Method*.



```julia
BCB(x, p, rho; W=missing, eps=GLOBAL_EPS)
```

Calculation of the minimum Bayesian cost with a quadratic cost function.

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/BayesEstimation.jl#L363-L373' class='documenter-source'>source</a><br>

##  **`QuanEstimation.BCFIM`** &mdash; *Method*.



```julia
BCFIM(x::AbstractVector, p, rho, drho; M=missing, eps=GLOBAL_EPS)
```

Calculation of the Bayesian classical Fisher information (BCFI) and the Bayesian classical Fisher information matrix (BCFIM) of the form $\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}$ with $\mathcal{I}$ the CFIM and $p(\textbf{x})$ the prior distribution.

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L2-L14' class='documenter-source'>source</a><br>

##  **`QuanEstimation.BCRB`** &mdash; *Method*.



```julia
BCRB(x::AbstractVector, p, dp, rho, drho; M=missing, b=missing, db=missing, btype=1, eps=GLOBAL_EPS)
```

Calculation of the Bayesian Cramer-Rao bound (BCRB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `b`: Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
  * `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.
  * `btype`: Types of the BCRB. Options are 1, 2 and 3.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L176-L191' class='documenter-source'>source</a><br>

##  **`QuanEstimation.BQCRB`** &mdash; *Method*.



```julia
BQCRB(x::AbstractVector, p, dp, rho, drho; b=missing, db=missing, LDtype=:SLD, btype=1, eps=GLOBAL_EPS)
```

Calculation of the Bayesian quantum Cramer-Rao bound (BQCRB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `b`: Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
  * `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `btype`: Types of the BCRB. Options are 1, 2 and 3.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L79-L94' class='documenter-source'>source</a><br>

##  **`QuanEstimation.BQFIM`** &mdash; *Method*.



```julia
BQFIM(x::AbstractVector, p, rho, drho; LDtype=:SLD, eps=GLOBAL_EPS)
```

Calculation of the Bayesian quantum Fisher information (BQFI) and the Bayesian quantum Fisher information matrix (BQFIM) of the form $\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x}$ with $\mathcal{F}$ the QFIM of all types and $p(\textbf{x})$ the prior distribution.

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L44-L56' class='documenter-source'>source</a><br>

##  **`QuanEstimation.Bayes`** &mdash; *Method*.



```julia
Bayes(x, p, rho, y; M=missing, savefile=false)
```

Bayesian estimation. The prior distribution is updated via the posterior distribution obtained by the Bayes' rule and the estimated value of parameters obtained via the maximum a posteriori probability (MAP).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `y`: The experimental results obtained in practice.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `savefile`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/BayesEstimation.jl#L1-L12' class='documenter-source'>source</a><br>

##  **`QuanEstimation.BayesCost`** &mdash; *Method*.



```julia
BayesCost(x, p, xest, rho, M; W=missing, eps=GLOBAL_EPS)
```

Calculation of the average Bayesian cost with a quadratic cost function.

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `xest`: The estimators.
  * `rho`: Parameterized density matrix.
  * `M`: A set of POVM.
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/BayesEstimation.jl#L314-L326' class='documenter-source'>source</a><br>

##  **`QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the classical Fisher information matrix (CFIM). 

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L278-L287' class='documenter-source'>source</a><br>

##  **`QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; M=nothing, eps=GLOBAL_EPS) where {T<:Complex}
```

When the set of POVM is not given. Calculate the CFIM with SIC-POVM. The SIC-POVM is generated from the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L322-L327' class='documenter-source'>source</a><br>

##  **`QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter. Calculate the classical Fisher information (CFI). 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L300-L305' class='documenter-source'>source</a><br>

##  **`QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter and the set of POVM is not given. Calculate the CFI with SIC-POVM. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L341-L346' class='documenter-source'>source</a><br>

##  **`QuanEstimation.FIM`** &mdash; *Method*.



```julia
FIM(p::Vector{R}, dp::Vector{R}; eps=GLOBAL_EPS) where {R<:Real}
```

Calculation of the classical Fisher information matrix for classical scenarios. 

  * `p`: The probability distribution.
  * `dp`: Derivatives of the probability distribution on the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L529-L537' class='documenter-source'>source</a><br>

##  **`QuanEstimation.FIM`** &mdash; *Method*.



```julia
FIM(p::Vector{R}, dp::Vector{R}; eps=GLOBAL_EPS) where {R<:Real}
```

When applied to the case of single parameter and the set of POVM is not given. Calculate the classical Fisher information for classical scenarios. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L508-L513' class='documenter-source'>source</a><br>

##  **`QuanEstimation.FI_Expt`** &mdash; *Method*.



```julia
FI_Expt(y1, y2, dx; ftype=:norm)
```

Calculate the classical Fisher information (CFI) based on the experiment data.

  * `y1`: Experimental data obtained at the truth value (x).
  * `y1`: Experimental data obtained at x+dx.
  * `dx`: A known small drift of the parameter.
  * `ftype`: The distribution the data follows. Options are: norm, gamma, rayleigh, and poisson.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L568-L577' class='documenter-source'>source</a><br>

##  **`QuanEstimation.HCRB`** &mdash; *Method*.



```julia
HCRB(ρ::AbstractMatrix, dρ::AbstractVector, C::AbstractMatrix; eps=GLOBAL_EPS)
```

Caltulate the Holevo Cramer-Rao bound (HCRB) via the semidefinite program (SDP).

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/AnalogCramerRao.jl#L7-L16' class='documenter-source'>source</a><br>

##  **`QuanEstimation.MLE`** &mdash; *Method*.



```julia
MLE(x, rho, y; M=missing, savefile=false)
```

Bayesian estimation. The estimated value of parameters obtained via the maximum likelihood estimation (MLE).

  * `x`: The regimes of the parameters for the integral.
  * `rho`: Parameterized density matrix.
  * `y`: The experimental results obtained in practice.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `savefile`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/BayesEstimation.jl#L180-L190' class='documenter-source'>source</a><br>

##  **`QuanEstimation.MeasurementOpt`** &mdash; *Method*.



```julia
MeasurementOpt(mtype=:Projection, kwargs...)
```

Measurement optimization.

  * `mtype`: The type of scenarios for the measurement optimization. Options are `:Projection` (default), `:LC` and `:Rotation`.
  * `kwargs...`: keywords and the correponding default vaules. `mtype=:Projection`, `mtype=:LC` and `mtype=:Rotation`, the `kwargs...` are `M=missing`, `B=missing, POVM_basis=missing`, and `s=missing, POVM_basis=missing`, respectively.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/OptScenario/OptScenario.jl#L75-L82' class='documenter-source'>source</a><br>

##  **`QuanEstimation.NHB`** &mdash; *Method*.



```julia
NHB(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix)
```

Nagaoka-Hayashi bound (NHB) via the semidefinite program (SDP).

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
  * `W`: Weight matrix.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/AnalogCramerRao.jl#L98-L106' class='documenter-source'>source</a><br>

##  **`QuanEstimation.OBB`** &mdash; *Method*.



```julia
OBB(x::AbstractVector, p, dp, rho, drho, d2rho; LDtype=:SLD, eps=GLOBAL_EPS)
```

Calculation of the Bayesian version of Cramer-Rao bound introduced by Van Trees (VTB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `d2rho`: Second order Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L423-L436' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QFIM`** &mdash; *Method*.



```julia
QFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; LDtype=:SLD, exportLD::Bool= false, eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter. Calculation of the quantum Fisher information (QFI) for all types.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L393-L398' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QFIM`** &mdash; *Method*.



```julia
QFIM(ρ::Matrix{T}, dρ::Matrix{T}; LDtype=:SLD, exportLD::Bool= false, eps=GLOBAL_EPS) where {T<:Complex}
```

Calculation of the quantum Fisher information (QFI) for all types. 

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
  * `exportLD`: export logarithmic derivatives apart from F.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L365-L375' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QFIM_Gauss`** &mdash; *Method*.



```julia
QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}
```

Calculate the SLD based quantum Fisher information matrix (QFIM) with gaussian states.  

  * `R̄` : First-order moment.
  * `dR̄`: Derivatives of the first-order moment with respect to the unknown parameters to be estimated. For example, dR[1] is the derivative vector on the first parameter.
  * `D`: Second-order moment.
  * `dD`: Derivatives of the second-order moment with respect to the unknown parameters to be estimated.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L661-L671' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QFIM_Kraus`** &mdash; *Method*.



```julia
QFIM_Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector; LDtype=:SLD, exportLD::Bool=false, eps=GLOBAL_EPS)
```

Calculation of the quantum Fisher information (QFI) and quantum Fisher information matrix (QFIM) with Kraus operator(s) for all types.

  * `ρ0`: Density matrix.
  * `K`: Kraus operator(s).
  * `dK`: Derivatives of the Kraus operator(s) on the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
  * `exportLD`: Whether or not to export the values of logarithmic derivatives. If set True then the the values of logarithmic derivatives will be exported.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/AsymptoticBound/CramerRao.jl#L418-L429' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QVTB`** &mdash; *Method*.



```julia
QVTB(x::AbstractVector, p, dp, rho, drho; LDtype=:SLD, eps=GLOBAL_EPS)
```

Calculation of the Bayesian version of Cramer-Rao bound in troduced by Van Trees (VTB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L294-L306' class='documenter-source'>source</a><br>

##  **`QuanEstimation.QZZB`** &mdash; *Method*.



```julia
QZZB(x::AbstractVector, p::AbstractVector, rho::AbstractVecOrMat; eps=GLOBAL_EPS)
```

Calculation of the quantum Ziv-Zakai bound (QZZB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/ZivZakai.jl#L25-L34' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SIC`** &mdash; *Method*.



```julia
SIC(dim::Int64)
```

Generation of a set of rank-one symmetric informationally complete positive operator-valued measure (SIC-POVM).

  * `dim`: The dimension of the system.

Note: SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/Common.jl#L146-L153' class='documenter-source'>source</a><br>

##  **`QuanEstimation.SpinSqueezing`** &mdash; *Method*.



```julia
SpinSqueezing(ρ::AbstractMatrix; basis="Dicke", output="KU")
```

Calculate the spin squeezing parameter for the input density matrix. The `basis` can be `"Dicke"` for the Dicke basis, or `"Pauli"` for the Pauli basis. The `output` can be both `"KU"`(for spin squeezing defined by Kitagawa and Ueda) and `"WBIMH"`(for spin squeezing defined by Wineland et al.).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Resource/Resource.jl#L23-L29' class='documenter-source'>source</a><br>

##  **`QuanEstimation.TargetTime`** &mdash; *Method*.



```julia
TargetTime(f::Number, tspan::AbstractVector, func::Function, args...; kwargs...)
```

Calculate the minimum time to reach a precision limit of given level. The `func` can be any objective function during the control optimization, e.g. QFIM, CFIM, HCRB, etc.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Resource/Resource.jl#L73-L78' class='documenter-source'>source</a><br>

##  **`QuanEstimation.VTB`** &mdash; *Method*.



```julia
VTB(x::AbstractVector, p, dp, rho, drho; M=missing, eps=GLOBAL_EPS)
```

Calculation of the Bayesian version of Cramer-Rao bound introduced by Van Trees (VTB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/ObjectiveFunc/BayesianBound/BayesianCramerRao.jl#L344-L356' class='documenter-source'>source</a><br>

##  **`QuanEstimation.evolve`** &mdash; *Method*.



```julia
evolve(dynamics::Kraus{dm})
```

Evolution of density matrix under time-independent Hamiltonian without noise and controls.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Kraus/KrausDynamics.jl#L20-L25' class='documenter-source'>source</a><br>

##  **`QuanEstimation.evolve`** &mdash; *Method*.



```julia
evolve(dynamics::Kraus{ket})
```

Evolution of pure states under time-independent Hamiltonian without noise and controls


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Kraus/KrausDynamics.jl#L2-L7' class='documenter-source'>source</a><br>

##  **`QuanEstimation.expm`** &mdash; *Method*.



```julia
expm(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractMatrix; decay::Union{AbstractVector, Missing}=missing, Hc::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing)
```

When applied to the case of single parameter. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladDynamics.jl#L112-L117' class='documenter-source'>source</a><br>

##  **`QuanEstimation.expm`** &mdash; *Method*.



```julia
expm(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractVector; decay::Union{AbstractVector, Missing}=missing, Hc::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing)
```

The dynamics of a density matrix is of the form  $\partial_t\rho=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right)$, where $\rho$ is the evolved density matrix, $H$ is the Hamiltonian of the system, $\Gamma_i$ and $\gamma_i$ are the $i\mathrm{th}$ decay operator and the corresponding decay rate.

  * `tspan`: Time length for the evolution.
  * `ρ0`: Initial state (density matrix).
  * `H0`: Free Hamiltonian.
  * `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated. For example, dH[0] is the derivative vector on the first parameter.
  * `decay`: Decay operators and the corresponding decay rates. Its input rule is decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...], where $\Gamma_1$ $(\Gamma_2)$ represents the decay operator and $\gamma_1$ $(\gamma_2)$ is the corresponding decay rate.
  * `Hc`: Control Hamiltonians.
  * `ctrl`: Control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladDynamics.jl#L184-L196' class='documenter-source'>source</a><br>

##  **`QuanEstimation.mintime`** &mdash; *Method*.



```julia
mintime(f::Number, opt::ControlOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics; savefile::Bool=false, method::String="binary")
```

Search of the minimum time to reach a given value of the objective function.

  * `f`: The given value of the objective function.
  * `opt`: Control Optimization.
  * `alg`: Optimization algorithms, options are `auto-GRAPE`, `GRAPE`, `PSO`, `DE` and `DDPG`.
  * `obj`: Objective function, options are `QFIM_obj`, `CFIM_obj` and `HCRB_obj`.
  * `dynamics`: Lindblad dynamics.
  * `savefile`: Whether or not to save all the control coeffients.
  * `method`: Methods for searching the minimum time to reach the given value of the objective function. Options are `binary` and `forward`.
  * `system`: control system.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/run.jl#L57-L71' class='documenter-source'>source</a><br>

##  **`QuanEstimation.ode`** &mdash; *Method*.



```julia
ode(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractMatrix; decay::Union{AbstractVector, Missing}=missing, Hc::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing)
```

When applied to the case of single parameter. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladDynamics.jl#L393-L398' class='documenter-source'>source</a><br>

##  **`QuanEstimation.ode`** &mdash; *Method*.



```julia
ode(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractVector, dH::AbstractVector; decay::Union{AbstractVector, Missing}=missing, Hc::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing)
```

The dynamics of a density matrix is of the form  $\partial_t\rho=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right)$, where $\rho$ is the evolved density matrix, $H$ is the Hamiltonian of the system, $\Gamma_i$ and $\gamma_i$ are the $i\mathrm{th}$ decay operator and the corresponding decay rate.

  * `tspan`: Time length for the evolution.
  * `ρ0`: Initial state (density matrix).
  * `H0`: Free Hamiltonian.
  * `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated. For example, dH[0] is the derivative vector on the first parameter.
  * `decay`: Decay operators and the corresponding decay rates. Its input rule is decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...], where $\Gamma_1$ $(\Gamma_2)$ represents the decay operator and $\gamma_1$ $(\gamma_2)$ is the corresponding decay rate.
  * `Hc`: Control Hamiltonians.
  * `ctrl`: Control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Parameterization/Lindblad/LindbladDynamics.jl#L467-L479' class='documenter-source'>source</a><br>

##  **`QuanEstimation.offline`** &mdash; *Method*.



```julia
offline(apt::Adapt_MZI, alg; target::Symbol=:sharpness, eps = GLOBAL_EPS, seed=1234)
```

Offline adaptive phase estimation in the MZI.

  * `apt`: Adaptive MZI struct which contains `x`, `p`, and `rho0`.
  * `alg`: The algorithms for searching the optimal tunable phase. Here, DE and PSO are available.
  * `target`: Setting the target function for calculating the tunable phase. Options are: "sharpness" and "MI".
  * `eps`: Machine epsilon.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/AdaptiveScheme.jl#L691-L701' class='documenter-source'>source</a><br>

##  **`QuanEstimation.online`** &mdash; *Method*.



```julia
online(apt::Adapt_MZI; target::Symbol=:sharpness, output::String="phi")
```

Online adaptive phase estimation in the MZI.

  * `apt`: Adaptive MZI struct which contains x, p, and rho0.
  * `target`: Setting the target function for calculating the tunable phase. Options are: "sharpness" and "MI".
  * `output`: Choose the output variables. Options are: "phi" and "dphi".


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/AdaptiveScheme.jl#L568-L576' class='documenter-source'>source</a><br>

##  **`QuanEstimation.run`** &mdash; *Method*.



```julia
run(opt::AbstractOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics; savefile::Bool=false)
```

Run the optimization problem.

  * `opt`: Types of optimization, options are `ControlOpt`, `StateOpt`, `MeasurementOpt`, `SMopt`, `SCopt`, `CMopt` and `SCMopt`.
  * `alg`: Optimization algorithms, options are `auto-GRAPE`, `GRAPE`, `AD`, `PSO`, `DE`, 'NM' and `DDPG`.
  * `obj`: Objective function, options are `QFIM_obj`, `CFIM_obj` and `HCRB_obj`.
  * `dynamics`: Lindblad or Kraus parameterization process.
  * `savefile`: Whether or not to save all the control coeffients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/run.jl#L24-L35' class='documenter-source'>source</a><br>

##  **`QuanEstimation.suN_generator`** &mdash; *Method*.



```julia
suN_generator(n::Int64)
```

Generation of the SU($N$) generators with $N$ the dimension of the system.

  * `N`: The dimension of the system.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/5f47a686c13b059023de2abe67017d7c9564bc9d/src/Common/Common.jl#L73-L79' class='documenter-source'>source</a><br>

