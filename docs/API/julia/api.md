This part contains the methods and structs in Julia that are called by the Python-Julia packagea and the full Julia package.

##  **`Main.QuanEstimation.AD`** &mdash; *Method*.

```julia
AD(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)
```

Optimization algorithm: AD.

  * `max_episode`: The number of episodes.
  * `epsilon`: Learning rate.
  * `beta1`: The exponential decay rate for the first moment estimates.
  * `beta2`: The exponential decay rate for the second moment estimates.
  * `Adam`: Whether or not to use Adam for updating control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L75-L85' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.CFIM_obj`** &mdash; *Method*.



```julia
CFIM_obj(;M=missing, W=missing, eps=GLOBAL_EPS)
```

Choose CFI [$\mathrm{Tr}(WI^{-1})$] as the objective function with $W$ the weight matrix and $I$ the CFIM.

  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/AsymptoticBound.jl#L34-L42' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.CMopt`** &mdash; *Type*.



```julia
CMopt(ctrl::Union{AbstractVector, Missing}=missing, M::Union{AbstractVector, Missing}=missing, ctrl_bound::AbstractVector=[-Inf, Inf])
```

Control and measurement optimization.

  * `ctrl`: Guessed control coefficients.
  * `M`: Guessed projective measurement (a set of basis)
  * `ctrl_bound`: Lower and upper bounds of the control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L100-L108' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.ControlOpt`** &mdash; *Type*.



```julia
ControlOpt(ctrl::Union{AbstractVector, Missing}=missing, ctrl_bound::AbstractVector=[-Inf, Inf])
```

Control optimization.

  * `ctrl`: Guessed control coefficients.
  * `ctrl_bound`: Lower and upper bounds of the control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L10-L17' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.DDPG`** &mdash; *Method*.



```julia
DDPG(;max_episode::Int=500, layer_num::Int=3, layer_dim::Int=200, seed::Number=1234)
```

Optimization algorithm: DE.

  * `max_episode`: The number of populations.
  * `layer_num`: The number of layers (include the input and output layer).
  * `layer_dim`: The number of neurons in the hidden layer.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L157-L166' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.DE`** &mdash; *Method*.



```julia
DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0, cr::Number=0.5, seed::Number=1234)
```

Optimization algorithm: DE.

  * `max_episode`: The number of populations.
  * `p_num`: The number of particles.
  * `ini_population`: Initial guesses of the optimization variables.
  * `c`: Mutation constant.
  * `cr`: Crossover constant.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L131-L142' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.GRAPE`** &mdash; *Method*.



```julia
GRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)
```

Control optimization algorithm: GRAPE.

  * `max_episode`: The number of episodes.
  * `epsilon`: Learning rate.
  * `beta1`: The exponential decay rate for the first moment estimates.
  * `beta2`: The exponential decay rate for the second moment estimates.
  * `Adam`: Whether or not to use Adam for updating control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L20-L30' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.HCRB_obj`** &mdash; *Method*.



```julia
HCRB_obj(;W=missing, eps=GLOBAL_EPS)
```

Choose HCRB as the objective function. 

  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/AsymptoticBound.jl#L45-L52' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(K::AbstractVector, dK::AbstractVector, ρ0::AbstractMatrix)
```

The parameterization of a state is $\rho=\sum_i K_i\rho_0K_i^{\dagger}$ with $\rho$ the evolved density matrix and $K_i$ the Kraus operator.

  * `K`: Kraus operators.
  * `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
  * `ρ0`: Initial state (density matrix).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/kraus/KrausData.jl#L16-L24' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(K::AbstractVector, dK::AbstractVector, ψ0::AbstractMatrix)
```

The parameterization of a state is $\psi\rangle=\sum_i K_i|\psi_0\rangle$ with $\psi$ the evolved state and $K_i$ the Kraus operator.

  * `K`: Kraus operators.
  * `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
  * `ψ0`: Initial state (ket).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/kraus/KrausData.jl#L27-L35' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(opt::AbstractMopt, ρ₀::AbstractMatrix, K, dK;rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Kraus operators for the measurement optimization. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/kraus/KrausWrapper.jl#L25-L30' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Kraus`** &mdash; *Method*.



```julia
Kraus(opt::CompOpt, K, dK;rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Kraus operators for the comprehensive optimization. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/kraus/KrausWrapper.jl#L41-L46' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.LLD`** &mdash; *Method*.



```julia
LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the left logarrithmic derivatives (LLDs). The LLD operator is defined as $\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho$, where ρ is the parameterized density matrix.    

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the LLD operator. Options can be: "original" (default) and "eigen".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L152-L161' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.LLD`** &mdash; *Method*.



```julia
LLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L166-L171' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::AbstractMopt, tspan, ρ₀, H0, dH; Hc=missing, ctrl=missing, decay=missing, rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the measurement optimization.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L205-L210' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::ControlMeasurementOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on control and measurement.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L363-L368' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::ControlOpt, tspan, ρ₀, H0, dH, Hc; decay=missing, rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the control optimization.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L2-L7' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateControlMeasurementOpt, tspan, H0, dH, Hc; decay=missing, rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state, control and measurement.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L509-L514' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateControlOpt, tspan, H0, dH, Hc; decay=missing, rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state and control.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L300-L305' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateMeasurementOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing, rng=GLOBAL_RNG)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the comprehensive optimization on state and measurement.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L421-L426' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Lindblad`** &mdash; *Method*.



```julia
Lindblad(opt::StateOpt, tspan, H0, dH; Hc=missing, ctrl=missing, decay=missing, rng=GLOBAL_RNG, eps=GLOBAL_EPS)
```

Initialize the parameterization described by the Lindblad master equation governed dynamics for the state optimization.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladWrapper.jl#L59-L64' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.NM`** &mdash; *Method*.



```julia
NM(;max_episode::Int=1000, state_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5, seed::Number=1234)
```

State optimization algorithm: NM.

  * `max_episode`: The number of populations.
  * `state_num`: The number of the input states.
  * `nelder_mead`: Initial guesses of the optimization variables.
  * `ar`: Reflection constant.
  * `ae`: Expansion constant.
  * `ac`: Constraction constant.
  * `as0`: Shrink constant.
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L185-L198' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.PSO`** &mdash; *Method*.



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
  * `seed`: Random seed.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L102-L114' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.QFIM_obj`** &mdash; *Method*.



```julia
QFIM_obj(;W=missing, eps=GLOBAL_EPS, LDtype::Symbol=:SLD)
```

Choose QFI [$\mathrm{Tr}(WF^{-1})$] as the objective function with $W$ the weight matrix and $F$ the QFIM.

  * `W`: Weight matrix.
  * `eps`: Machine epsilon.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/AsymptoticBound.jl#L23-L31' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.RLD`** &mdash; *Method*.



```julia
RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the right logarrithmic derivatives (RLDs). The RLD operator is defined as  $\partial_{a}\rho=\rho \mathcal{R}_a$, where $\rho$ is the parameterized density matrix.  

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the RLD operator. Options can be: "original" (default) and "eigen".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L93-L103' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.RLD`** &mdash; *Method*.



```julia
RLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L108-L113' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SCMopt`** &mdash; *Type*.



```julia
SCMopt(psi::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing, M::Union{AbstractVector, Missing}=missing, ctrl_bound::AbstractVector=[-Inf, Inf])
```

State, control and measurement optimization.

  * `psi`: Guessed probe state.
  * `ctrl`: Guessed control coefficients.
  * `M`: Guessed projective measurement (a set of basis).
  * `ctrl_bound`:  Lower and upper bounds of the control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L132-L141' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SCopt`** &mdash; *Type*.



```julia
SCopt(psi::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing, ctrl_bound::AbstractVector=[-Inf, Inf])
```

State and control optimization.

  * `psi`: Guessed probe state.
  * `ctrl`: Guessed control coefficients.
  * `ctrl_bound`: Lower and upper bounds of the control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L83-L91' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SLD`** &mdash; *Method*.



```julia
SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the symmetric logarrithmic derivatives (SLDs). The SLD operator $L_a$ is defined  as$\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)$, where $\rho$ is the parameterized density matrix. 

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the SLD operator. Options can be: "original" (default) and "eigen" .
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L7-L17' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SLD`** &mdash; *Method*.



```julia
SLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L22-L27' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SMopt`** &mdash; *Type*.



```julia
SMopt(psi::Union{AbstractVector, Missing}=missing, M::Union{AbstractVector, Missing}=missing)
```

State and control optimization.

  * `psi`: Guessed probe state.
  * `M`: Guessed projective measurement (a set of basis).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L116-L123' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.StateOpt`** &mdash; *Type*.



```julia
StateOpt(psi::Union{AbstractVector, Missing} = missing)
```

State optimization.

  * `psi`: Guessed probe state.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L26-L32' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.autoGRAPE`** &mdash; *Method*.



```julia
autoGRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)
```

Control optimization algorithm: auto-GRAPE.

  * `max_episode`: The number of episodes.
  * `epsilon`: Learning rate.
  * `beta1`: The exponential decay rate for the first moment estimates.
  * `beta2`: The exponential decay rate for the second moment estimates.
  * `Adam`: Whether or not to use Adam for updating control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/algorithm/algorithm.jl#L48-L58' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.BCFIM`** &mdash; *Method*.



```julia
BCFIM(x::V, p::V, rho::V, drho::V; M::Union{V,Nothing}=nothing, eps=GLOBAL_EPS) where{V<:AbstractVector}
```

Calculation of the Bayesian classical Fisher information (BCFI) and the Bayesian classical Fisher information matrix (BCFIM) of the form $\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}$ with $\mathcal{I}$ the CFIM and $p(\textbf{x})$ the prior distribution.

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L2-L14' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.BCRB`** &mdash; *Method*.



```julia
BCRB(x::V, p::V, rho::V, drho::V; M::Union{V,Nothing}=nothing, b::Union{V,Nothing}=nothing, db::Union{V,Nothing}=nothing, btype=1, eps=GLOBAL_EPS) where{V<:AbstractVector}
```

Calculation of the Bayesian Cramer-Rao bound (BCRB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `b`: Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
  * `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.
  * `btype`: Types of the BCRB. Options are 1 and 2.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L162-L176' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.BQCRB`** &mdash; *Method*.



```julia
BQCRB(x::V, p::V, rho::V, drho::V; b::Union{V,Nothing}=nothing, db::Union{V,Nothing}=nothing, LDtype=:SLD, btype=1, eps=GLOBAL_EPS) where{V<:AbstractVector}
```

Calculation of the Bayesian quantum Cramer-Rao bound (BQCRB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `b`: Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
  * `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `btype`: Types of the BCRB. Options are 1 and 2.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L79-L93' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.BQFIM`** &mdash; *Method*.



```julia
BQFIM(x::V, p::V, rho::V, drho::V; LDtype=:SLD, eps=GLOBAL_EPS) where{V<:AbstractVector}
```

Calculation of the Bayesian quantum Fisher information (BQFI) and the Bayesian quantum Fisher information matrix (BQFIM) of the form $\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x}$ with $\mathcal{F}$ the QFIM of all types and $p(\textbf{x})$ the prior distribution.

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L44-L56' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.Bayes`** &mdash; *Method*.



```julia
Bayes(x, p, rho, y; M=nothing, savefile=false)
```

Bayesian estimation. The prior distribution is updated via the posterior distribution obtained by the Bayes’ rule and the estimated value of parameters obtained via the maximum a posteriori probability (MAP).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `y`: The experimental results obtained in practice.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `savefile`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/common/BayesEstimation.jl#L1-L12' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps=GLOBAL_EPS) where {T<:Complex}
```

Calculate the classical Fisher information matrix (CFIM). 

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L285-L294' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; M=nothing, eps=GLOBAL_EPS) where {T<:Complex}
```

When the set of POVM is not given. Calculate the CFIM with SIC-POVM. The SIC-POVM is generated from the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L329-L334' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter. Calculate the classical Fisher information (CFI). 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L307-L312' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.CFIM`** &mdash; *Method*.



```julia
CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter and the set of POVM is not given. Calculate the CFI with SIC-POVM. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L348-L353' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.FIM`** &mdash; *Method*.



```julia
FIM(p::Vector{R}, dp::Vector{R}; eps=GLOBAL_EPS) where {R<:Real}
```

Calculation of the classical Fisher information matrix for classical scenarios. 

  * `p`: The probability distribution.
  * `dp`: Derivatives of the probability distribution on the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L553-L561' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.FIM`** &mdash; *Method*.



```julia
FIM(p::Vector{R}, dp::Vector{R}; eps=GLOBAL_EPS) where {R<:Real}
```

When applied to the case of single parameter and the set of POVM is not given. Calculate the classical Fisher information for classical scenarios. 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L532-L537' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.HCRB`** &mdash; *Method*.



```julia
HCRB(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, C::Matrix{Float64}; eps=GLOBAL_EPS) where {T<:Complex}
```

Caltulate the Holevo Cramer-Rao bound (HCRB) via the semidefinite program (SDP).

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/Holevo.jl#L7-L16' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.MLE`** &mdash; *Method*.



```julia
Bayes(x, p, rho, y; M=nothing, savefile=false)
```

Bayesian estimation. The estimated value of parameters obtained via the maximum likelihood estimation (MLE).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `y`: The experimental results obtained in practice.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `savefile`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/common/BayesEstimation.jl#L117-L128' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.MeasurementOpt`** &mdash; *Method*.



```julia
MeasurementOpt(mtype=:Projection, kwargs...)
```

Measurement optimization.

  * `mtype`: The type of scenarios for the measurement optimization. Options are `:Projection` (default), `:LC` and `:Rotation`.
  * `kwargs...`: keywords and the correponding default vaules. `mtype=:Projection`, `mtype=:LC` and `mtype=:Rotation`, the `kwargs...` are `M=missing`, `B=missing, POVM_basis=missing`, and `s=missing, POVM_basis=missing`, respectively.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/optim/optim.jl#L55-L62' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.OBB`** &mdash; *Method*.



```julia
OBB(x::V, p::V, dp::V, rho::V, drho::V, d2rho::V; LDtype=:SLD, eps=GLOBAL_EPS) where {V<:AbstractVector}
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


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L404-L417' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.QFIM`** &mdash; *Method*.



```julia
QFIM(ρ::Matrix{T}, dρ::Matrix{T}; LDtype=:SLD, eps=GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter. Calculation of the quantum Fisher information (QFI) for all types.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L371-L376' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.QFIM_Gauss`** &mdash; *Method*.



```julia
QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}
```

Calculate the SLD based quantum Fisher information matrix (QFIM) with gaussian states.  

  * `R̄` : First-order moment.
  * `dR̄`: Derivatives of the first-order moment with respect to the unknown parameters to be estimated. For example, dR[1] is the derivative vector on the first parameter.
  * `D`: Second-order moment.
  * `dD`: Derivatives of the second-order moment with respect to the unknown parameters to be estimated.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L635-L645' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.QFIM_Kraus`** &mdash; *Method*.



```julia
QFIM_Kraus(ρ0::Matrix{T}, K::Vector{Matrix{T}}, dK::Vector{Vector{Matrix{T}}}; LDtype=:SLD, exportLD::Bool=false, eps=eps_default) where {T<:Complex}
```

Calculation of the quantum Fisher information (QFI) and quantum Fisher information matrix (QFIM) with Kraus operator(s) for all types.

  * `ρ`: Density matrix.
  * `K`: Kraus operator(s).
  * `dK`: Derivatives of the Kraus operator(s) on the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
  * `exportLD`: Whether or not to export the values of logarithmic derivatives. If set True then the the values of logarithmic derivatives will be exported.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/AsymptoticBound/CramerRao.jl#L431-L443' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.QVTB`** &mdash; *Method*.



```julia
QVTB(x::V, p::V, dp::V, rho::V, drho::V; LDtype=:SLD, btype=1, eps=GLOBAL_EPS) where{V<:AbstractVector}
```

Calculation of the Bayesian version of Cramer-Rao bound in troduced by Van Trees (VTB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
  * `btype`: Types of the BCRB. Options are 1 and 2.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L249-L262' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.QZZB`** &mdash; *Method*.



```julia
QZZB(x::AbstractVector, p::AbstractVector, rho::AbstractVecOrMat; eps=GLOBAL_EPS)
```

Calculation of the quantum Ziv-Zakai bound (QZZB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho`: Parameterized density matrix.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/ZivZakai.jl#L25-L34' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SIC`** &mdash; *Method*.



```julia
SIC(dim::Int64)
```

Generation of a set of rank-one symmetric informationally complete positive operator-valued measure (SIC-POVM).

  * `dim`: The dimension of the system.

Note: SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/common/common.jl#L146-L153' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.SpinSqueezing`** &mdash; *Method*.



```julia
SpinSqueezing(ρ::AbstractMatrix; basis="Dicke", output="KU")
```

Calculate the spin squeezing parameter for the input density matrix. The `basis` can be `"Dicke"` for the Dicke basis, or `"Pauli"` for the Pauli basis. The `output` can be both `"KU"`(for spin squeezing defined by Kitagawa and Ueda) and `"WBIMH"`(for spin squeezing defined by Wineland et al.).


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/resources/resources.jl#L23-L29' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.TargetTime`** &mdash; *Method*.



```julia
TargetTime(f::Number, tspan::AbstractVector, func::Function, args...; kwargs...)
```

Calculate the minimum time to reach a precision limit of given level. The `func` can be any objective function during the control optimization, e.g. QFIM, CFIM, HCRB, etc.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/resources/resources.jl#L73-L78' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.VTB`** &mdash; *Method*.



```julia
VTB(x::V, p::V, dp::V, rho::V, drho::V; M::Union{V,Nothing}=nothing, btype=1, eps=GLOBAL_EPS) where{V<:AbstractVector}
```

Calculation of the Bayesian version of Cramer-Rao bound introduced by Van Trees (VTB).

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
  * `rho`: Parameterized density matrix.
  * `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `btype`: Types of the BCRB. Options are 1 and 2.
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/objective/BayesianBound/BayesianCramerRao.jl#L313-L326' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.adaptive`** &mdash; *Method*.



```julia
adaptive(x::AbstractVector, p, rho0::AbstractMatrix, tspan, H, dH; save_file=false, max_episode::Int=1000, eps::Float64=1e-8, Hc::Union{Vector,Nothing}=nothing, ctrl::Union{Vector,Nothing}=nothing, decay::Union{Vector,Nothing}=nothing, M::Union{AbstractVector,Nothing}=nothing, W::Union{Matrix,Nothing}=nothing)
```

In QuanEstimation, the Hamiltonian of the adaptive system should be written as $H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$ the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the  Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. 

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho0`: Density matrix.
  * `tspan`: The experimental results obtained in practice.
  * `H`: Free Hamiltonian with respect to the values in x.
  * `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated.
  * `savefile`: Whether or not to save all the posterior distributions.
  * `max_episode`: The number of episodes.
  * `eps`: Machine epsilon.
  * `Hc`: Control Hamiltonians.
  * `ctrl`: Control coefficients.
  * `decay`: Decay operators and the corresponding decay rates.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `W`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/common/adaptive.jl#L1-L23' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.adaptive`** &mdash; *Method*.



```julia
adaptive(x::AbstractVector, p, rho0::AbstractMatrix, K, dK; save_file=false, max_episode::Int=1000, eps::Float64=1e-8, M::Union{AbstractVector,Nothing}=nothing, W::Union{Matrix,Nothing}=nothing)
```

In QuanEstimation, the Hamiltonian of the adaptive system should be written as $H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$ the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the  Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. 

  * `x`: The regimes of the parameters for the integral.
  * `p`: The prior distribution.
  * `rho0`: Density matrix.
  * `K`: Kraus operator(s) with respect to the values in x.
  * `dK`: Derivatives of the Kraus operator(s) with respect to the unknown parameters to be estimated.
  * `savefile`: Whether or not to save all the posterior distributions.
  * `max_episode`: The number of episodes.
  * `eps`: Machine epsilon.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `W`: Whether or not to save all the posterior distributions.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/common/adaptive.jl#L218-L236' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.evolve`** &mdash; *Method*.



```julia
evolve(dynamics::Kraus{dm})
```

Evolution of density matrix under time-independent Hamiltonian without noise and controls.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/kraus/KrausDynamics.jl#L20-L25' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.evolve`** &mdash; *Method*.



```julia
evolve(dynamics::Kraus{ket})
```

Evolution of pure states under time-independent Hamiltonian without noise and controls


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/kraus/KrausDynamics.jl#L2-L7' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.expm`** &mdash; *Function*.



expm(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractVector, decay::Union{AbstractVector, Missing}=missing, Hc::Union{AbstractVector, Missing}=missing, ctrl::Union{AbstractVector, Missing}=missing)

The dynamics of a density matrix is of the form  $\partial_t\rho=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right)$, where $\rho$ is the evolved density matrix, $H$ is the Hamiltonian of the system, $\Gamma_i$ and $\gamma_i$ are the $i\mathrm{th}$ decay operator and the corresponding decay rate.

  * `tspan`: Time length for the evolution.
  * `ρ0`: Initial state (density matrix).
  * `H0`: Free Hamiltonian.
  * `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated. For example, dH[0] is the derivative vector on the first parameter.
  * `decay`: Decay operators and the corresponding decay rates. Its input rule is decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...], where $\Gamma_1$ $(\Gamma_2)$ represents the decay operator and $\gamma_1$ $(\gamma_2)$ is the corresponding decay rate.
  * `Hc`: Control Hamiltonians.
  * `ctrl`: Control coefficients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/dynamics/lindblad/LindbladDynamics.jl#L177-L189' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.mintime`** &mdash; *Method*.



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


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/run.jl#L43-L57' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.run`** &mdash; *Method*.



```julia
run(opt::AbstractOpt, alg::AbstractAlgorithm, obj::AbstractObj, dynamics::AbstractDynamics;savefile::Bool=false)
```

Run the optimization problem.

  * `opt`: Types of optimization, options are `ControlOpt`, `StateOpt`, `MeasurementOpt`, `SMopt`, `SCopt`, `CMopt` and `SCMopt`.
  * `alg`: Optimization algorithms, options are `auto-GRAPE`, `GRAPE`, `AD`, `PSO`, `DE`, 'NM' and `DDPG`.
  * `obj`: Objective function, options are `QFIM_obj`, `CFIM_obj` and `HCRB_obj`.
  * `dynamics`: Lindblad or Kraus parameterization process.
  * `savefile`: Whether or not to save all the control coeffients.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/run.jl#L24-L35' class='documenter-source'>source</a><br>

##  **`Main.QuanEstimation.suN_generator`** &mdash; *Method*.



```julia
suN_generator(n::Int64)
```

Generation of the SU($N$) generators with $N$ the dimension of the system.

  * `N`: The dimension of the system.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/38c09a785a9cfc9533ae704100edf8b3a24598dc/src/common/common.jl#L73-L79' class='documenter-source'>source</a><br>

