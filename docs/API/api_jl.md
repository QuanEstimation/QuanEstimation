


<a id='QuanEstimation'></a>

<a id='QuanEstimation-1'></a>

# QuanEstimation

## Asymptotic Bounds
**`QuanEstimation.LLD`** &mdash; *Method*.

```jl
LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}
```

Calculate the left logarrithmic derivatives (LLDs). The LLD operator is defined as $\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho$, where ρ is the parameterized density matrix.

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the LLD operator. Options can be:
    - "original" (default) -- The RLD matrix will be written in terms of the same basis as the input density matrix (ρ).
    - "eigen" -- The RLD matrix will be written in terms of the eigenbasis of the input ρ.
  * `eps`: Machine epsilon


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L157-L169' class='documenter-source'>source</a><br>

**`QuanEstimation.LLD`** &mdash; *Method*.

```jl
LLD(ρ::Matrix{T},dρ::Matrix{T};rep = "original",eps = GLOBAL_EPS,) where {T<:Complex}
```

When applied to the case of single parameter.

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L174-L179' class='documenter-source'>source</a><br>

**`QuanEstimation.RLD`** &mdash; *Method*.

```jl
RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}
```

Calculate the right logarrithmic derivatives (RLDs). The RLD operator is defined as $\partial_{a}\rho=\rho \mathcal{R}_a$, where ρ is the parameterized density matrix.

  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the RLD operator. Options can be:
    - "original" (default) -- The RLD matrix will be written in terms of the same basis as the input density matrix (ρ).
    - "eigen" -- The RLD matrix will be written in terms of the eigenbasis of the input ρ.

  * `eps`: Machine epsilon

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L95-L108' class='documenter-source'>source</a><br>

**`QuanEstimation.RLD`** &mdash; *Method*.

```julia
RLD(ρ::Matrix{T},dρ::Matrix{T};rep = "original",eps = GLOBAL_EPS,) where {T<:Complex}
```

When applied to the case of single parameter.

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L113-L118' class='documenter-source'>source</a><br>

**`QuanEstimation.SLD`** &mdash; *Method*.

```julia
SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}
```

Calculate the symmetric logarrithmic derivatives (SLDs).The SLD operator $L_a$ is defined as$\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)$, where ρ is the parameterized density matrix.

  * `ρ`: Density matrix.
  * `dρ`:  Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `rep`: Representation of the SLD operator. Options can be:
    - "original" (default) -- The SLD matrix will be written in terms of the same basis as the input density matrix (ρ).
    - "eigen" -- The SLD matrix will be written in terms of the eigenbasis of the input ρ.
  * `eps`: Machine epsilon

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L7-L19' class='documenter-source'>source</a><br>

**`QuanEstimation.SLD`** &mdash; *Method*.

```julia
SLD(ρ::Matrix{T},dρ::Matrix{T};rep = "original",eps = GLOBAL_EPS,) where {T<:Complex}
```

When applied to the case of single parameter.

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L24-L29' class='documenter-source'>source</a><br>

**`QuanEstimation.CFIM`** &mdash; *Method*.

```julia
CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps = GLOBAL_EPS) where {T<:Complex}
```

Calculate the classical Fisher information matrix (CFIM). 

  * `ρ`: Density matrix.
  * `dρ`:  Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
  * `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
  * `eps`: Machine epsilon.


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L293-L303' class='documenter-source'>source</a><br>

**`QuanEstimation.CFIM`** &mdash; *Method*.

```julia
CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
```

When the set of POVM is not given. Calculate the CFIM with SIC-POVM. The SIC-POVM is generated from the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from the [website](http://www.physics.umb.edu/Research/QBism/solutions.html).

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L338-L343' class='documenter-source'>source</a><br>

**`QuanEstimation.CFIM`** &mdash; *Method*.

```julia
CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps = GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter. Calculate the classical Fisher information(CFI). 


<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L316-L321' class='documenter-source'>source</a><br>

**`QuanEstimation.CFIM`** &mdash; *Method*.

```julia
CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
```

When applied to the case of single parameter and the set of POVM is not given. Calculate the CFI with SIC-POVM. 

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L357-L362' class='documenter-source'>source</a><br>

**`QuanEstimation.HCRB`** &mdash; *Method*.

```julia
HCRB(ρ::Matrix{T},dρ::Vector{Matrix{T}},C::Matrix{Float64};eps = 1e-8,) where {T<:Complex}
```

Caltulate the Holevo Cramer-Rao bound (HCRB) via the semidefinite program (SDP).
  * `ρ`: Density matrix.
  * `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
  * `W`: Weight matrix.
  * `eps`: Machine epsilon.

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/Holevo.jl#L7-L21' class='documenter-source'>source</a><br>

**`QuanEstimation.QFIM_Gauss`** &mdash; *Method*.

```julia
QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}
```

Calculate the SLD based quantum Fisher information matrix (QFIM) with gaussian states.
  * `R̄` : First-order moment.
  * `dR̄`: Derivatives of the first-order moment with respect to the unknown parameters to be estimated. For example, dR[1] is the derivative vector on the first  parameter.
  * `D`: Second-order moment.
  * `dD`: Derivatives of the second-order moment with respect to the unknown parameters to be estimated. 

  * `eps`: Machine epsilon

<a target='_blank' href='https://github.com/QuanEstimation/QuanEstimation.jl/blob/eb5acd75722bf76585e543db1d609b610c38a0ee/src/objective/AsymptoticBound/CramerRao.jl#L558-L577' class='documenter-source'>source</a><br>

