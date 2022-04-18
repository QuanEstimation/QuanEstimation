using Zygote: @adjoint
const σ_x = [0.0 1.0; 1.0 0.0im]
const σ_y = [0.0 -1.0im; 1.0im 0.0]
const σ_z = [1.0 0.0im; 0.0 -1.0]

############## logarrithmic derivative ###############
@doc raw"""

	SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) Complex}

Calculate the symmetric logarrithmic derivatives (SLDs).The SLD operator $L_a$ is defined as``\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)``, where ρ is the parameterized density matrix.

- `ρ`: Density matrix.
- `dρ`:  Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the SLD operator. Options can be:
	- "original" (default) -- The SLD matrix will be written in terms of the same basis as the input density matrix (ρ).
	- "eigen" -- The SLD matrix will be written in terms of the eigenbasis of the input ρ.
- `eps`: Machine epsilon
"""
function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}
    (x -> SLD(ρ, x; rep=rep, eps = eps)).(dρ)
end

"""

	SLD(ρ::Matrix{T},dρ::Matrix{T};rep = "original",eps = GLOBAL_EPS,) where {T<:Complex}

When applied to the case of single parameter.
"""
function SLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}

    dim = size(ρ)[1]
    SLD = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(ρ)
    val = val |> real
    SLD_eig = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
             if abs(val[fi] + val[fj]) > eps
                SLD_eig[fi, fj] = 2 * (vec[:, fi]' * dρ * vec[:, fj]) / (val[fi] + val[fj])
            end
        end
    end
    SLD_eig[findall(SLD_eig == Inf)] .= 0.0

    if rep == "original"
        SLD = vec * (SLD_eig * vec')
    elseif rep == "eigen"
        SLD = SLD_eig
	else
		throw(ArgumentError("the rep should be chosen between"))
    end
    SLD
end

@adjoint function SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    L = SLD(ρ, dρ; eps = eps)
    SLD_pullback = L̄ -> (Ḡ -> (-Ḡ * L - L * Ḡ, 2 * Ḡ))(SLD((ρ) |> Array, L̄ / 2))
    L, SLD_pullback
end

function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    (x -> SLD(ρ, x; eps = eps)).(dρ)
end

function SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), rtol = eps) * vec(∂ρ_∂x) |>
    vec2mat
end

function SLD_liouville(ρ::Vector{T}, ∂ρ_∂x::Vector{T}; eps = GLOBAL_EPS) where {T<:Complex}
    SLD_liouville(ρ |> vec2mat, ∂ρ_∂x |> vec2mat; eps = eps)
end

function SLD_liouville(
    ρ::Matrix{T},
    ∂ρ_∂x::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}

    (x -> SLD_liouville(ρ, x; eps = eps)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |>
    vec2mat
end

@doc raw"""

RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}

Calculate the right logarrithmic derivatives (RLDs). The RLD operator is defined as ``\partial_{a}\rho=\rho \mathcal{R}_a``, where ρ is the parameterized density matrix.

- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the RLD operator. Options can be:
	- "original" (default) -- The RLD matrix will be written in terms of the same basis as the input density matrix (ρ).
	- "eigen" -- The RLD matrix will be written in terms of the eigenbasis of the input ρ.
- `eps`: Machine epsilon

"""
function RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}
(x -> RLD(ρ, x; rep=rep, eps = eps)).(dρ)
end

"""

	RLD(ρ::Matrix{T},dρ::Matrix{T};rep = "original",eps = GLOBAL_EPS,) where {T<:Complex}

When applied to the case of single parameter.
"""
function RLD(
	ρ::Matrix{T},
	dρ::Matrix{T};
	rep = "original",
	eps = GLOBAL_EPS,
) where {T<:Complex}

dim = size(ρ)[1]
    RLD = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(ρ)
    val = val |> real
    RLD_eig = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
            if abs(val[fi]) > eps
                RLD_eig[fi, fj] = (vec[:, fi]' * dρ * vec[:, fj]) / val[fi]
            end
        end
    end
    RLD_eig[findall(RLD_eig == Inf)] .= 0.0

    if rep == "original"
        RLD = vec * (RLD_eig * vec')
    elseif rep == "eigen"
        RLD = RLD_eig
    end
    RLD
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    pinv(ρ, rtol = eps) * dρ
end

function RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    (x -> RLD(ρ, x; eps = eps)).(dρ)
end

@doc raw"""

LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}

Calculate the left logarrithmic derivatives (LLDs). The LLD operator is defined as ``\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho``, where ρ is the parameterized density matrix.
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the LLD operator. Options can be:
	- "original" (default) -- The RLD matrix will be written in terms of the same basis as the input density matrix (ρ).
	- "eigen" -- The RLD matrix will be written in terms of the eigenbasis of the input ρ.
- `eps`: Machine epsilon

"""
function LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep = "original", eps = GLOBAL_EPS) where {T<:Complex}
    (x -> LLD(ρ, x; rep=rep, eps = eps)).(dρ)
end

"""

LLD(ρ::Matrix{T},dρ::Matrix{T};rep = "original",eps = GLOBAL_EPS,) where {T<:Complex}

When applied to the case of single parameter.
"""
function LLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}

    dim = size(ρ)[1]
    LLD = Matrix{ComplexF64}(undef, dim, dim)

    val, vec = eigen(ρ)
    val = val |> real
    LLD_eig = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
            if abs(val[fj]) > eps
                LLD_eig[fj, fi] = ((vec[:, fi]' * dρ * vec[:, fj]) / val[fj]) |> conj()
            end
        end
    end
    LLD_eig[findall(LLD_eig == Inf)] .= 0.0

    if rep == "original"
        LLD = vec * (LLD_eig * vec')
    elseif rep == "eigen"
        LLD = LLD_eig
    end
    LLD
end

function LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    (dρ * pinv(ρ, rtol = eps))
end

function LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    (x -> LLD(ρ, x; eps = eps)).(dρ)
end

#========================================================#
####################### calculate QFI ####################
function QFIM_SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    SLD_tp = SLD(ρ, dρ; eps = eps)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFIM_RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    RLD_tp = RLD(ρ, dρ; eps = eps)
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end

function QFIM_LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    LLD_tp = LLD(ρ, dρ; eps = eps)
    F = tr(ρ * LLD_tp * LLD_tp')
    F |> real
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    SLD = 2 * ∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#==========================================================#
####################### calculate QFIM #####################
function QFIM_SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = SLD(ρ, dρ; eps = eps)
    (
        [0.5 * ρ] .*
        (kron(LD_tp, reshape(LD_tp, 1, p_num)) + kron(reshape(LD_tp, 1, p_num), LD_tp))
    ) .|> tr .|> real
end

function QFIM_RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = RLD(ρ, dρ; eps = eps)
    LD_dag = [LD_tp[i]' for i = 1:p_num]
    ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr .|> real
end

function QFIM_LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = LLD(ρ, dρ; eps = eps)
    LD_dag = [LD_tp[i]' for i = 1:p_num]
    ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr .|> real
end

function QFIM_liouville(ρ, dρ)
    p_num = length(dρ)
    LD_tp = SLD_liouville(ρ, dρ)
    (
        [0.5 * ρ] .*
        (kron(LD_tp, reshape(LD_tp, 1, p_num)) + kron(reshape(LD_tp, 1, p_num), LD_tp))
    ) .|> tr .|> real
end

function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    SLD = [2 * ∂ρ_∂x[i] for i = 1:p_num]
    (
        [0.5 * ρ] .*
        (kron(SLD, reshape(SLD, 1, p_num)) + kron(reshape(SLD, 1, p_num), SLD))
    ) .|>
    tr .|>
    real
end

#======================================================#
#################### calculate CFIM ####################
@doc raw"""

	CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps = GLOBAL_EPS) where {T<:Complex}

Calculate the classical Fisher information matrix (CFIM). 

- `ρ`: Density matrix.
- `dρ`:  Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `eps`: Machine epsilon.
"""
function CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps = GLOBAL_EPS) where {T<:Complex}
    m_num = length(M)
    p_num = length(dρ)
    [
        real(tr(ρ * M[i])) < eps ? zeros(ComplexF64, p_num, p_num) :
        (kron(tr.(dρ .* [M[i]]), reshape(tr.(dρ .* [M[i]]), 1, p_num)) / tr(ρ * M[i])) for
        i = 1:m_num
    ] |>
    sum .|>
    real
end

"""

	CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps = GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter. Calculate the classical Fisher information(CFI). 
"""
function CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps = GLOBAL_EPS) where {T<:Complex}
    m_num = length(M)
    F = 0.0
    for i = 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.0
        if p > eps
            cadd = (dp * dp) / p
        end
        F += cadd
    end
    real(F)
end 

"""

	CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}

When the set of POVM is not given. Calculate the CFIM with SIC-POVM. The SIC-POVM is generated from the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from the [website](http://www.physics.umb.edu/Research/QBism/solutions.html).
"""
function CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    M = SIC(size(ρ)[1])
    m_num = length(M)
    p_num = length(dρ)
    [
        real(tr(ρ * M[i])) < eps ? zeros(ComplexF64, p_num, p_num) :
        (kron(tr.(dρ .* [M[i]]), reshape(tr.(dρ .* [M[i]]), 1, p_num)) / tr(ρ * M[i])) for
        i = 1:m_num
    ] |>
    sum .|>
    real
end

"""

	CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter and the set of POVM is not given. Calculate the CFI with SIC-POVM. 
"""
function CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    M = SIC(size(ρ)[1])
    m_num = length(M)
    F = 0.0
    for i = 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.0
        if p > eps
            cadd = (dp * dp) / p
        end
        F += cadd
    end
    real(F)
end

"""

"""
function QFIM(
    ρ::Matrix{T},
    dρ::Matrix{T};
    LDtype = :SLD,
    eps = GLOBAL_EPS,
) where {T<:Complex}

    eval(Symbol("QFIM_" * string(LDtype)))(ρ, dρ; eps = eps)
end

## QFIM 
function QFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    LDtype = :SLD,
    eps = GLOBAL_EPS,
) where {T<:Complex}

    eval(Symbol("QFIM_" * string(LDtype)))(ρ, dρ; eps = eps)
end


QFIM(sym::Symbol, args...; kwargs...) = QFIM(Val{sym}, args...; kwargs...)

## QFI with exportLD
function QFIM(
    ::Val{:exportLD},
    ρ::Matrix{T},
    dρ::Matrix{T};
    LDtype = :SLD,
    eps = GLOBAL_EPS,
) where {T<:Complex}

    F = QFIM(ρ, dρ; LDtype = LDtype, eps = eps)
    LD = eval(LDtype)(ρ, dρ; eps = eps)
    return F, LD
end

## QFIM with exportLD
function QFIM(
    ::Val{:exportLD},
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    LDtype = :SLD,
    eps = GLOBAL_EPS,
) where {T<:Complex}

    F = QFIM(ρ, dρ; LDtype = LDtype, eps = eps)
    LD = eval(LDtype)(ρ, dρ; eps = eps)
    return F, LD
end


"""

	QFIM_Bloch(r, dr; eps = 1e-8)

Calculate the SLD based quantum Fisher information (QFI) or quantum  
Fisher information matrix (QFIM) in Bloch representation.

- `r`: Parameterized Bloch vector.

- `dr`: Derivative(s) of the Bloch vector with respect to the unknown parameters to be 
	estimated. For example, dr[1] is the derivative vector with respect to the first 
	parameter.

- `eps`: Machine epsilon.

"""
## TODO: 👇 check type stability
function QFIM_Bloch(r, dr; eps = 1e-8)
    para_num = length(dr)
    QFIM_res = zeros(para_num, para_num)

    dim = Int(sqrt(length(r) + 1))
    Lambda = suN_generator(dim)
    if dim == 2
        r_norm = norm(r)^2
        if abs(r_norm - 1.0) < eps
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    QFIM_res[para_i, para_j] = dr[para_i]' * dr[para_j]
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        else
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    QFIM_res[para_i, para_j] =
                        dr[para_i]' * dr[para_j] +
                        (r' * dr[para_i]) * (r' * dr[para_j]) / (1 - r_norm)
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        end
    else
        rho = (Matrix(I, dim, dim) + sqrt(dim * (dim - 1) / 2) * r' * Lambda) / dim
        G = zeros(ComplexF64, dim^2 - 1, dim^2 - 1)
        for row_i = 1:dim^2-1
            for col_j = row_i:dim^2-1
                anti_commu = Lambda[row_i] * Lambda[col_j] + Lambda[col_j] * Lambda[row_i]
                G[row_i, col_j] = 0.5 * tr(rho * anti_commu)
                G[col_j, row_i] = G[row_i, col_j]
            end
        end

        mat_tp = G * dim / (2 * (dim - 1)) - r * r'
        mat_inv = pinv(mat_tp)

        for para_m = 1:para_num
            for para_n = para_m:para_num
                # println(dr[para_n]*mat_inv*dr[para_m]')
                QFIM_res[para_m, para_n] = dr[para_n]' * mat_inv * dr[para_m]
                QFIM_res[para_n, para_m] = QFIM_res[para_m, para_n]
            end
        end
    end
    if para_num == 1
        return QFIM_res[1, 1]
    else
        return QFIM_res
    end
end

#======================================================#
################# Gaussian States QFIM #################
function Williamson_form(A::AbstractMatrix)
    n = size(A)[1] // 2 |> Int
    J = zeros(n, n) |> x -> [x one(x); -one(x) x]
    A_sqrt = sqrt(A)
    B = A_sqrt * J * A_sqrt
    P = one(A) |> x -> [x[:, 1:2:2n-1] x[:, 2:2:2n]]
    t, Q, vals = schur(B)
    c = vals[1:2:2n-1] .|> imag
    D = c |> diagm |> x -> x^(-0.5)
    S =
        (J * A_sqrt * Q * P * [zeros(n, n) -D; D zeros(n, n)] |> transpose |> inv) *
        transpose(P)
    return S, c
end

# const a_Gauss = [im*σ_y,σ_z,σ_x|>one, σ_x]

function A_Gauss(m::Int)
    e = bases(m)
    s = e .* e'
    a_Gauss .|> x -> [kron(s, x) / sqrt(2) for s in s]
end

function G_Gauss(S::M, dC::VM, c::V) where {M<:AbstractMatrix,V,VM<:AbstractVector}
    para_num = length(dC)
    m = size(S)[1] // 2 |> Int
    As = A_Gauss(m)
    gs = [
        [[inv(S) * ∂ₓC * inv(transpose(S)) * a' |> tr for a in A] for A in As] for ∂ₓC in dC
    ]
    #[[inv(S)*∂ₓC*inv(transpose(S))*As[l][j,k]|>tr for j in 1:m, k in 1:m] for l in 1:4]
    G = [zero(S) for _ = 1:para_num]

    for i = 1:para_num
        for j = 1:m
            for k = 1:m
                for l = 1:4
                    G[i] +=
                        gs[i][l][j, k] / (4 * c[j]c[k] + (-1)^l) *
                        inv(transpose(S)) *
                        As[l][j, k] *
                        inv(S)
                end
            end
        end
    end
    G
end

"""

	QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}

Calculate the SLD based quantum Fisher information matrix (QFIM) with gaussian states.

- `R̄` : First-order moment.

- `dR̄`: Derivatives of the first-order moment with respect to the unknown parameters to be 
estimated. For example, dR[1] is the derivative vector on the first 
parameter.

- `D`: Second-order moment.

- `dD`: Derivatives of the second-order moment with respect to the unknown parameters to be 
estimated. 

- `eps`: Machine epsilon

"""
function QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}
    para_num = length(dR̄)
    quad_num = length(R̄)
    C = [(D[i, j] + D[j, i]) / 2 - R̄[i]R̄[j] for i = 1:quad_num, j = 1:quad_num]
    dC = [
        [
            (dD[k][i, j] + dD[k][j, i]) / 2 - dR̄[k][i]R̄[j] - R̄[i]dR̄[k][j] for
            i = 1:quad_num, j = 1:quad_num
        ] for k = 1:para_num
    ]

    S, cs = Williamson_form(C)
    Gs = G_Gauss(S, dC, cs)
    F = [
        tr(Gs[i] * dC[j]) + transpose(dR̄[i]) * inv(C) * dR̄[j] for i = 1:para_num,
        j = 1:para_num
    ]

    F |> real
end
