using Zygote: @adjoint
const σ_x = [0.0 1.0; 1.0 0.0im]
const σ_y = [0.0 -1.0im; 1.0im 0.0]
const σ_z = [1.0 0.0im; 0.0 -1.0]
const eps_default = 1e-8

############## logarrithmic derivative ###############
function SLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = eps_default,
    rep = "original",
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
    end
    SLD
end

@adjoint function SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
    L = SLD(ρ, dρ; eps = eps)
    SLD_pullback = L̄ -> (Ḡ -> (-Ḡ * L - L * Ḡ, 2 * Ḡ))(SLD((ρ) |> Array, L̄ / 2))
    L, SLD_pullback
end

function SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
    (x -> SLD(ρ, x; eps = eps)).(dρ)
end

function SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}; eps = eps_default) where {T<:Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), rtol = eps) * vec(∂ρ_∂x) |>
    vec2mat
end

function SLD_liouville(ρ::Vector{T}, ∂ρ_∂x::Vector{T}; eps = eps_default) where {T<:Complex}
    SLD_liouville(ρ |> vec2mat, ∂ρ_∂x |> vec2mat; eps = eps)
end

function SLD_liouville(
    ρ::Matrix{T},
    ∂ρ_∂x::Vector{Matrix{T}};
    eps = eps_default,
) where {T<:Complex}

    (x -> SLD_liouville(ρ, x; eps = eps)).(∂ρ_∂x)
end

function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |>
    vec2mat
end

function RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
    pinv(ρ, rtol = eps) * dρ
end

function RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
    (x -> RLD(ρ, x; eps = eps)).(dρ)
end

function LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
    (dρ * pinv(ρ, rtol = eps))'
end

function LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
    (x -> LLD(ρ, x; eps = eps)).(dρ)
end

#========================================================#
####################### calculate QFI ####################
function QFIM_SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
    SLD_tp = SLD(ρ, dρ; eps = eps)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

function QFIM_RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
    RLD_tp = RLD(ρ, dρ; eps = eps)
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end

function QFIM_LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
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
function QFIM_SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = SLD(ρ, dρ; eps = eps)
    (
        [0.5 * ρ] .*
        (kron(LD_tp, reshape(LD_tp, 1, p_num)) + kron(reshape(LD_tp, 1, p_num), LD_tp))
    ) .|> tr .|> real
end

function QFIM_RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = RLD(ρ, dρ; eps = eps)
    LD_dag = [LD_tp[i]' for i = 1:p_num]
    ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr
    # ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr .|> real
end

function QFIM_LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = LLD(ρ, dρ; eps = eps)
    LD_dag = [LD_tp[i]' for i = 1:p_num]
    ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr
    # ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr .|> real
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

function QFIM_Kraus(ρ0::Matrix{T}, K::Vector{Matrix{T}}, dK::Vector{Vector{Matrix{T}}}; LDtype=:SLD, exportLD::Bool=false, eps = eps_default) where {T<:Complex}
    dK = [[dK[i][j] for i in 1:length(K)] for j in 1:length(dK[1])]
    ρ = [K * ρ0 * K' for K in K] |> sum
    dρ = [[dK * ρ0 * K' + K * ρ0 * dK' for (K,dK) in zip(K,dK)] |> sum for dK in dK]
    return QFIM(ρ, dρ; LDtype=LDtype, exportLD=exportLD, eps=eps)
end

#======================================================#
#################### calculate CFIM ####################
function CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps = eps_default) where {T<:Complex}
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

function CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = eps_default) where {T<:Complex}
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

function CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps = eps_default) where {T<:Complex}
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

function CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps = eps_default) where {T<:Complex}
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

function FIM(p::Vector{R}, dp::Vector{R}; eps = eps_default) where {R<:Real}
    m_num = length(p)
    F = 0.0
    for i = 1:m_num
        p_tp = p[i]
        dp_tp = dp[i]
        cadd = 0.0
        if p_tp > eps
            cadd = (dp_tp * dp_tp) / p_tp
        end
        F += cadd
    end
    real(F)
end

function FIM(p::Vector{R}, dp::Vector{Vector{R}}; eps = eps_default) where {R<:Real}
    m_num = length(p)
    para_num = length(dp[1])

    FIM_res = zeros(para_num, para_num)
    for pj in 1:m_num
        p_tp = p[pj]
        Cadd = zeros(para_num, para_num)
        if p_tp > eps
            for para_i in 1:para_num
                dp_i = dp[pj][para_i]
                for para_j in para_i:para_num
                    dp_j = dp[pj][para_j]
                    Cadd[para_i,para_j] = real(dp_i * dp_j / p_tp)
                    Cadd[para_j,para_i] = real(dp_i * dp_j / p_tp)
                end
            end
            FIM_res += Cadd
        end
    end
    FIM_res
end

## QFI
function QFIM(
    ρ::Matrix{T},
    dρ::Matrix{T};
    LDtype = :SLD,
    eps = eps_default,
) where {T<:Complex}

    eval(Symbol("QFIM_" * string(LDtype)))(ρ, dρ; eps = eps)
end

## QFIM 
function QFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    LDtype = :SLD,
    eps = eps_default,
) where {T<:Complex}

    eval(Symbol("QFIM_" * string(LDtype)))(ρ, dρ; eps = eps)
end


QFIM(ρ, dρ; LDtype=LDtype, exportLD=false, eps=eps_default) = QFIM(ρ, dρ; LDtype=LDtype, eps=eps_default)

## QFI with exportLD
function QFIM(
    ρ::Matrix{T},
    dρ::Matrix{T}; 
    LDtype = :SLD,
    exportLD ::Bool= false,
    eps = eps_default,
) where {T<:Complex}
    F = eval(Symbol("QFIM_" * string(LDtype)))(ρ, dρ; eps = eps)
    if exportLD == false
        return F
    else
        LD = eval(Symbol(LDtype))(ρ, dρ; eps = eps)
        return F, LD
    end
end

## QFIM with exportLD
function QFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    LDtype = :SLD,
    exportLD ::Bool= false,
    eps = eps_default,
) where {T<:Complex}

    F = eval(Symbol("QFIM_" * string(LDtype)))(ρ, dρ; eps = eps)
    if exportLD == false
        return F
    else
        LD = eval(Symbol(LDtype))(ρ, dρ; eps = eps)
        return F, LD
    end
end


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
                    QFIM_res[para_i, para_j] = real(dr[para_i]' * dr[para_j])
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        else
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    QFIM_res[para_i, para_j] =
                        real(dr[para_i]' * dr[para_j] +
                        (r' * dr[para_i]) * (r' * dr[para_j]) / (1 - r_norm))
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
                QFIM_res[para_m, para_n] = real(dr[para_n]' * mat_inv * dr[para_m])
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

const a_Gauss = [im*σ_y, σ_z, σ_x|>one, σ_x]

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
