abstract type AbstractLDtype end

abstract type SLD <: AbstractLDtype end
abstract type RLD <: AbstractLDtype end
abstract type LLD <: AbstractLDtype end

struct QFIM_obj{P,D} <: AbstractObj
    W::Union{AbstractMatrix, Missing}
    eps::Number
end

struct CFIM_obj{P} <: AbstractObj
    M::Union{AbstractVecOrMat, Missing}
    W::Union{AbstractMatrix, Missing}
    eps::Number
end

struct HCRB_obj{P} <: AbstractObj
    W::Union{AbstractMatrix, Missing}
    eps::Number
end

@doc raw"""

    QFIM_obj(;W=missing, eps=GLOBAL_EPS, LDtype::Symbol=:SLD)

Choose QFI [``\mathrm{Tr}(WF^{-1})``] as the objective function with ``W`` the weight matrix and ``F`` the QFIM.
- `W`: Weight matrix.
- `eps`: Machine epsilon.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
"""
QFIM_obj(;W=missing, eps=GLOBAL_EPS, para_type::Symbol=:single_para, LDtype::Symbol=:SLD) = QFIM_obj{eval.([para_type, LDtype])...}(W, eps)

@doc raw"""

    CFIM_obj(;M=missing, W=missing, eps=GLOBAL_EPS)

Choose CFI [``\mathrm{Tr}(WI^{-1})``] as the objective function with ``W`` the weight matrix and ``I`` the CFIM.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
CFIM_obj(;M=missing, W=missing, eps=GLOBAL_EPS, para_type::Symbol=:single_para) = CFIM_obj{eval(para_type)}(M, W, eps)

@doc raw"""

    HCRB_obj(;W=missing, eps=GLOBAL_EPS)

Choose HCRB as the objective function. 
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
HCRB_obj(;W=missing, eps=GLOBAL_EPS, para_type::Symbol=:single_para) = HCRB_obj{eval(para_type)}(W, eps)

QFIM_obj(W, eps, para_type::Symbol, LDtype::Symbol) = QFIM_obj{eval.([para_type, LDtype])...}(W, eps)
CFIM_obj(M, W, eps, para_type::Symbol) = CFIM_obj{eval(para_type)}(M, W, eps)
HCRB_obj(W, eps, para_type::Symbol) = HCRB_obj{eval(para_type)}(W, eps)

QFIM_obj(W::AbstractMatrix, eps::Number, para_type::String, LDtype::String) = QFIM_obj(W, eps, Symbol.([para_type, LDtype])...)
CFIM_obj(M::AbstractVecOrMat, W::AbstractMatrix, eps::Number, para_type::String) = CFIM_obj(M, W, eps, Symbol(para_type))
HCRB_obj(W::AbstractMatrix, eps::Number, para_type::String) = HCRB_obj(W, eps, Symbol(para_type))

obj_type(::QFIM_obj) = :QFIM
obj_type(::CFIM_obj) = :CFIM
obj_type(::HCRB_obj) = :HCRB

para_type(::QFIM_obj{single_para,D}) where {D} = :single_para
para_type(::QFIM_obj{multi_para,D}) where {D} = :multi_para
para_type(::CFIM_obj{single_para}) = :single_para
para_type(::CFIM_obj{multi_para}) = :multi_para
para_type(::HCRB_obj{single_para}) = :single_para
para_type(::HCRB_obj{multi_para}) = :multi_para

LDtype(::QFIM_obj{P,SLD}) where {P} = :SLD
LDtype(::QFIM_obj{P,RLD}) where {P} = :RLD
LDtype(::QFIM_obj{P,LLD}) where {P} = :LLD

QFIM_obj(opt::CFIM_obj{P}) where P = QFIM_obj{P, SLD}(opt.W, opt.eps)
QFIM_obj(opt::CFIM_obj{P}, LDtype::Symbol) where P = QFIM_obj{P, eval(LDtype)}(opt.W, opt.eps)

const obj_idx = Dict(
    :QFIM => QFIM_obj,
    :CFIM => CFIM_obj,
    :HCRB => HCRB_obj
)

function set_M(obj::CFIM_obj{P}, M::AbstractVector) where P
    CFIM_obj{P}(M, obj.W, obj.eps)
end

function objective(obj::QFIM_obj{single_para,SLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{single_para,SLD}, ρ, dρ)
    (; W, eps) = obj
    f = W[1] * QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,SLD}, ρ, dρ)
    (; W, eps) = obj
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps))) 
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,RLD}, ρ, dρ)
    (; W, eps) = obj
    f = W[1] * QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,RLD}, ρ, dρ)
    (; W, eps) = obj
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,LLD}, ρ, dρ)
    (; W, eps) = obj
    f = W[1] * QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,LLD}, ρ, dρ)
    (; W, eps) = obj
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{multi_para,SLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,RLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,RLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,LLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,LLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,SLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,SLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,RLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,RLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_obj{single_para,LLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_obj{multi_para,LLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::CFIM_obj{single_para}, ρ, dρ)
    (; M, W, eps) = obj
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(obj::CFIM_obj{multi_para}, ρ, dρ)
    (; M, W, eps) = obj
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::CFIM_obj{single_para}, dynamics::Lindblad)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(obj::CFIM_obj{multi_para}, dynamics::Lindblad)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::CFIM_obj{single_para}, dynamics::Kraus)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(obj::CFIM_obj{multi_para}, dynamics::Kraus)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::HCRB_obj{multi_para}, ρ, dρ)
    (; W, eps) = obj
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

function objective(obj::HCRB_obj{multi_para}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

function objective(obj::HCRB_obj{multi_para}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

#### objective function for linear combination in Mopt ####
function objective(opt::Mopt_LinearComb, obj::CFIM_obj{single_para}, dynamics::Lindblad)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_LinearComb, obj::CFIM_obj{multi_para}, dynamics::Lindblad)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(opt::Mopt_LinearComb, obj::CFIM_obj{single_para}, dynamics::Kraus)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_LinearComb, obj::CFIM_obj{multi_para}, dynamics::Kraus)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

#### objective function for rotation in Mopt ####
function objective(opt::Mopt_Rotation, obj::CFIM_obj{single_para}, dynamics::Lindblad)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_Rotation, obj::CFIM_obj{multi_para}, dynamics::Lindblad)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(opt::Mopt_Rotation, obj::CFIM_obj{single_para}, dynamics::Kraus)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_Rotation, obj::CFIM_obj{multi_para}, dynamics::Kraus)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

#####
# function objective(::Type{Val{:expm}}, obj, dynamics)
#     temp = []
#     (; tspan, ctrl) = dynamics.data
#     for i = 1:length( ctrl)
#         dynamics_copy = set_ctrl(dynamics, [ctrl[1:i] for ctrl in ctrl])
#         dynamics_copy.data.tspan = tspan[1:i+1]
#         append!(temp, [objective(obj, dynamics_copy)])
#     end
#     temp
# end  # function objective

include("CramerRao.jl")
include("Holevo.jl")
include("AsymptoticBoundWrapper.jl")
