abstract type AbstractLDtype end

abstract type SLD <: AbstractLDtype end
abstract type RLD <: AbstractLDtype end
abstract type LLD <: AbstractLDtype end

Base.@kwdef struct QFIM_Obj{P,D} <: AbstractObj
    W::Union{AbstractMatrix, Missing}
    eps::Number = GLOBAL_EPS
end

Base.@kwdef struct CFIM_Obj{P} <: AbstractObj
    M::Union{AbstractVecOrMat, Missing}
    W::Union{AbstractMatrix, Missing}
    eps::Number = GLOBAL_EPS
end

Base.@kwdef struct HCRB_Obj{P} <: AbstractObj
    W::Union{AbstractMatrix, Missing}
    eps::Number = GLOBAL_EPS
end

QFIM_Obj(;W=missing, eps=GLOBAL_EPS, para_type::Symbol=:single_para, LD_type::Symbol=:SLD) = QFIM_Obj{eval.([para_type, LD_type])...}(W, eps)
CFIM_Obj(;M=missing, W=missing, eps=GLOBAL_EPS, para_type::Symbol=:single_para) = CFIM_Obj{eval(para_type)}(M, W, eps)
HCRB_Obj(;W=missing, eps=GLOBAL_EPS, para_type::Symbol=:single_para) = HCRB_Obj{eval(para_type)}(W, eps)

QFIM_Obj(W::AbstractMatrix, eps::Number, para_type::String, LD_type::String) = QFIM_Obj(W, eps, Symbol.([para_type, LD_type])...)
CFIM_Obj(M::AbstractVecOrMat, W::AbstractMatrix, eps::Number, para_type::String) = CFIM_Obj(M, W, eps, Symbol(para_type))
HCRB_Obj(W::AbstractMatrix, eps::Number, para_type::String) = HCRB_Obj(W, eps, Symbol(para_type))

obj_type(::QFIM_Obj) = :QFIM
obj_type(::CFIM_Obj) = :CFIM
obj_type(::HCRB_Obj) = :HCRB

para_type(::QFIM_Obj{single_para,D}) where {D} = :single_para
para_type(::QFIM_Obj{multi_para,D}) where {D} = :multi_para
para_type(::CFIM_Obj{single_para}) = :single_para
para_type(::CFIM_Obj{multi_para}) = :multi_para
para_type(::HCRB_Obj{single_para}) = :single_para
para_type(::HCRB_Obj{multi_para}) = :multi_para

LD_type(::QFIM_Obj{P,SLD}) where {P} = :SLD
LD_type(::QFIM_Obj{P,RLD}) where {P} = :RLD
LD_type(::QFIM_Obj{P,LLD}) where {P} = :LLD

QFIM_Obj(opt::CFIM_Obj{P}) where P = QFIM_Obj{P, SLD}(opt.W, opt.eps)
QFIM_Obj(opt::CFIM_Obj{P}, LDtype::Symbol) where P = QFIM_Obj{P, eval(LDtype)}(opt.W, opt.eps)

const obj_idx = Dict(
    :QFIM => QFIM_Obj,
    :CFIM => CFIM_Obj,
    :HCRB => HCRB_Obj
)

function set_M(obj::CFIM_Obj{P}, M::AbstractVector) where P
    CFIM_Obj{P}(M, obj.W, obj.eps)
end

function objective(obj::QFIM_Obj{single_para,SLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{single_para,SLD}, ρ, dρ)
    (; W, eps) = obj
    f = W[1] * QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,SLD}, ρ, dρ)
    (; W, eps) = obj
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps))) 
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,RLD}, ρ, dρ)
    (; W, eps) = obj
    f = W[1] * QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,RLD}, ρ, dρ)
    (; W, eps) = obj
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,LLD}, ρ, dρ)
    (; W, eps) = obj
    f = W[1] * QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,LLD}, ρ, dρ)
    (; W, eps) = obj
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{multi_para,SLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,RLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,RLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,LLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,LLD}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,SLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,SLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,RLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,RLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::QFIM_Obj{single_para,LLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

function objective(obj::QFIM_Obj{multi_para,LLD}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps))) |> real
    return f, 1.0 / f
end

function objective(obj::CFIM_Obj{single_para}, ρ, dρ)
    (; M, W, eps) = obj
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(obj::CFIM_Obj{multi_para}, ρ, dρ)
    (; M, W, eps) = obj
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::CFIM_Obj{single_para}, dynamics::Lindblad)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(obj::CFIM_Obj{multi_para}, dynamics::Lindblad)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::CFIM_Obj{single_para}, dynamics::Kraus)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(obj::CFIM_Obj{multi_para}, dynamics::Kraus)
    (; M, W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(obj::HCRB_Obj{multi_para}, ρ, dρ)
    (; W, eps) = obj
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

function objective(obj::HCRB_Obj{multi_para}, dynamics::Lindblad)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

function objective(obj::HCRB_Obj{multi_para}, dynamics::Kraus)
    (; W, eps) = obj
    ρ, dρ = evolve(dynamics)
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

#### objective function for linear combination in Mopt ####
function objective(opt::Mopt_LinearComb, obj::CFIM_Obj{single_para}, dynamics::Lindblad)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_LinearComb, obj::CFIM_Obj{multi_para}, dynamics::Lindblad)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(opt::Mopt_LinearComb, obj::CFIM_Obj{single_para}, dynamics::Kraus)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_LinearComb, obj::CFIM_Obj{multi_para}, dynamics::Kraus)
    (; W, eps) = obj
    M = [sum([opt.B[i][j]*opt.POVM_basis[j] for j in 1:length(opt.POVM_basis)]) for i in 1:opt.M_num]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

#### objective function for rotation in Mopt ####
function objective(opt::Mopt_Rotation, obj::CFIM_Obj{single_para}, dynamics::Lindblad)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_Rotation, obj::CFIM_Obj{multi_para}, dynamics::Lindblad)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

function objective(opt::Mopt_Rotation, obj::CFIM_Obj{single_para}, dynamics::Kraus)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U*opt.POVM_basis[i]*U' for i in 1:length(opt.POVM_basis)]
    ρ, dρ = evolve(dynamics)
    f = W[1] * CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

function objective(opt::Mopt_Rotation, obj::CFIM_Obj{multi_para}, dynamics::Kraus)
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
