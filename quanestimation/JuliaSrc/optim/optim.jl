abstract type AbstractOpt end

abstract type AbstractMeasurementType end
abstract type Projection <: AbstractMeasurementType end
abstract type LinearComb <: AbstractMeasurementType end
abstract type Rotation <: AbstractMeasurementType end

abstract type Opt <: AbstractOpt end

mutable struct ControlOpt <: Opt
    ctrl::AbstractVector
    ctrl_bound::AbstractVector
end

mutable struct StateOpt <: Opt
    ψ0::AbstractVector
end

abstract type AbstractMopt <: Opt end

mutable struct Mopt_Projection <: AbstractMopt
    C::AbstractVector
end

mutable struct Mopt_LinearComb <: AbstractMopt
    B::AbstractVector
    POVM_basis::AbstractVector
    M_num::Number
end

mutable struct Mopt_Rotation <: AbstractMopt
    s::AbstractVector
    POVM_basis::AbstractVector
    Lambda::AbstractVector
end

abstract type CompOpt <: Opt end

mutable struct StateControlOpt <: CompOpt
    ψ0::AbstractVector
    ctrl::AbstractVector
end

mutable struct ControlMeasurementOpt <: CompOpt
    ctrl::AbstractVector
    C::AbstractVector
end

mutable struct StateMeasurementOpt <: CompOpt
    ψ0::AbstractVector
    C::AbstractVector
end

mutable struct StateControlMeasurementOpt <: CompOpt
    ctrl::AbstractVector
    ψ0::AbstractVector
    C::AbstractVector
end

MeasurementOpt(M, mtype::Symbol = :Projection) = MeasurementOpt{eval(mtype)}(M)
opt_target(::ControlOpt) = :Copt
opt_target(::StateOpt) = :Sopt
opt_target(::Mopt_Projection) = :Mopt_proj
opt_target(::Mopt_LinearComb) = :Mopt_lc
opt_target(::Mopt_Rotation) = :Mopt_rot
opt_target(::CompOpt) = :CompOpt
opt_target(::StateControlOpt) = :SCopt
opt_target(::ControlMeasurementOpt) = :CMopt
opt_target(::StateMeasurementOpt) = :SMopt
opt_target(::StateControlMeasurementOpt) = :SCMopt

result(opt::ControlOpt) = [opt.ctrl]
result(opt::StateOpt) = [opt.ψ0]
result(opt::Mopt_Projection) = [opt.C]
result(opt::Mopt_LinearComb) = [opt.B, opt.POVM_basis, opt.M_num]
result(opt::Mopt_Rotation) = [opt.s, opt.POVM_basis, Lambda]
result(opt::StateControlOpt) = [opt.ψ0, opt.ctrl]
result(opt::ControlMeasurementOpt) = [opt.ctrl, opt.C]
result(opt::StateMeasurementOpt) = [opt.ψ0, opt.C]
result(opt::StateControlMeasurementOpt) = [opt.ψ0, opt.ctrl, opt.C]

#with reward
result(opt, ::Type{Val{:save_reward}}) = [result(opt)..., [0.0]]

const res_file_name = Dict(
    :Copt => ["controls.csv"],
    :Sopt => ["states.csv"],
    :Mopt => ["measurements.csv"],
    :SCopt => ["states.csv", "controls.csv"],
    :CMopt => ["controls.csv", "measurements.csv"],
    :SMopt => ["states.csv", "measurements.csv"],
    :SCMopt => ["states.csv", "controls.csv", "measurements.csv"],
)

res_file(opt::AbstractOpt) = res_file_name[opt_target(opt)]
