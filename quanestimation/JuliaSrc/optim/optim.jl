abstract type AbstractOpt end

abstract type AbstractMeasurementType end
abstract type Projection <: AbstractMeasurementType end
abstract type LinearComb <: AbstractMeasurementType end
abstract type Rotation <: AbstractMeasurementType end

abstract type Opt <: AbstractOpt end

Base.@kwdef mutable struct ControlOpt <: Opt
	ctrl::Union{AbstractVector, Missing} = missing
	ctrl_bound::AbstractVector = [-Inf, Inf]
end

Copt = ControlOpt
ControlOpt(ctrl::Matrix{R}, ctrl_bound::AbstractVector) where R<:Number = ControlOpt([c[:] for c in eachrow(ctrl)], ctrl_bound)  

Base.@kwdef mutable struct StateOpt <: Opt
	ψ₀::Union{AbstractVector, Missing} = missing
end

Sopt = StateOpt

abstract type AbstractMopt <: Opt end

Base.@kwdef mutable struct Mopt_Projection <: AbstractMopt
	C::Union{AbstractVector, Missing} = missing
end
Base.@kwdef mutable struct Mopt_LinearComb <: AbstractMopt
	B::Union{AbstractVector, Missing} = missing
	POVM_basis::Union{AbstractVector, Missing} = missing
	M_num::Int = 1
end
Base.@kwdef mutable struct Mopt_Rotation <: AbstractMopt
	s::Union{AbstractVector, Missing} = missing
	POVM_basis::Union{AbstractVector, Missing} = missing
	Lambda::Union{AbstractVector, Missing} = missing
end

function MeasurementOpt(;method=:Projection,kwargs...)
	if method==:Projection
		return Mopt_Projection(;kwargs...)
	elseif method==:LinearCombination
		return Mopt_LinearComb(;kwargs...)
	elseif method==:Rotation
		return Mopt_Rotation(;kwargs...)
	end
end
Mopt = MeasurementOpt

abstract type CompOpt <: Opt end

Base.@kwdef mutable struct StateControlOpt <: CompOpt
	ψ₀::Union{AbstractVector, Missing} = missing
	ctrl::Union{AbstractVector, Missing} = missing
	ctrl_bound::AbstractVector = [-Inf, Inf]
end

SCopt = StateControlOpt

Base.@kwdef mutable struct ControlMeasurementOpt <: CompOpt
	ctrl::Union{AbstractVector, Missing} = missing
	C::Union{AbstractVector, Missing} = missing
	ctrl_bound::AbstractVector = [-Inf, Inf]
end 

CMopt = ControlMeasurementOpt

Base.@kwdef mutable struct StateMeasurementOpt <: CompOpt 
	ψ₀::Union{AbstractVector, Missing} = missing
	C::Union{AbstractVector, Missing} = missing
end

SMopt = StateMeasurementOpt

Base.@kwdef mutable struct StateControlMeasurementOpt <: CompOpt
	ctrl::Union{AbstractVector, Missing} = missing
	ψ₀::Union{AbstractVector, Missing} = missing
	C::Union{AbstractVector, Missing} = missing
	ctrl_bound::AbstractVector = [-Inf, Inf]
end

SCMopt = StateControlMeasurementOpt

opt_target(::ControlOpt) = :Copt
opt_target(::StateOpt) = :Sopt
opt_target(::Mopt_Projection) = :Mopt
opt_target(::Mopt_LinearComb) = :Mopt_input
opt_target(::Mopt_Rotation) = :Mopt_input
opt_target(::CompOpt) = :CompOpt
opt_target(::StateControlOpt) = :SCopt
opt_target(::ControlMeasurementOpt) = :CMopt
opt_target(::StateMeasurementOpt) = :SMopt
opt_target(::StateControlMeasurementOpt) = :SCMopt

result(opt::ControlOpt) = [opt.ctrl]
result(opt::StateOpt) = [opt.ψ₀]
result(opt::Mopt_Projection) = [opt.C]
result(opt::Mopt_LinearComb) = [opt.B, opt.POVM_basis, opt.M_num]
result(opt::Mopt_Rotation) = [opt.s]
result(opt::StateControlOpt) = [opt.ψ₀, opt.ctrl]
result(opt::ControlMeasurementOpt) = [opt.ctrl, opt.C]
result(opt::StateMeasurementOpt) = [opt.ψ₀, opt.C]
result(opt::StateControlMeasurementOpt) = [opt.ψ₀, opt.ctrl, opt.C]

#with reward
result(opt, ::Type{Val{:save_reward}}) = [result(opt)..., [0.0]]

const res_file_name = Dict(
    :Copt => ["controls.csv"],
    :Sopt => ["states.csv"],
    :Mopt => ["measurements.csv"],
    :Mopt_input => ["measurements.csv"],
    :SCopt => ["states.csv", "controls.csv"],
    :CMopt => ["controls.csv", "measurements.csv"],
    :SMopt => ["states.csv", "measurements.csv"],
    :SCMopt => ["states.csv", "controls.csv", "measurements.csv"],
)

res_file(opt::AbstractOpt) = res_file_name[opt_target(opt)]
