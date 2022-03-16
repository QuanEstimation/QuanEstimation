abstract type AbstractObj end

abstract type quantum end
abstract type classical end

abstract type AbstractParaType end
abstract type single_para <: AbstractParaType end
abstract type multi_para <: AbstractParaType end

include("AsymptoticBound/AsymptoticBound.jl")
