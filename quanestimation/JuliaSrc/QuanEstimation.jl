module QuanEstimation

using LinearAlgebra
using Zygote
# using DifferentialEquations
using JLD
using Random
using SharedArrays
using Base.Threads
using SparseArrays
using DelimitedFiles
using StatsBase
using ReinforcementLearning
using Convex
using SCS
using BoundaryValueDiffEq
using Trapz
using Interpolations

include("AsymptoticBound/CramerRao.jl")
include("AsymptoticBound/Holevo.jl")
include("Bayesian/Bayes.jl")
include("Common/common.jl")
include("ControlOpt/GRAPE_Copt.jl")
include("ControlOpt/DDPG_Copt.jl")
include("ControlOpt/PSO_Copt.jl")
include("ControlOpt/DE_Copt.jl")
include("ControlOpt/common.jl")
include("Dynamics/dynamcs.jl")
include("StateOpt/common.jl")
include("StateOpt/DE_Sopt.jl")
include("StateOpt/NM_Sopt.jl")
include("StateOpt/PSO_Sopt.jl")
include("StateOpt/AD_Sopt.jl")
include("StateOpt/DDPG_Sopt.jl")
include("MeasurementOpt/common.jl")
include("MeasurementOpt/AD_Mopt.jl")
include("MeasurementOpt/PSO_Mopt.jl")
include("MeasurementOpt/DE_Mopt.jl")

# include("QuanResources/")
export QFI, QFIM, CFI, CFIM
export suN_generator, expm
######## control optimization ########
export Gradient, auto_GRAPE_QFIM, auto_GRAPE_CFIM, GRAPE_QFIM, GRAPE_CFIM
export DiffEvo, DE_QFIM, DE_CFIM
export PSO, PSO_QFIM, PSO_CFIM
export DDPG, DDPG_QFIM, DDPG_CFIM
######## state optimization ########
export TimeIndepend_noiseless, TimeIndepend_noise
export AD_QFIM, AD_CFIM
export DE_QFIM, DE_CFIM
export PSO_QFIM, PSO_CFIM
export NM_QFIM, NM_CFIM
export DDPG_QFIM, DDPG_CFIM
end
