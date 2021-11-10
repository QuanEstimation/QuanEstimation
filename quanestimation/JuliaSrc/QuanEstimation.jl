module QuanEstimation

# using LinearAlgebra: similar
using LinearAlgebra
using Zygote
using DifferentialEquations
using JLD
using Random
using SharedArrays
using Base.Threads
using SparseArrays
using DelimitedFiles
using StatsBase
using ReinforcementLearning

include("AsymptoticBound/CramerRao.jl")
include("Common/common.jl")
include("Control/GRAPE.jl")
include("Control/DDPG.jl")
include("Control/PSO.jl")
include("Control/DiffEvo.jl")
include("Control/common.jl")
include("Dynamics/dynamcs.jl")
include("StateOptimization/common.jl")
include("StateOptimization/StateOpt_DE.jl")
include("StateOptimization/StateOpt_NM.jl")
include("StateOptimization/StateOpt_PSO.jl")
include("StateOptimization/StateOpt_AD.jl")
# include("QuanResources/")

export QFI, QFIM, CFI, CFIM
export suN_generator, expm
######## control optimization ########
export Gradient, auto_GRAPE_QFIM, auto_GRAPE_CFIM, GRAPE_QFIM, GRAPE_CFIM
export DiffEvo, DE_QFIM, DE_CFIM
export PSO, PSO_QFIM, PSO_CFIM
######## state optimization ########
export TimeIndepend_noiseless, TimeIndepend_noise
export AD_QFIM, AD_CFIM
export DE_QFIM, DE_CFIM
export PSO_QFIM, PSO_CFIM
export NM_QFIM, NM_CFIM

end
