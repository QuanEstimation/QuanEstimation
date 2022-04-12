using Random
using StableRNGs
include("../src/QuanEstimation.jl")

# Kraus operators for the amplitude damping channel
γ = 0.1
K1 = [1. 0; 0 sqrt(1-γ)]
K2 = [0. sqrt(γ); 0 0]
K = [K1, K2]

dK1 = [1. 0; 0 -sqrt(1-γ)/2]
dK2 = [0. sqrt(γ); 0 0]
dK = [[dK1], [dK2]]

opt = QuanEstimation.SMopt()
dynamics = QuanEstimation.Kraus(opt, K, dK)

# set objective
obj = QuanEstimation.CFIM_Obj()

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)
