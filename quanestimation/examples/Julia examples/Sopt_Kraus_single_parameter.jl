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

opt = QuanEstimation.Sopt()
dynamics = QuanEstimation.Kraus(opt, K, dK)

# control algorithm: AD
alg = QuanEstimation.AD(Adam=true, max_episode=50, ϵ=0.01, beta1=0.90, beta2=0.99)
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj()
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DDPG
alg = QuanEstimation.DDPG(max_episode=100, layer_num=4, layer_dim=250, rng=StableRNG(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj()
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj()
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj()
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: NM
alg = QuanEstimation.NM(max_episode=100, state_num=10, ar=1.0, ae=2.0, ac=0.5, as0=0.5,rng=MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj()
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

