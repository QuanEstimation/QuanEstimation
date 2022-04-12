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

# measurement
dim = size(K[1],1)
POVM_basis = QuanEstimation.SIC(dim)
M_num = dim

# Measurement optimization -- Projection
opt = QuanEstimation.Mopt()

# Measurement optimization -- Linear combination 
opt = QuanEstimation.Mopt(method=:LinearCombination, POVM_basis=POVM_basis, M_num=M_num)

# Measurement optimization -- Rotation
opt = QuanEstimation.Mopt(method=:Rotation, POVM_basis=POVM_basis)

# initial state
ρ₀ = 0.5*ones(2,2)

# dynamics
dynamics = QuanEstimation.Kraus(opt, ρ₀, K, dK)

# set objective
obj = QuanEstimation.CFIM_Obj()

# control algorithm: AD
alg = QuanEstimation.AD(Adam=true, max_episode=50, ϵ=0.01, beta1=0.90, beta2=0.99)
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: NM
alg = QuanEstimation.NM(max_episode=100, state_num=10, ar=1.0, ae=2.0, ac=0.5, as0=0.5,rng=MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

