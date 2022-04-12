using Random
using StableRNGs
include("../src/QuanEstimation.jl")

# Kraus operators for the generalized qubit amplitude damping
n, p = 0.1, 0.1
ψ0 = [1., 0]
ψ1 = [0., 1]
K0 =sqrt(1-n)*(ψ0*ψ0' + sqrt(1-p)*ψ1*ψ1')
K1 = sqrt(p-p*n)*ψ0*ψ1'
K2 = sqrt(n)*(sqrt(1-p)*ψ0*ψ0' + ψ1*ψ1')
K3 = sqrt(p*n)*ψ1*ψ0'
K = [K0, K1, K2, K3]

dK0_n = -0.5*(ψ0*ψ0'+sqrt(1-p)*ψ1*ψ1')/sqrt(1-n)
dK1_n = -0.5*p*ψ0*ψ1'/sqrt(p-p*n)
dK2_n = 0.5*(sqrt(1-p)*ψ0*ψ0'+ψ1*ψ1')/sqrt(n)
dK3_n = 0.5*p*ψ1*ψ0'/sqrt(p*n)
dK0_p = -0.5*sqrt(1-n)*ψ1*ψ1'/sqrt(1-p)
dK1_p = 0.5*(1-n)*ψ0*ψ1'/sqrt(p-p*n)
dK2_p = -0.5*sqrt(n)*ψ0*ψ0'/sqrt(1-p)
dK3_p = -0.5*sqrt(n)*ψ0*ψ0'/sqrt(1-p)
dK3_p = 0.5*n*ψ1*ψ0'/sqrt(p*n)
dK = [[dK0_n, dK0_p], [dK1_n, dK1_p], [dK2_n, dK2_p], [dK3_n, dK3_p]]

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
