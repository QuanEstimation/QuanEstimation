using Random
using StableRNGs
using LinearAlgebra
include("../src/QuanEstimation.jl")

# initial state
ρ₀ = 0.5 * ones(2,2)
dim = size(ρ₀,1)
# Hamiltonian
ω₀ = 1.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -1.0im; 1.0im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
H0 = 0.5 * ω₀ * sz
dH = [0.5 * sz]
# measurement
M_num = dim
rng = MersenneTwister(1234)
M= [ComplexF64[] for _ in 1:M_num]
for i in 1:M_num
    r_ini = 2*rand(rng, dim) - ones(dim)
    r = r_ini / norm(r_ini)
    ϕ = 2pi*rand(rng, dim)
    M[i] = [r*exp(im*ϕ) for (r,ϕ) in zip(r,ϕ)] 
end
Measurement = QuanEstimation.gramschmidt(M)
POVM_basis = [m*m' for  m in Measurement]
# dissipation
sp = [0 1; 0 0.0im]
sm = [0 0; 1 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# dynamics
tspan = range(0.0, 10.0, length=2500)

# Measurement optimization -- Projection
opt = QuanEstimation.Mopt(C=Measurement)

# Measurement optimization -- Linear combination 
opt = QuanEstimation.Mopt(method=:LinearCombination, POVM_basis=POVM_basis, M_num=M_num)

# Measurement optimization -- Rotation
opt = QuanEstimation.Mopt(method=:Rotation, POVM_basis=POVM_basis)

# set objective
obj = QuanEstimation.CFIM_Obj()

# set dynamics
dynamics = QuanEstimation.Lindblad(opt, tspan ,ρ₀, H0, dH, decay)

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# measurement optimization algorithm: AD
alg = QuanEstimation.AD(Adam=true, max_episode=100, ϵ=0.01, beta1=0.90, beta2=0.99)
QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)