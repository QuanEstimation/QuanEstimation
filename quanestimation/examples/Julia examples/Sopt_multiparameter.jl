using Random
using StableRNGs
using LinearAlgebra
using SparseArrays
include("../src/QuanEstimation.jl")

N = 4
# initial state
j, θ, ϕ = N÷2, 0.5pi, 0.5pi
Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
Jm = Jp'
ψ₀ = exp(0.5*θ*exp(im*ϕ)*Jm - 0.5*θ*exp(-im*ϕ)*Jp)*QuanEstimation.basis(Int(2*j+1), 1)
dim = length(ψ₀)
# free Hamiltonian
Λ = 1.0
g = 0.5
h = 0.1

Jx = 0.5*(Jp + Jm)
Jy = -0.5im*(Jp - Jm)
Jz = spdiagm(j:-1:-j)
H0 = -Λ*(Jx*Jx + g*Jy*Jy) / N + g * Jy^2 / N - h*Jz
dH = [-Λ*Jy*Jy/N, -Jz]
# measurement
M_num = N
rng = MersenneTwister(1234)
M_tp= [ComplexF64[] for _ in 1:M_num]
for i in 1:M_num
    r_ini = 2*rand(rng, dim) - ones(dim)
    r = r_ini / norm(r_ini)
    φ = 2pi*rand(rng, dim)
    M_tp[i] = [r*exp(im*φ) for (r,φ) in zip(r,φ)] 
end
M_basis = QuanEstimation.gramschmidt(M_tp)
M = [m_basis*m_basis' for  m_basis in M_basis]
# dissipation
decay = [[Jz, 0.1]]
# dynamics
tspan = range(0.0, 10.0, length=2500)

W = [1/3 0.0;0.0 2/3]

opt = QuanEstimation.Sopt(ψ₀=ψ₀)
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay)   

# control algorithm: AD
alg = QuanEstimation.AD(Adam=true, max_episode=50, ϵ=0.01, beta1=0.90, beta2=0.99)
obj = QuanEstimation.QFIM_Obj(W=W)
obj = QuanEstimation.CFIM_Obj(M=M, W=W)
# obj = QuanEstimation.HCRB_Obj(W=W)

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DDPG
alg = QuanEstimation.DDPG(max_episode=100, layer_num=4, layer_dim=250, rng=StableRNG(1234))
obj = QuanEstimation.QFIM_Obj(W=W)
obj = QuanEstimation.CFIM_Obj(M=M, W=W)
# obj = QuanEstimation.HCRB_Obj(W=W)

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj(W=W)
obj = QuanEstimation.CFIM_Obj(M=M, W=W)
# obj = QuanEstimation.HCRB_Obj(W=W)

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj(W=W)
obj = QuanEstimation.CFIM_Obj(M=M, W=W)
# obj = QuanEstimation.HCRB_Obj(W=W)

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: NM
alg = QuanEstimation.NM(max_episode=100, state_num=10, ar=1.0, ae=2.0, ac=0.5, as0=0.5,rng=MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj()
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)
