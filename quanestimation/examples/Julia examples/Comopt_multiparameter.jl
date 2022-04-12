using Random
using LinearAlgebra
include("../src/QuanEstimation.jl")

# initial state
ρ₀ = zeros(ComplexF64, 6, 6)
ρ₀[1:4:5, 1:4:5].=0.5
dim = size(ρ₀, 1)

# Hamiltonian
sx = [0. 1; 1 0]
sy = [0. -im; im 0]
sz = [1. 0; 0 -1]
s1 = [0. 1 0; 1 0 1; 0 1 0] / sqrt(2)
s2 = [0. -im 0; im 0 -im; 0 im 0] / sqrt(2)
s3 = [1. 0 0; 0 0 0; 0 0 -1]
Is = I1, I2, I3 =[ kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
S = S1, S2, S3 = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
B = B1, B2, B3 = [5.0e-4, 5.0e-4, 5.0e-4]
cons = 100
D = (2pi * 2.87 * 1000) / cons
gS = (2pi * 28.03 * 1000) / cons
gI = (2pi * 4.32) / cons
A1 = (2pi * 3.65) / cons
A2 = (2pi * 3.03) / cons
H0 = sum([
    D*kron(s3^2, I(2)),
    sum(gS * B .* S),
    sum(gI * B .* Is),
    A1 * (kron(s1, sx) + kron(s2, sy)),
    A2 * kron(s3, sz)
])
dH = gS * S + gI * Is
Hc = [S1, S2, S3]

# dissipation
decay = [[S3, 2pi/cons]]

# measurement
M = [QuanEstimation.basis(dim, i)*QuanEstimation.basis(dim, i)' for i in 1:dim]

#dynamics 
tspan = range(0, 2, length=4000)

# initial control coefficients
cnum = 10
ctrl0 = [zeros(cnum) for _ in 1:length(Hc)]

# state control optimization 
opt = QuanEstimation.SCopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay)   

# control measurement optimization
opt = QuanEstimation.CMopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, ρ₀, H0, dH, Hc, decay)

# state measurement optimization
opt = QuanEstimation.SMopt()
dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, ctrl0, decay)

# # state control measurement optimization 
# opt = QuanEstimation.SCMopt()
# dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay) 
    
# control algorithm: AD
if opt isa QuanEstimation.SCopt
    alg = QuanEstimation.AD(Adam=true, max_episode=50, ϵ=0.01, beta1=0.90, beta2=0.99)
    obj = QuanEstimation.QFIM_Obj()
    obj = QuanEstimation.CFIM_Obj(M=M)
    # obj = QuanEstimation.HCRB_Obj()

    QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)
end

# control algorithm: PSO
alg = QuanEstimation.PSO(max_episode=[100, 10], p_num=10, c0=1.0, c1=2.0, c2=2.0, rng= MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj(M=M)
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)

# control algorithm: DE
alg = QuanEstimation.DE(max_episode=100, p_num=10, c=1.0, cr=0.5, rng=MersenneTwister(1234))
obj = QuanEstimation.QFIM_Obj()
obj = QuanEstimation.CFIM_Obj(M=M)
# obj = QuanEstimation.HCRB_Obj()

QuanEstimation.run(opt, alg, obj, dynamics;savefile=false)